"""PLY/PCD readers staged on disk, so display sampling does not require all RAM.

PLY: ASCII and both binary byte orders, scalar vertex properties.
PCD: ASCII, little-endian binary and PCL LZF binary_compressed (field-major).
Only XYZ and optional RGB are retained. No coordinate/unit/viewpoint transform.
"""
from contextlib import contextmanager
from pathlib import Path
import struct
import tempfile

import numpy as np


RECORD = np.dtype([('xyz', '<f8', (3,)), ('rgb', 'u1', (3,))])
BLOCK = 65_536
DEFAULT_RGB = (180, 190, 205)
PLY_TYPES = dict(zip(
    ('char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double',
     'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64'),
    ('i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'f4', 'f8') * 2))


def _line(stream):
    value = stream.readline(65537)
    if not value or len(value) > 65536:
        raise ValueError('파일 헤더/행이 잘렸거나 너무 깁니다.')
    try:
        return value.decode('ascii').strip()
    except UnicodeDecodeError as exc:
        raise ValueError('ASCII 헤더/행을 읽을 수 없습니다.') from exc


def _normalize(raw):
    names = set(raw.keys()) if isinstance(raw, dict) else set(raw.dtype.names)
    n = len(raw['x'])
    records = np.empty(n, RECORD)
    for i, axis in enumerate('xyz'):
        values = np.asarray(raw[axis])
        if values.ndim != 1:
            raise ValueError('X/Y/Z는 각각 단일 값이어야 합니다.')
        records['xyz'][:, i] = values
    colors = next((keys for keys in (('red', 'green', 'blue'), ('r', 'g', 'b'))
                   if set(keys) <= names), None)
    if colors:
        for i, name in enumerate(colors):
            value = np.asarray(raw[name])
            if value.ndim != 1:
                raise ValueError('색상 채널은 단일 값이어야 합니다.')
            if value.dtype.kind == 'f':
                value = value * 255  # floating color channels are normalized [0,1]
            records['rgb'][:, i] = np.clip(np.nan_to_num(value), 0, 255).astype('u1')
    elif 'rgb' in names or 'rgba' in names:
        color = np.asarray(raw['rgb' if 'rgb' in names else 'rgba'])
        if color.ndim != 1 or color.dtype.itemsize != 4:
            raise ValueError('PCD rgb/rgba는 32비트 F 또는 U 형식이어야 합니다.')
        packed = (np.ascontiguousarray(color, dtype='<f4').view('<u4')
                  if color.dtype.kind == 'f' else color.astype('<u4'))
        records['rgb'] = np.column_stack(((packed >> 16) & 255, (packed >> 8) & 255, packed & 255))
    else:
        records['rgb'] = DEFAULT_RGB
    return records


def _has_rgb(names):
    names = set(names)
    return (set(('red', 'green', 'blue')) <= names or set('rgb') <= names
            or 'rgb' in names or 'rgba' in names)


def _check_fields(dtype):
    if not set('xyz') <= set(dtype.names):
        raise ValueError('파일에 x, y, z 좌표 필드가 필요합니다.')
    for axis in 'xyz':
        if dtype[axis].subdtype is not None:
            raise ValueError('x, y, z 필드의 COUNT는 1이어야 합니다.')


def _chunks(stream, dtype, count, ascii_mode, check):
    for start in range(0, count, BLOCK):
        check()
        n = min(BLOCK, count - start)
        if ascii_mode:
            lines = []
            while len(lines) < n:
                line = _line(stream)
                if line and not line.startswith('#'):
                    lines.append(line)
            try:
                raw = np.loadtxt(lines, dtype=dtype, comments=None, ndmin=1)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'ASCII 점 데이터의 필드 수/형식이 맞지 않습니다: {exc}') from exc
        else:
            raw = np.fromfile(stream, dtype=dtype, count=n)
        if len(raw) != n:
            raise ValueError('헤더에 기록된 점 수보다 파일 데이터가 짧습니다.')
        yield raw


def _ply_header(stream):
    if _line(stream) != 'ply':
        raise ValueError('PLY 파일 서명이 없습니다.')
    elements, encoding = [], None
    for _ in range(10000):
        line = _line(stream).split()
        if not line or line[0] in ('comment', 'obj_info'):
            continue
        if line[0] == 'end_header':
            break
        if line[0] == 'format' and len(line) == 3 and line[2] == '1.0':
            encoding = line[1]
        elif line[0] == 'element' and len(line) == 3:
            count = int(line[2])
            if count < 0:
                raise ValueError('PLY element 개수가 음수입니다.')
            elements.append([line[1], count, []])
        elif line[0] == 'property' and elements:
            if len(line) == 3 and line[1] in PLY_TYPES:
                elements[-1][2].append((line[2], PLY_TYPES[line[1]], None))
            elif len(line) == 5 and line[1] == 'list' and line[2] in PLY_TYPES and line[3] in PLY_TYPES:
                if PLY_TYPES[line[2]][0] not in 'iu':
                    raise ValueError('PLY list 개수는 정수여야 합니다.')
                elements[-1][2].append((line[4], PLY_TYPES[line[3]], PLY_TYPES[line[2]]))
            else:
                raise ValueError('지원하지 않는 PLY property 형식입니다.')
        else:
            raise ValueError('PLY 헤더 형식이 올바르지 않습니다.')
    else:
        raise ValueError('PLY end_header를 찾지 못했습니다.')
    if encoding not in ('ascii', 'binary_little_endian', 'binary_big_endian'):
        raise ValueError('PLY ASCII 또는 binary little/big endian 형식이 필요합니다.')
    vertices = [e for e in elements if e[0] == 'vertex']
    if len(vertices) != 1 or vertices[0][1] <= 0:
        raise ValueError('PLY에 유효한 vertex 목록이 없습니다.')
    return encoding, elements, vertices[0]


def _skip_ply_element(stream, element, ascii_mode, endian, check):
    _, count, props = element
    if not ascii_mode and all(p[2] is None for p in props):
        stream.seek(count * sum(np.dtype(endian + p[1]).itemsize for p in props), 1)
        return
    for i in range(count):
        if i % 4096 == 0:
            check()
        if ascii_mode:
            _line(stream)
        else:
            for _, kind, count_kind in props:
                n = 1
                if count_kind:
                    value = np.fromfile(stream, np.dtype(endian + count_kind), 1)
                    if len(value) != 1 or value[0] < 0:
                        raise ValueError('PLY list 데이터가 잘렸습니다.')
                    n = int(value[0])
                stream.seek(n * np.dtype(endian + kind).itemsize, 1)


def _pcd_header(stream):
    header = {}
    for _ in range(10000):
        fields = _line(stream).split()
        if not fields or fields[0].startswith('#'):
            continue
        key = fields[0].upper()
        if key in header:
            raise ValueError(f'PCD 헤더 항목 중복: {key}')
        header[key] = fields[1:]
        if key == 'DATA':
            break
    try:
        names = header['FIELDS']
        sizes = [int(s) for s in header['SIZE']]
        types = header['TYPE']
        counts = [int(c) for c in header.get('COUNT', ['1'] * len(names))]
        width = int(header['WIDTH'][0])
        height = int(header.get('HEIGHT', ['1'])[0])
        count = int(header.get('POINTS', [str(width * height)])[0])
        encoding = header['DATA'][0].lower()
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError('PCD 필수 헤더가 없거나 형식이 잘못되었습니다.') from exc
    if not names or len({len(names), len(sizes), len(types), len(counts)}) != 1:
        raise ValueError('PCD FIELDS/SIZE/TYPE/COUNT 개수가 다릅니다.')
    if min(width, height, count) <= 0 or width * height != count:
        raise ValueError('PCD WIDTH × HEIGHT와 POINTS가 맞지 않습니다.')
    if encoding not in ('ascii', 'binary', 'binary_compressed'):
        raise ValueError('PCD DATA는 ascii, binary, binary_compressed만 지원합니다.')
    definitions = []
    for i, (name, size, kind, n) in enumerate(zip(names, sizes, types, counts)):
        if n <= 0 or size not in (1, 2, 4, 8) or kind not in ('F', 'I', 'U') or (kind == 'F' and size not in (4, 8)):
            raise ValueError('지원하지 않는 PCD 필드 크기/형식입니다.')
        name = f'_padding_{i}' if name == '_' else name
        scalar = '<' + {'F': 'f', 'I': 'i', 'U': 'u'}[kind] + str(size)
        definitions.append((name, scalar) if n == 1 else (name, scalar, (n,)))
    try:
        dtype = np.dtype(definitions)
    except ValueError as exc:
        raise ValueError('PCD 필드 이름이 중복됩니다.') from exc
    _check_fields(dtype)
    return width, height, encoding, dtype


def _decompress_lzf(stream, output, expected, check):
    sizes = stream.read(8)
    if len(sizes) != 8:
        raise ValueError('PCD 압축 크기 헤더가 잘렸습니다.')
    compressed, uncompressed = struct.unpack('<II', sizes)
    if uncompressed != expected:
        raise ValueError('PCD 압축 해제 크기가 필드/점 수와 맞지 않습니다.')
    remaining, produced, pending = compressed, 0, bytearray()

    def take(n):
        nonlocal remaining
        if n > remaining:
            raise ValueError('PCD LZF 데이터가 잘렸습니다.')
        value = stream.read(n)
        if len(value) != n:
            raise ValueError('PCD LZF 데이터가 잘렸습니다.')
        remaining -= n
        return value

    while remaining:
        ctrl = take(1)[0]
        if ctrl < 32:
            pending.extend(take(ctrl + 1))
            produced += ctrl + 1
        else:
            length = ctrl >> 5
            if length == 7:
                length += take(1)[0]
            offset = ((ctrl & 31) << 8) + take(1)[0] + 1
            length += 2
            if offset > len(pending):
                raise ValueError('PCD LZF 역참조가 잘못되었습니다.')
            produced += length
            while length:
                n = min(length, offset)
                begin = len(pending) - offset
                pending.extend(pending[begin:begin + n])
                length -= n
        if produced > expected:
            raise ValueError('PCD LZF 해제 데이터가 예상 크기를 초과합니다.')
        if len(pending) >= BLOCK:
            check()
            output.write(pending[:-8192])
            del pending[:-8192]
    if produced != expected:
        raise ValueError('PCD LZF 해제 데이터 크기가 맞지 않습니다.')
    output.write(pending)
    output.flush()


@contextmanager
def point_records(path, check, progress):
    """Yield (normalized mmap records, width, height, has_rgb), cleaning temp files."""
    path = Path(path)
    with path.open('rb') as stream, tempfile.TemporaryFile() as staging:
        check()
        if path.suffix.lower() == '.ply':
            encoding, elements, vertex = _ply_header(stream)
            endian = '>' if encoding == 'binary_big_endian' else '<'
            ascii_mode = encoding == 'ascii'
            for element in elements:
                if element is vertex:
                    break
                _skip_ply_element(stream, element, ascii_mode, endian, check)
            if any(prop[2] is not None for prop in vertex[2]):
                raise ValueError('PLY vertex의 list 속성은 지원하지 않습니다. XYZ/RGB scalar 정점으로 저장하세요.')
            dtype = np.dtype([(name, endian + kind) for name, kind, _ in vertex[2]])
            _check_fields(dtype)
            width, height = vertex[1], 1
            chunks = _chunks(stream, dtype, width, ascii_mode, check)
            for raw in chunks:
                staging.write(_normalize(raw).tobytes())
                progress(min(34, int(35 * staging.tell() / (width * RECORD.itemsize))))
        elif path.suffix.lower() == '.pcd':
            width, height, encoding, dtype = _pcd_header(stream)
            count = width * height
            if encoding == 'binary_compressed':
                with tempfile.TemporaryFile() as unpacked:
                    _decompress_lzf(stream, unpacked, count * dtype.itemsize, check)
                    mapped = np.memmap(unpacked, mode='r', dtype='u1')
                    try:
                        for start in range(0, count, BLOCK):
                            check()
                            fields, offset = {}, 0
                            for name in dtype.names:
                                kind = dtype[name]
                                shape = (count,)
                                base = kind
                                if kind.subdtype:
                                    base, dims = kind.subdtype
                                    shape += dims
                                fields[name] = np.ndarray(shape, dtype=base, buffer=mapped, offset=offset)[start:start + BLOCK]
                                offset += count * kind.itemsize
                            staging.write(_normalize(fields).tobytes())
                            progress(min(34, int(35 * (start + len(fields['x'])) / count)))
                    finally:
                        mapped._mmap.close()
            else:
                for raw in _chunks(stream, dtype, count, encoding == 'ascii', check):
                    staging.write(_normalize(raw).tobytes())
                    progress(min(34, int(35 * staging.tell() / (count * RECORD.itemsize))))
        else:
            raise ValueError('지원하지 않는 포인트 클라우드 형식입니다.')
        staging.flush()
        mapped = np.memmap(staging, mode='r', dtype=RECORD, shape=(width * height,))
        try:
            progress(35)
            yield mapped, width, height, _has_rgb(dtype.names)
        finally:
            mapped._mmap.close()
