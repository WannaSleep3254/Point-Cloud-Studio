"""Read BIN/PLY/PCD and sample XYZ/RGB without changing coordinate units.

8 byte header: two big-endian uint32 dimensions.
15 byte record: three little-endian float32 coordinates + three uint8 RGB.
Only this BIN layout is supported; BIN is not a universal point-cloud format.
"""
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Callable, Optional

import numpy as np


POINT_DTYPE = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])
CHUNK = 262_144


class LoadCancelled(Exception):
    pass


@dataclass(frozen=True)
class Header:
    width: int
    height: int
    size: int

    @property
    def count(self):
        return self.width * self.height


@dataclass
class Cloud:
    path: Path
    header: Header
    xyz: np.ndarray
    rgb: np.ndarray
    valid_count: int
    zero_count: int
    nonfinite_count: int
    bounds: np.ndarray
    has_rgb: bool = True


def read_header(path):
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        data = f.read(8)
    if len(data) != 8:
        raise ValueError("BIN 헤더가 잘렸습니다 (최소 8바이트 필요).")
    width, height = struct.unpack(">II", data)
    if not width or not height:
        raise ValueError("BIN 크기 정보가 0입니다.")
    expected = 8 + width * height * POINT_DTYPE.itemsize
    if size != expected:
        raise ValueError(
            f"지원하는 BIN 레코드 구조와 파일 크기가 일치하지 않습니다.\n"
            f"헤더 {width:,} × {height:,}, 예상 {expected:,} bytes, 실제 {size:,} bytes."
        )
    return Header(width, height, size)


def _masks(xyz, exclude_origin):
    finite = np.isfinite(xyz).all(axis=1)
    zero = (xyz == 0).all(axis=1)
    return finite & (~zero if exclude_origin else True), zero, ~finite


def load_cloud(path, max_points=1_000_000, exclude_origin=True,
               progress: Optional[Callable[[int], None]] = None,
               cancelled: Optional[Callable[[], bool]] = None):
    """Read BIN/PLY/PCD, retain original precision/units, sample valid point ranks."""
    if max_points < 0:
        raise ValueError("max_points must be nonnegative")
    path = Path(path).resolve()
    report = progress or (lambda _: None)
    check = cancelled or (lambda: False)
    if path.suffix.lower() in ('.ply', '.pcd'):
        from point_formats import point_records

        def check_cancelled():
            if check():
                raise LoadCancelled()

        report(0)
        with point_records(path, check_cancelled, report) as (data, width, height, has_rgb):
            header = Header(width, height, path.stat().st_size)
            return _sample_records(path, header, data, max_points, exclude_origin,
                                   lambda n: report(35 + int(n * 0.65)), check, has_rgb)
    if path.suffix.lower() != '.bin':
        raise ValueError("지원 형식: BIN, PLY, PCD")
    header = read_header(path)
    data = np.memmap(path, mode="r", offset=8, dtype=POINT_DTYPE, shape=(header.count,))
    try:
        return _sample_records(path, header, data, max_points, exclude_origin, report, check)
    finally:
        data._mmap.close()


def _sample_records(path, header, data, max_points, exclude_origin, report, check, has_rgb=True):
    valid_count = zero_count = nonfinite_count = 0
    bounds = np.array([[np.inf] * 3, [-np.inf] * 3], dtype=np.float64)
    report(0)
    for start in range(0, header.count, CHUNK):
        if check():
            raise LoadCancelled()
        xyz = data["xyz"][start:start + CHUNK]
        valid, zero, bad = _masks(xyz, exclude_origin)
        n = int(np.count_nonzero(valid))
        valid_count += n
        zero_count += int(np.count_nonzero(zero))
        nonfinite_count += int(np.count_nonzero(bad))
        if n:
            good = xyz[valid]
            bounds[0] = np.minimum(bounds[0], good.min(axis=0))
            bounds[1] = np.maximum(bounds[1], good.max(axis=0))
        report(min(45, int(45 * (start + len(xyz)) / header.count)))
    if valid_count == 0:
        raise ValueError("표시할 유효 좌표가 없습니다. 원점 제외 설정을 확인하세요.")
    shown = min(max_points, valid_count) if max_points else valid_count
    # Integer arithmetic includes both ends and cannot select duplicate ranks.
    ranks = (np.arange(shown, dtype=np.int64) * (valid_count - 1) //
             max(shown - 1, 1))
    out_xyz = np.empty((shown, 3), dtype=data.dtype["xyz"].base)
    out_rgb = np.empty((shown, 3), dtype=np.uint8)
    passed = written = 0
    for start in range(0, header.count, CHUNK):
        if check():
            raise LoadCancelled()
        block = data[start:start + CHUNK]
        valid, _, _ = _masks(block["xyz"], exclude_origin)
        indices = np.flatnonzero(valid)
        end = int(np.searchsorted(ranks, passed + len(indices), side="left"))
        if end > written:
            selected = indices[ranks[written:end] - passed]
            out_xyz[written:end] = block["xyz"][selected]
            out_rgb[written:end] = block["rgb"][selected]
        written = end
        passed += len(indices)
        report(min(99, 45 + int(54 * (start + len(block)) / header.count)))
    if written != shown or passed != valid_count:
        raise ValueError("읽는 동안 파일 내용이 바뀌었습니다. 다시 열어 주세요.")
    report(100)
    return Cloud(path, header, out_xyz, out_rgb, valid_count,
                 zero_count, nonfinite_count, bounds, has_rgb)


def write_ply(path, xyz, rgb):
    """Export precisely the provided display points with original RGB values."""
    if len(xyz) != len(rgb):
        raise ValueError("좌표와 색상 개수가 다릅니다.")
    coordinate = "double" if np.asarray(xyz).dtype.itemsize > 4 else "float"
    record_dtype = np.dtype([("xyz", "<f8" if coordinate == "double" else "<f4", (3,)), ("rgb", "u1", (3,))])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment Point Cloud Studio; original coordinate units; display sample\n"
        f"element vertex {len(xyz)}\n"
        f"property {coordinate} x\nproperty {coordinate} y\nproperty {coordinate} z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with Path(path).open("wb") as f:
        f.write(header.encode("ascii"))
        for start in range(0, len(xyz), CHUNK):
            records = np.empty(min(CHUNK, len(xyz) - start), dtype=record_dtype)
            records["xyz"] = xyz[start:start + CHUNK]
            records["rgb"] = rgb[start:start + CHUNK]
            f.write(records.tobytes())


def write_pcd(path, xyz, rgb):
    """PCD 0.7 little-endian binary XYZ + packed uint32 RGB; preserve float64."""
    if len(xyz) != len(rgb):
        raise ValueError("좌표와 색상 개수가 다릅니다.")
    size = 8 if np.asarray(xyz).dtype.itemsize > 4 else 4
    dtype = np.dtype([('xyz', '<f' + str(size), (3,)), ('rgb', '<u4')])
    header = (
        '# .PCD v0.7 - Point Cloud Data file format\n'
        '# Point Cloud Studio; original coordinate units; display sample\n'
        'VERSION .7\nFIELDS x y z rgb\n'
        f'SIZE {size} {size} {size} 4\nTYPE F F F U\nCOUNT 1 1 1 1\n'
        f'WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA binary\n'
    )
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'))
        for start in range(0, len(xyz), CHUNK):
            points = xyz[start:start + CHUNK]
            colors = np.asarray(rgb[start:start + CHUNK], dtype=np.uint32)
            records = np.empty(len(points), dtype=dtype)
            records['xyz'] = points
            records['rgb'] = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
            stream.write(records.tobytes())
