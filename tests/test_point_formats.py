import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from cloud_io import LoadCancelled, load_cloud, write_pcd, write_ply
from point_formats import _decompress_lzf


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.xyz = np.array([[1.000000001, 2, 3], [4, 5, 6], [-7, 8, 9]], dtype=np.float64)
        self.rgb = np.array([[255, 20, 3], [4, 240, 6], [7, 8, 230]], dtype=np.uint8)

    def tearDown(self):
        self.temp.cleanup()

    def test_binary_roundtrip_ply_pcd_both_precisions(self):
        for ext, writer in (('ply', write_ply), ('pcd', write_pcd)):
            for dtype in (np.float32, np.float64):
                path = self.root / ('roundtrip.' + ext)
                xyz = self.xyz.astype(dtype)
                writer(path, xyz, self.rgb)
                cloud = load_cloud(path, 0)
                np.testing.assert_array_equal(cloud.xyz, xyz)
                np.testing.assert_array_equal(cloud.rgb, self.rgb)
                self.assertTrue(cloud.has_rgb)

    def test_ascii_ply_reordered_fields_faces_and_normals(self):
        path = self.root / 'mesh.ply'
        path.write_text('ply\nformat ascii 1.0\nelement vertex 3\n'
                        'property double z\nproperty uchar blue\nproperty float nx\n'
                        'property double x\nproperty uchar red\nproperty double y\nproperty uchar green\n'
                        'element face 1\nproperty list uchar int vertex_indices\nend_header\n'
                        '3 3 0 1.000000001 255 2 20\n6 6 1 4 4 5 240\n9 230 0 -7 7 8 8\n3 0 1 2\n')
        cloud = load_cloud(path, 0)
        np.testing.assert_array_equal(cloud.xyz, self.xyz)
        np.testing.assert_array_equal(cloud.rgb, self.rgb)

    def test_big_endian_ply_face_before_vertex(self):
        path = self.root / 'big.ply'
        data = np.zeros(3, dtype=[('xyz', '>f8', (3,))])
        data['xyz'] = self.xyz
        path.write_bytes(b'ply\nformat binary_big_endian 1.0\nelement face 1\n'
                         b'property list uchar int vertex_indices\nelement vertex 3\n'
                         b'property double x\nproperty double y\nproperty double z\nend_header\n'
                         + struct.pack('>Biii', 3, 0, 1, 2) + data.tobytes())
        cloud = load_cloud(path, 0)
        np.testing.assert_array_equal(cloud.xyz, self.xyz)
        self.assertFalse(cloud.has_rgb)
        np.testing.assert_array_equal(cloud.rgb, np.tile([180, 190, 205], (3, 1)))

    def pcd_header(self, encoding, count=3, fields='x y z rgb', sizes='8 8 8 4', types='F F F U', counts='1 1 1 1'):
        return (f'VERSION .7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\n'
                f'WIDTH {count}\nHEIGHT 1\nVIEWPOINT 10 20 30 1 0 0 0\nPOINTS {count}\nDATA {encoding}\n').encode()

    def test_ascii_pcd_float_packed_rgb_and_count_descriptor(self):
        packed = (self.rgb[:, 0].astype('u4') << 16) | (self.rgb[:, 1].astype('u4') << 8) | self.rgb[:, 2]
        packed_float = packed.view('f4')
        body = '\n'.join('1 2 3 ' + ' '.join(map(str, point)) + ' ' + repr(float(color))
                         for point, color in zip(self.xyz, packed_float))
        path = self.root / 'float.pcd'
        path.write_bytes(self.pcd_header('ascii', fields='hist x y z rgb', sizes='4 8 8 8 4',
                                        types='F F F F F', counts='3 1 1 1 1') + body.encode() + b'\n')
        cloud = load_cloud(path, 0)
        np.testing.assert_array_equal(cloud.xyz, self.xyz)
        np.testing.assert_array_equal(cloud.rgb, self.rgb)

    def test_binary_compressed_field_major_and_rgba(self):
        packed = ((self.rgb[:, 0].astype('<u4') << 16) | (self.rgb[:, 1].astype('<u4') << 8)
                  | self.rgb[:, 2] | np.uint32(0xff000000))
        raw = b''.join(np.ascontiguousarray(self.xyz[:, axis], dtype='<f8').tobytes() for axis in range(3))
        raw += packed.tobytes()
        # LZF literal runs are a valid compressed representation independent of our decoder.
        compressed = b''.join(bytes([len(raw[i:i + 32]) - 1]) + raw[i:i + 32] for i in range(0, len(raw), 32))
        path = self.root / 'compressed.pcd'
        path.write_bytes(self.pcd_header('binary_compressed', fields='x y z rgba')
                         + struct.pack('<II', len(compressed), len(raw)) + compressed)
        cloud = load_cloud(path, 0)
        np.testing.assert_array_equal(cloud.xyz, self.xyz)
        np.testing.assert_array_equal(cloud.rgb, self.rgb)

    def test_lzf_overlapping_backreference_and_truncation(self):
        compressed = b'\x02abc\x80\x02'  # literal abc, copy six bytes at offset three
        output = io.BytesIO()
        _decompress_lzf(io.BytesIO(struct.pack('<II', 6, 9) + compressed), output, 9, lambda: None)
        self.assertEqual(output.getvalue(), b'abcabcabc')
        for payload in (b'\x80\x00', b'\x03ab', b'\xe0'):
            with self.assertRaises(ValueError):
                _decompress_lzf(io.BytesIO(struct.pack('<II', len(payload), 9) + payload), io.BytesIO(), 9, lambda: None)

    def test_compressed_backreferences_across_staging_blocks(self):
        count = 20000
        expected = count * 16
        compressed = bytearray(b'\x00\x00')  # one zero literal
        remaining = expected - 1
        while remaining >= 9:
            length = min(remaining, 264)
            compressed.extend((0xe0, length - 9, 0))  # repeat byte at distance one
            remaining -= length
        if remaining:
            compressed.append(remaining - 1)
            compressed.extend(b'\x00' * remaining)
        path = self.root / 'large_compressed.pcd'
        path.write_bytes(self.pcd_header('binary_compressed', count, sizes='4 4 4 4')
                         + struct.pack('<II', len(compressed), expected) + compressed)
        cloud = load_cloud(path, 500, exclude_origin=False)
        self.assertEqual(cloud.valid_count, count)
        np.testing.assert_array_equal(cloud.xyz, np.zeros((500, 3)))
        np.testing.assert_array_equal(cloud.rgb, np.zeros((500, 3)))

    def test_sampling_origin_nonfinite_and_progress(self):
        xyz = np.vstack([self.xyz, [0, 0, 0], [np.nan, 1, 2]])
        path = self.root / 'filtered.pcd'
        write_pcd(path, xyz, np.zeros((5, 3), 'u1'))
        progress = []
        cloud = load_cloud(path, 2, progress=progress.append)
        np.testing.assert_array_equal(cloud.xyz, self.xyz[[0, 2]])
        self.assertEqual((cloud.valid_count, cloud.zero_count, cloud.nonfinite_count), (3, 1, 1))
        self.assertEqual(progress[-1], 100)
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(load_cloud(path, 0, exclude_origin=False).valid_count, 4)

    def test_cancelled_and_malformed_files(self):
        for ext, writer in (('ply', write_ply), ('pcd', write_pcd)):
            path = self.root / ('cancel.' + ext)
            writer(path, self.xyz, self.rgb)
            with self.assertRaises(LoadCancelled):
                load_cloud(path, cancelled=lambda: True)
            path.write_bytes(path.read_bytes()[:-8])
            with self.assertRaises(ValueError):
                load_cloud(path)
        path = self.root / 'missing.pcd'
        path.write_bytes(self.pcd_header('ascii', fields='x y intensity'))
        with self.assertRaises(ValueError):
            load_cloud(path)


if __name__ == '__main__':
    unittest.main()
