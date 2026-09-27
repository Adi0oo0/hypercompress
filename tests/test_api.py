import os
import pytest
from hypercompress import (
    compress, decompress, compress_bytes, decompress_bytes,
    CompressionLevel
)

class TestCompressDecompress:
    @pytest.mark.slow
    def test_file_roundtrip(self, input_file, tmp_dir, multimodal_data):
        out_file = str(tmp_dir / "out.hcmp")
        restored_file = str(tmp_dir / "restored.dat")
        compress(input_file, out_file, workers=1)
        decompress(out_file, restored_file, workers=1)
        with open(restored_file, 'rb') as f:
            assert f.read() == multimodal_data

    def test_file_roundtrip_fast(self, input_file, tmp_dir, multimodal_data):
        out_file = str(tmp_dir / "out_fast.hcmp")
        restored_file = str(tmp_dir / "restored_fast.dat")
        compress(input_file, out_file, level=CompressionLevel.FAST, workers=1)
        decompress(out_file, restored_file, workers=1)
        with open(restored_file, 'rb') as f:
            assert f.read() == multimodal_data

    @pytest.mark.slow
    def test_file_roundtrip_high(self, input_file, tmp_dir, multimodal_data):
        out_file = str(tmp_dir / "out_high.hcmp")
        restored_file = str(tmp_dir / "restored_high.dat")
        compress(input_file, out_file, level=CompressionLevel.HIGH, workers=1)
        decompress(out_file, restored_file, workers=1)
        with open(restored_file, 'rb') as f:
            assert f.read() == multimodal_data

    @pytest.mark.slow
    def test_file_roundtrip_maximum(self, input_file, tmp_dir, multimodal_data):
        out_file = str(tmp_dir / "out_max.hcmp")
        restored_file = str(tmp_dir / "restored_max.dat")
        compress(input_file, out_file, level=CompressionLevel.MAXIMUM, workers=1)
        decompress(out_file, restored_file, workers=1)
        with open(restored_file, 'rb') as f:
            assert f.read() == multimodal_data

    def test_compress_returns_stats(self, input_file, tmp_dir):
        out_file = str(tmp_dir / "stats_out.hcmp")
        stats = compress(input_file, out_file, workers=1)
        assert "original_size" in stats
        assert "compressed_size" in stats
        assert "ratio" in stats
        assert "time" in stats

    def test_decompress_returns_stats(self, input_file, tmp_dir, multimodal_data):
        out_file = str(tmp_dir / "stats_out_d.hcmp")
        restored_file = str(tmp_dir / "stats_restored.dat")
        compress(input_file, out_file, workers=1)
        stats = decompress(out_file, restored_file, workers=1)
        assert "original_size" in stats
        assert "time" in stats
        assert stats["original_size"] == len(multimodal_data)

    def test_workers_1(self, input_file, tmp_dir, multimodal_data):
        out_file = str(tmp_dir / "workers_out.hcmp")
        restored_file = str(tmp_dir / "workers_restored.dat")
        compress(input_file, out_file, workers=1)
        decompress(out_file, restored_file, workers=1)
        with open(restored_file, 'rb') as f:
            assert f.read() == multimodal_data

class TestBytesApi:
    def test_bytes_roundtrip(self, multimodal_data):
        comp = compress_bytes(multimodal_data, workers=1)
        decomp = decompress_bytes(comp, workers=1)
        assert decomp == multimodal_data

    def test_bytes_roundtrip_all_levels(self, multimodal_data):
        for level in CompressionLevel:
            comp = compress_bytes(multimodal_data, level=level, workers=1)
            decomp = decompress_bytes(comp, workers=1)
            assert decomp == multimodal_data

    def test_empty_bytes(self):
        comp = compress_bytes(b'', workers=1)
        assert decompress_bytes(comp, workers=1) == b''

    def test_small_bytes(self):
        comp = compress_bytes(b'hello', workers=1)
        assert decompress_bytes(comp, workers=1) == b'hello'

    def test_text_only(self, text_data):
        comp = compress_bytes(text_data, workers=1)
        assert decompress_bytes(comp, workers=1) == text_data

    def test_random_only(self, random_data):
        comp = compress_bytes(random_data, workers=1)
        assert decompress_bytes(comp, workers=1) == random_data

class TestDeduplication:
    def test_duplicate_chunks_detected(self):
        chunk = b"A" * (4 * 1024 * 1024)
        data = chunk * 3
        comp = compress_bytes(data, workers=1)
        # Verify high compression ratio (smaller size than 2 chunks at least)
        assert len(comp) < len(chunk)
