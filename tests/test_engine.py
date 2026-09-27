import os
import pytest
from hypercompress.engine import (
    compress_block, decompress_block, apply_preprocess, reverse_preprocess,
    detect_data_type, select_algorithm, trial_compress, compute_entropy,
    compute_checksum, available_algorithms
)
from hypercompress import Algorithm, Preprocess, DataType

def has_module(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False

class TestAlgorithms:
    def test_compress_decompress_roundtrip_zlib(self, text_data):
        compressed = compress_block(text_data, Algorithm.ZLIB)
        decompressed = decompress_block(compressed, Algorithm.ZLIB)
        assert decompressed == text_data

    def test_compress_decompress_roundtrip_bz2(self, text_data):
        compressed = compress_block(text_data, Algorithm.BZ2)
        decompressed = decompress_block(compressed, Algorithm.BZ2)
        assert decompressed == text_data

    def test_compress_decompress_roundtrip_lzma(self, text_data):
        compressed = compress_block(text_data, Algorithm.LZMA)
        decompressed = decompress_block(compressed, Algorithm.LZMA)
        assert decompressed == text_data

    @pytest.mark.skipif(not has_module('zstandard'), reason="zstandard not installed")
    def test_compress_decompress_roundtrip_zstd(self, text_data):
        compressed = compress_block(text_data, Algorithm.ZSTD)
        decompressed = decompress_block(compressed, Algorithm.ZSTD)
        assert decompressed == text_data

    @pytest.mark.skipif(not has_module('lz4'), reason="lz4 not installed")
    def test_compress_decompress_roundtrip_lz4(self, text_data):
        compressed = compress_block(text_data, Algorithm.LZ4)
        decompressed = decompress_block(compressed, Algorithm.LZ4)
        assert decompressed == text_data

    @pytest.mark.skipif(not has_module('brotli'), reason="brotli not installed")
    def test_compress_decompress_roundtrip_brotli(self, text_data):
        compressed = compress_block(text_data, Algorithm.BROTLI)
        decompressed = decompress_block(compressed, Algorithm.BROTLI)
        assert decompressed == text_data

    def test_none_algorithm_passthrough(self, text_data):
        compressed = compress_block(text_data, Algorithm.NONE)
        assert compressed == text_data
        decompressed = decompress_block(compressed, Algorithm.NONE)
        assert decompressed == text_data

    def test_empty_data(self):
        for algo in available_algorithms():
            compressed = compress_block(b'', algo)
            decompressed = decompress_block(compressed, algo)
            assert decompressed == b''

    def test_compress_actually_reduces_size(self, text_data):
        compressed = compress_block(text_data, Algorithm.ZLIB)
        assert len(compressed) < len(text_data)

class TestPreprocessing:
    def test_delta_roundtrip(self, numeric_data):
        pp = apply_preprocess(numeric_data, Preprocess.DELTA)
        rev = reverse_preprocess(pp, Preprocess.DELTA)
        assert rev == numeric_data

    def test_shuffle4_roundtrip(self, numeric_data):
        pp = apply_preprocess(numeric_data, Preprocess.SHUFFLE_4)
        rev = reverse_preprocess(pp, Preprocess.SHUFFLE_4)
        assert rev == numeric_data

    def test_shuffle8_roundtrip(self, numeric_data):
        pp = apply_preprocess(numeric_data, Preprocess.SHUFFLE_8)
        rev = reverse_preprocess(pp, Preprocess.SHUFFLE_8)
        assert rev == numeric_data

    def test_none_preprocess_passthrough(self, numeric_data):
        pp = apply_preprocess(numeric_data, Preprocess.NONE)
        assert pp == numeric_data

    def test_delta_empty_data(self):
        assert apply_preprocess(b'', Preprocess.DELTA) == b''
        assert reverse_preprocess(b'', Preprocess.DELTA) == b''

    def test_shuffle_small_data(self):
        data = b'123'
        assert reverse_preprocess(apply_preprocess(data, Preprocess.SHUFFLE_4), Preprocess.SHUFFLE_4) == data

    def test_delta_on_sequential_data(self):
        data = bytes(i % 256 for i in range(1000))
        pp = apply_preprocess(data, Preprocess.DELTA)
        assert pp.count(b'\x01') > 900

    def test_shuffle_on_structured_data(self, numeric_data):
        pp = apply_preprocess(numeric_data, Preprocess.SHUFFLE_4)
        assert len(pp) == len(numeric_data)

class TestDetection:
    def test_detect_text(self, text_data):
        assert detect_data_type(text_data) == DataType.TEXT

    def test_detect_high_entropy(self, random_data):
        assert detect_data_type(random_data) == DataType.HIGH_ENTROPY

    def test_detect_binary(self, binary_data):
        assert detect_data_type(binary_data) == DataType.BINARY

    def test_detect_numeric(self, numeric_data):
        dt = detect_data_type(numeric_data)
        assert dt in (DataType.NUMERIC, DataType.BINARY)

    def test_detect_empty(self):
        assert detect_data_type(b'') == DataType.BINARY

class TestSelection:
    def test_select_high_entropy_returns_none(self):
        assert select_algorithm(DataType.HIGH_ENTROPY, 5) == (Algorithm.NONE, Preprocess.NONE)

    def test_select_text(self):
        algo, pp = select_algorithm(DataType.TEXT, 5)
        assert algo != Algorithm.NONE
        assert pp == Preprocess.NONE

    def test_select_numeric(self):
        algo, pp = select_algorithm(DataType.NUMERIC, 5)
        assert pp == Preprocess.SHUFFLE_4

class TestUtilities:
    def test_compute_entropy_uniform(self, random_data):
        entropy = compute_entropy(random_data)
        assert entropy > 7.9

    def test_compute_entropy_constant(self):
        entropy = compute_entropy(b'A' * 10000)
        assert entropy == 0.0

    def test_compute_checksum_deterministic(self, text_data):
        c1 = compute_checksum(text_data)
        c2 = compute_checksum(text_data)
        assert c1 == c2

    def test_compute_checksum_different(self, text_data, binary_data):
        assert compute_checksum(text_data) != compute_checksum(binary_data)

    def test_available_algorithms_includes_stdlib(self):
        algos = available_algorithms()
        assert Algorithm.ZLIB in algos
        assert Algorithm.BZ2 in algos
        assert Algorithm.LZMA in algos

    def test_trial_compress_returns_valid(self, text_data):
        algo, pp, comp = trial_compress(text_data, 5)
        decomp = reverse_preprocess(decompress_block(comp, algo), pp)
        assert decomp == text_data
