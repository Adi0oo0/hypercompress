# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-27

### Added
- Core compression engine with 6 algorithm backends (zstd, lz4, brotli, zlib, bz2, lzma)
- Adaptive data type detection (TEXT, NUMERIC, BINARY, HIGH_ENTROPY)
- Preprocessing transforms: delta encoding, byte shuffling (stride 4 and 8)
- Chunk-level CRC32 deduplication
- Parallel compression and decompression via multiprocessing.Pool
- Streamable .hcz binary container format
- Streaming API: StreamCompressor and StreamDecompressor context managers
- High-level API: compress(), decompress(), compress_bytes(), decompress_bytes()
- 4 compression levels: FAST, BALANCED, HIGH, MAXIMUM
- NumPy-accelerated preprocessing with pure-Python fallback
- Comprehensive test suite
- CI/CD with GitHub Actions (test matrix, linting, PyPI publishing)
