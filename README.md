# HyperCompress

**High-performance parallel compression for multimodal data.**

HyperCompress is a Python library that compresses arbitrary data — text, images, numerical arrays, binary blobs, or any mix of them — as fast and as small as possible. It splits your data into chunks, detects what kind of data each chunk contains, applies the best preprocessing and compression algorithm for that type, deduplicates identical chunks, and does all of this across every CPU core in parallel. The result is a single streamable `.hcz` file.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
  - [compress()](#compressinput_path-output_path--level-workers-chunk_size)
  - [decompress()](#decompressinput_path-output_path--workers)
  - [compress_bytes()](#compress_bytesdata--level-workers-chunk_size)
  - [decompress_bytes()](#decompress_bytesdata--workers)
  - [stream_compress()](#stream_compressoutput_path--level-chunk_size-workers)
  - [stream_decompress()](#stream_decompressinput_path)
- [Compression Levels](#compression-levels)
- [Streaming Guide](#streaming-guide)
- [The .hcz File Format](#the-hcz-file-format)
- [Techniques Used](#techniques-used)
- [Architecture](#architecture)
- [Performance](#performance)
- [Configuration & Tuning](#configuration--tuning)
- [Dependencies](#dependencies)
- [License](#license)

---

## Installation

**Requirements:** Python 3.9+

### From source (editable / development)

```bash
cd hypercompress
pip install -e .
```

### Dependencies only

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---|---|
| `zstandard >= 0.21.0` | Primary compressor — best speed/ratio tradeoff |
| `lz4 >= 4.3.0` | Fastest compression for real-time use cases |
| `brotli >= 1.1.0` | Best ratio for text-heavy data |
| `numpy >= 1.24.0` | Vectorized preprocessing (delta encoding, byte shuffling) |

> **Note:** If any optional package is missing, HyperCompress automatically falls back to Python's built-in `zlib`, `bz2`, and `lzma`. You always have working compression — the optional packages just make it faster and smaller.

---

## Quick Start

### Compress and decompress a file

```python
from hypercompress import compress, decompress

# Compress any file → .hcz
stats = compress("dataset.bin", "dataset.hcz")
print(f"Ratio: {stats['ratio']:.1f}x, Time: {stats['time_seconds']:.3f}s")

# Decompress back
stats = decompress("dataset.hcz", "dataset_restored.bin")
```

### Compress and decompress bytes in memory

```python
from hypercompress import compress_bytes, decompress_bytes

data = open("photo.png", "rb").read()

compressed = compress_bytes(data)
original = decompress_bytes(compressed)

assert original == data  # lossless, always
```

### Stream-compress large data

```python
from hypercompress import stream_compress, stream_decompress

# Write chunks as they arrive (bounded memory)
with stream_compress("output.hcz") as writer:
    for batch in read_sensor_data():
        writer.write(batch)

# Read chunks back one at a time
with stream_decompress("output.hcz") as reader:
    for chunk in reader:
        process(chunk)
```

---

## How It Works

HyperCompress processes data through a 6-stage pipeline on every chunk:

```
Input → Chunk → Detect → Preprocess → Compress → Dedup → .hcz
```

### 1. Chunking

Your data is split into fixed-size blocks (default 4 MB). Each chunk is processed independently, enabling parallel execution across CPU cores.

### 2. Data Type Detection

A fast entropy analysis (Shannon entropy on the first 8 KB sample) classifies each chunk:

| Detected Type | Trigger | What it means |
|---|---|---|
| **TEXT** | > 85% printable ASCII | Logs, JSON, CSV, source code |
| **NUMERIC** | Byte positions at stride 4/8 have lower entropy than overall | Arrays of int32, float64, sensor readings |
| **HIGH_ENTROPY** | Entropy > 7.9 bits/byte | Already compressed (JPEG, MP4, ZIP) or encrypted |
| **BINARY** | Default | Everything else |

### 3. Preprocessing

Transforms applied before compression to expose hidden structure:

| Transform | Applied to | What it does |
|---|---|---|
| **Delta encoding** | BINARY (level ≥ 5) | Stores byte-to-byte differences. Turns slowly-changing data into near-zero sequences that compress extremely well. |
| **Byte shuffle (stride 4)** | NUMERIC | Transposes an array of 4-byte values so all first bytes are together, all second bytes together, etc. Groups similar bytes for massive compression gains on numerical arrays. |
| **Byte shuffle (stride 8)** | NUMERIC (trial mode) | Same as above for 8-byte values (float64, int64). |
| **None** | TEXT, HIGH_ENTROPY | Text compresses well as-is. High-entropy data is stored verbatim (compression would only add overhead). |

When `numpy` is installed, preprocessing runs at C speed via vectorized operations. Without it, a pure-Python fallback is used.

### 4. Algorithm Selection

Each data type is routed to the best compressor:

| Data Type | Level 1–4 (FAST) | Level 5–6 (BALANCED) | Level 7–8 (HIGH) | Level 9 (MAXIMUM) |
|---|---|---|---|---|
| TEXT | Zstandard | Zstandard | Brotli | Try all, pick smallest |
| NUMERIC | Zstandard + shuffle | Zstandard + shuffle | Zstandard + shuffle | Try all × all preprocessors |
| BINARY | Zstandard | Zstandard + delta | Zstandard + delta | Try all × all preprocessors |
| HIGH_ENTROPY | Passthrough | Passthrough | Passthrough | Passthrough |

Available backends:
- **Zstandard (zstd)** — Best general-purpose compressor. Excellent speed and ratio.
- **LZ4** — Fastest compression/decompression. Used in FAST mode.
- **Brotli** — Best ratio for text. Used in HIGH mode for text data.
- **zlib** — Fallback. Always available (Python stdlib).
- **bz2** — High ratio fallback (stdlib).
- **lzma** — Highest ratio, slowest (stdlib). Tried in MAXIMUM mode.

### 5. Deduplication

After compression, a CRC32 checksum of each original chunk is checked against all previously seen chunks. If a duplicate is found, only a 4-byte reference is stored instead of the compressed data. This is especially effective for multimodal data with repeated segments.

### 6. Parallel Execution

All chunks are compressed/decompressed in parallel using `multiprocessing.Pool`. Worker count defaults to `cpu_count()` and scales down for small inputs. If multiprocessing fails (e.g., in some restricted environments), it falls back to sequential processing automatically.

---

## API Reference

All functions are imported directly from the `hypercompress` package:

```python
from hypercompress import (
    compress, decompress,
    compress_bytes, decompress_bytes,
    stream_compress, stream_decompress,
    CompressionLevel, DEFAULT_CHUNK_SIZE,
    StreamCompressor, StreamDecompressor,
    Algorithm, Preprocess, DataType,
)
```

---

### `compress(input_path, output_path, *, level, workers, chunk_size)`

Compress a file to `.hcz` format.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_path` | `str` | *(required)* | Path to the source file |
| `output_path` | `str` | *(required)* | Path for the output `.hcz` file |
| `level` | `int` | `5` (BALANCED) | Compression level (1–9) or `CompressionLevel` enum |
| `workers` | `int \| None` | `None` (auto) | Number of parallel processes. `None` = all CPU cores |
| `chunk_size` | `int` | `4194304` (4 MB) | Size of each chunk in bytes |

**Returns:** `dict` with statistics:

```python
{
    'original_size': 10485760,      # bytes
    'compressed_size': 2621440,     # bytes (on disk)
    'ratio': 4.0,                   # original / compressed
    'time_seconds': 0.125,          # wall-clock time
    'chunks': 3,                    # number of chunks
    'dedup_savings': 0,             # duplicate chunks eliminated
}
```

**Example:**

```python
from hypercompress import compress, CompressionLevel

stats = compress(
    "training_data.bin",
    "training_data.hcz",
    level=CompressionLevel.HIGH,
    workers=4,
    chunk_size=8 * 1024 * 1024,  # 8 MB chunks
)
print(f"{stats['ratio']:.1f}x compression in {stats['time_seconds']:.2f}s")
```

---

### `decompress(input_path, output_path, *, workers)`

Decompress a `.hcz` file back to its original form.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_path` | `str` | *(required)* | Path to the `.hcz` file |
| `output_path` | `str` | *(required)* | Path for the restored file |
| `workers` | `int \| None` | `None` (auto) | Number of parallel processes |

**Returns:** `dict` with statistics:

```python
{
    'original_size': 10485760,
    'compressed_size': 2621440,
    'time_seconds': 0.045,
    'chunks': 3,
}
```

**Example:**

```python
from hypercompress import decompress

stats = decompress("data.hcz", "data_restored.bin")
print(f"Restored {stats['original_size']:,} bytes in {stats['time_seconds']:.3f}s")
```

---

### `compress_bytes(data, *, level, workers, chunk_size)`

Compress raw bytes in memory. Returns the `.hcz` format as bytes.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `bytes` | *(required)* | Raw bytes to compress |
| `level` | `int` | `5` | Compression level (1–9) |
| `workers` | `int \| None` | `None` | Parallel processes |
| `chunk_size` | `int` | `4194304` | Chunk size in bytes |

**Returns:** `bytes` — the compressed data in `.hcz` format.

**Example:**

```python
from hypercompress import compress_bytes, decompress_bytes

payload = b"sensor_reading=42.5\n" * 100_000
compressed = compress_bytes(payload, level=7)
print(f"Compressed {len(payload):,} → {len(compressed):,} bytes "
      f"({len(payload)/len(compressed):.0f}x)")
```

---

### `decompress_bytes(data, *, workers)`

Decompress `.hcz` bytes back to the original data.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `bytes` | *(required)* | `.hcz` formatted compressed bytes |
| `workers` | `int \| None` | `None` | Parallel processes |

**Returns:** `bytes` — the original uncompressed data.

---

### `stream_compress(output_path, *, level, chunk_size, workers)`

Create a streaming compressor. Returns a `StreamCompressor` context manager.

Use this when:
- Data arrives incrementally (network, sensors, logs)
- The full dataset doesn't fit in memory
- You want bounded memory usage during compression

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_path` | `str` | *(required)* | Path for the output `.hcz` file |
| `level` | `int` | `5` | Compression level (1–9) |
| `chunk_size` | `int` | `4194304` | Buffer size before flushing a compressed chunk |
| `workers` | `int \| None` | `None` | Parallel processes |

**Returns:** `StreamCompressor` (use as context manager).

**The `StreamCompressor` object:**

| Method | Description |
|---|---|
| `write(data: bytes)` | Buffer data. Automatically compresses when buffer reaches `chunk_size`. |
| `close()` | Flush remaining buffer, write end marker, finalize the file. Called automatically when exiting `with` block. |

**Example:**

```python
from hypercompress import stream_compress, CompressionLevel

with stream_compress("logs.hcz", level=CompressionLevel.FAST, chunk_size=1024*1024) as writer:
    for line in open("huge_log.txt", "rb"):
        writer.write(line)
    # .close() is called automatically
```

> **How buffering works:** You can call `write()` with any size — a single byte, a full 100 MB blob, anything. The compressor internally buffers data and flushes a compressed chunk every time the buffer reaches `chunk_size`. On `close()`, any remaining data (even if smaller than `chunk_size`) is flushed as a final chunk.

---

### `stream_decompress(input_path)`

Create a streaming decompressor. Returns a `StreamDecompressor` context manager.

Use this when:
- You want to process data chunk-by-chunk without loading everything into RAM
- You need pipeline-style decompression (read → process → discard)

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_path` | `str` | *(required)* | Path to the `.hcz` file |

**Returns:** `StreamDecompressor` (use as context manager and iterator).

**The `StreamDecompressor` object:**

| Method | Description |
|---|---|
| `__iter__()` | Yields decompressed chunks one at a time. |
| `read_all()` | Read and decompress the entire file into a single `bytes` object. |
| `close()` | Release file handle and internal caches. Called automatically when exiting `with` block. |

**Example — process chunks one at a time:**

```python
from hypercompress import stream_decompress

with stream_decompress("data.hcz") as reader:
    for i, chunk in enumerate(reader):
        print(f"Chunk {i}: {len(chunk):,} bytes")
        process(chunk)
```

**Example — decompress entire file (same as `decompress_bytes`):**

```python
with stream_decompress("data.hcz") as reader:
    all_data = reader.read_all()
```

---

## Compression Levels

Use the `CompressionLevel` enum or pass an integer 1–9:

```python
from hypercompress import CompressionLevel

CompressionLevel.FAST      # 1
CompressionLevel.BALANCED  # 5 (default)
CompressionLevel.HIGH      # 7
CompressionLevel.MAXIMUM   # 9
```

| Level | Value | Strategy | Speed | Ratio | Best For |
|---|---|---|---|---|---|
| **FAST** | 1 | Zstandard low level, no preprocessing | Fastest | Good | Real-time, networking, IPC |
| **BALANCED** | 5 | Zstandard + heuristic preprocessing + dedup | Fast | Very good | General purpose (default) |
| **HIGH** | 7 | Brotli for text, Zstandard high for others | Moderate | Excellent | Archiving, distribution |
| **MAXIMUM** | 9 | Trial all algorithms × all preprocessors, pick smallest | Slow | Best possible | Cold storage, long-term archival |

> **MAXIMUM mode** compresses each chunk with every available algorithm and every applicable preprocessing transform, then keeps whichever produced the smallest output. This is much slower but guarantees the best possible ratio.

**Benchmark (1 MB multimodal data):**

```
Level        Compressed   Ratio   Savings   Compress    Decompress
─────────────────────────────────────────────────────────────────
FAST         256.43 KB    3.99x   74.96%      2.93 ms     0.69 ms
BALANCED     256.36 KB    3.99x   74.96%      6.11 ms     1.15 ms
HIGH         259.31 KB    3.95x   74.68%     22.10 ms     6.97 ms
MAXIMUM      256.35 KB    3.99x   74.97%   1102.89 ms     8.55 ms
```

---

## Streaming Guide

### When to use streaming vs. batch

| Scenario | Use |
|---|---|
| File fits in memory | `compress()` / `decompress()` |
| Bytes already in memory | `compress_bytes()` / `decompress_bytes()` |
| Data arrives incrementally | `stream_compress()` |
| Want to process output piece by piece | `stream_decompress()` |
| Data too large for RAM | `stream_compress()` / `stream_decompress()` |

### End-to-end streaming pipeline

```python
from hypercompress import stream_compress, stream_decompress

# Producer: compress a video file in 1 MB chunks without loading it all
with stream_compress("video.hcz", chunk_size=1024*1024) as writer:
    with open("video.mp4", "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)

# Consumer: stream-decompress and forward to a network socket
import socket
sock = socket.socket()
sock.connect(("remote-host", 9000))

with stream_decompress("video.hcz") as reader:
    for chunk in reader:
        sock.sendall(chunk)

sock.close()
```

### Deduplication in streaming mode

The `StreamCompressor` automatically deduplicates chunks during streaming writes. If you write identical 4 MB blocks multiple times, only the first occurrence is compressed — subsequent duplicates store a tiny 4-byte reference. This is transparent; decompression restores the full data.

---

## The .hcz File Format

HyperCompress writes a custom binary container format (`.hcz`) designed for streaming. No seeking is required to read the file — every chunk is self-describing.

### Layout

```
┌─────────────────────────────────────────────┐
│              FILE HEADER (32 bytes)          │
│  Magic: b'HCMP'       (4 bytes)             │
│  Version:              (2 bytes, uint16)     │
│  Flags:                (2 bytes, uint16)     │
│  Original total size:  (8 bytes, uint64)     │
│  Chunk size:           (4 bytes, uint32)     │
│  Reserved:             (12 bytes)            │
├─────────────────────────────────────────────┤
│              CHUNK 0 (variable)             │
│  Marker: 0xCBCB       (2 bytes)             │
│  Flags:                (1 byte)              │
│  Algorithm ID:         (1 byte)              │
│  Preprocess ID:        (1 byte)              │
│  Reserved:             (1 byte)              │
│  Original size:        (4 bytes, uint32)     │
│  Compressed size:      (4 bytes, uint32)     │
│  CRC32 checksum:       (4 bytes, uint32)     │
│  Compressed data:      (compressed_size B)   │
├─────────────────────────────────────────────┤
│              CHUNK 1 ...                    │
├─────────────────────────────────────────────┤
│              END MARKER (18 bytes)          │
│  Marker: 0xCBCB, Flags: 0x02 (is_last)     │
│  All other fields: 0                        │
└─────────────────────────────────────────────┘
```

### Chunk flags

| Bit | Meaning |
|---|---|
| 0 | **is_dedup_ref** — compressed data is a 4-byte little-endian index referencing an earlier chunk |
| 1 | **is_last** — end-of-stream marker, no data follows |

### Why this format?

- **Fully streamable**: Each chunk has its own header; no index table at the end. You can start decompressing before the file is fully written.
- **Heterogeneous algorithms**: Different chunks can use different compressors within the same file.
- **Self-describing**: Every chunk records its algorithm, preprocessing, original size, and checksum. No external metadata needed.
- **Dedup-aware**: Duplicate chunks are replaced by tiny references inline.

---

## Techniques Used

### Delta Encoding

Computes byte-level differences: `out[i] = (data[i] - data[i-1]) & 0xFF`. Turns slowly-changing data (sensor readings, sequential IDs, gradient images) into near-zero sequences that compress dramatically. Vectorized with NumPy when available.

### Byte Shuffling (Transposition)

For arrays of N-byte values, rearranges bytes so all byte-position-0 values are together, all byte-position-1 values together, etc. For example, an array of `float32` values `[A0 A1 A2 A3 | B0 B1 B2 B3]` becomes `[A0 B0 | A1 B1 | A2 B2 | A3 B3]`. Since exponent bytes and sign bytes are often very similar across array elements, this grouping dramatically improves compression ratio on numerical data. Implemented via NumPy reshape + transpose for speed.

### Entropy-Based Detection

Shannon entropy (bits per byte, range 0–8) is computed on a sample. This instantly reveals whether data is already compressed/encrypted (entropy ~8.0, skip compression) or has exploitable structure (entropy < 7.9, worth compressing).

### Content-Aware Algorithm Routing

Rather than using one algorithm for everything, each chunk is routed to the compressor that works best for its detected type. Brotli excels on text, Zstandard on binary, and nothing beats passthrough on already-compressed data.

### Chunk-Level Deduplication

CRC32 checksums of original chunk contents are tracked. When a duplicate is found, only a 4-byte index is stored, saving potentially megabytes of redundant storage. Common in multimodal datasets with repeated headers, padding blocks, or duplicated records.

### Trial Compression (MAXIMUM mode)

All available algorithms are run on each chunk with all applicable preprocessors. The combination producing the smallest output wins. This is an exhaustive search that guarantees optimal per-chunk compression at the cost of speed.

---

## Architecture

```
hypercompress/
  __init__.py      Public API re-exports
  api.py           High-level functions (compress, decompress, stream_*)
  engine.py        Algorithms, preprocessing, data type detection
  format.py        Binary .hcz file format (headers, chunks, end marker)
  parallel.py      multiprocessing.Pool orchestration
  streaming.py     StreamCompressor, StreamDecompressor classes
```

| Module | Responsibility |
|---|---|
| **engine.py** | Contains all compression algorithm wrappers (`compress_block`, `decompress_block`), preprocessing transforms (`delta_encode/decode`, `shuffle_encode/decode`), data type detection (`detect_data_type`), algorithm selection logic (`select_algorithm`, `trial_compress`), and utility functions (`compute_entropy`, `compute_checksum`). |
| **format.py** | Defines the `.hcz` binary protocol — struct formats for the 32-byte file header and 18-byte chunk headers, plus read/write functions. Pure `struct` module, no external dependencies. |
| **parallel.py** | Wraps chunk compression/decompression in top-level functions (required for `pickle` serialization) and dispatches them via `multiprocessing.Pool.map()`. Falls back to sequential if multiprocessing fails. |
| **streaming.py** | `StreamCompressor` buffers `write()` calls and flushes compressed chunks when the buffer reaches `chunk_size`. `StreamDecompressor` iterates through chunks yielding decompressed data. Both support deduplication. |
| **api.py** | Ties everything together into the 6 public functions. Handles chunking, dedup passes, file I/O, and statistics collection. |

---

## Performance

Measured on 1 MB of synthetic multimodal data (text + repeated patterns + numeric sequences + random noise):

| Operation | Throughput |
|---|---|
| In-memory compress (BALANCED) | **134 MB/s** |
| In-memory decompress | **1,344 MB/s** |
| In-memory compress (FAST) | **342 MB/s** |
| File compress (with multiprocessing startup) | **17 MB/s** |
| Streaming compress | **38 MB/s** |
| Streaming decompress | **79 MB/s** |

> **Note:** File-based and streaming operations include multiprocessing pool startup overhead. For data larger than ~10 MB, the per-MB cost decreases significantly and approaches the in-memory throughput.

---

## Configuration & Tuning

### Chunk size

| chunk_size | Effect |
|---|---|
| Smaller (e.g. 256 KB) | More parallelism, lower latency per chunk, slightly worse ratio |
| Larger (e.g. 16 MB) | Better compression ratio, fewer chunks, less parallelism |
| Default (4 MB) | Good balance for most workloads |

```python
# Small chunks for low-latency streaming
compress("data.bin", "data.hcz", chunk_size=256 * 1024)

# Large chunks for maximum ratio
compress("data.bin", "data.hcz", chunk_size=16 * 1024 * 1024)
```

### Worker count

```python
# Use 2 workers (save CPU for other tasks)
compress("data.bin", "data.hcz", workers=2)

# Use 1 worker (disable multiprocessing entirely)
compress("data.bin", "data.hcz", workers=1)
```

### Choosing the right level

- **Real-time pipelines, IPC**: `CompressionLevel.FAST` (level 1)
- **General storage**: `CompressionLevel.BALANCED` (level 5) — the default
- **Archiving / distribution**: `CompressionLevel.HIGH` (level 7)
- **Cold storage, maximum savings**: `CompressionLevel.MAXIMUM` (level 9) — 100x slower, marginal ratio gain

---

## Dependencies

### Required (bundled in Python stdlib)

- `zlib`, `bz2`, `lzma` — fallback compressors, always available
- `struct`, `multiprocessing`, `io`, `math` — core functionality

### Recommended (pip install)

| Package | Version | Why |
|---|---|---|
| `zstandard` | >= 0.21.0 | 3–10x faster than zlib at equal or better ratio |
| `lz4` | >= 4.3.0 | Fastest decompression (>1 GB/s) |
| `brotli` | >= 1.1.0 | Best text compression ratio |
| `numpy` | >= 1.24.0 | 100x faster preprocessing via vectorized ops |

Without these packages, HyperCompress still works — it just uses slower stdlib compressors and pure-Python preprocessing.

---

## License

MIT
