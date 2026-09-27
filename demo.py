#!/usr/bin/env python3
"""
HyperCompress Demonstration Script
==================================

Demonstrates the core capabilities of the HyperCompress library:
1. In-memory compression and decompression (compress_bytes, decompress_bytes)
2. File-based compression and decompression (compress, decompress)
3. Multi-level compression benchmarks (FAST, BALANCED, HIGH, MAXIMUM)
4. Streaming compression and decompression context managers (stream_compress, stream_decompress)

Test payload simulates multimodal data:
- Structured text (logs / JSON)
- Repetitive byte sequences
- Numerical sequences (delta-friendly data)
- High-entropy pseudo-random noise
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
import time
from typing import List, Tuple

from hypercompress import (
    compress,
    decompress,
    compress_bytes,
    decompress_bytes,
    stream_compress,
    stream_decompress,
    CompressionLevel,
)


def format_size(size_bytes: int) -> str:
    """Format byte size into human-readable representation."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def print_header(title: str) -> None:
    """Print a visually prominent section header."""
    separator = "=" * 70
    print(f"\n{separator}")
    print(f" {title.upper()}")
    print(f"{separator}")


def print_stats(
    label: str,
    original_size: int,
    compressed_size: int,
    elapsed_time: float,
    decompress_time: float | None = None,
) -> None:
    """Print compression statistics including ratio, space savings, and throughput."""
    ratio = original_size / compressed_size if compressed_size > 0 else 0.0
    savings = (1.0 - (compressed_size / original_size)) * 100.0 if original_size > 0 else 0.0
    throughput = (original_size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0.0

    print(f"[{label}]")
    print(f"  • Original Size     : {format_size(original_size)} ({original_size:,} bytes)")
    print(f"  • Compressed Size   : {format_size(compressed_size)} ({compressed_size:,} bytes)")
    print(f"  • Compression Ratio : {ratio:.2f}x")
    print(f"  • Space Savings     : {savings:.2f}%")
    print(f"  • Compress Time     : {elapsed_time * 1000:.2f} ms ({throughput:.2f} MB/s)")
    if decompress_time is not None:
        decomp_throughput = (original_size / (1024 * 1024)) / decompress_time if decompress_time > 0 else 0.0
        print(f"  • Decompress Time   : {decompress_time * 1000:.2f} ms ({decomp_throughput:.2f} MB/s)")


def generate_multimodal_data(target_size_kb: int = 1024) -> bytes:
    """
    Generate synthetic multimodal test data comprising:
    - Structured JSON and log text
    - Repeated byte patterns (run-length and dictionary friendly)
    - Sequential numerical data (delta encoding friendly)
    - Random byte sequences (high-entropy entropy challenge)
    """
    print(f"\nGenerating ~{target_size_kb} KB of synthetic multimodal test data...")

    buffer = bytearray()
    chunk_size = 256 * 1024  # 256 KB per sub-block

    # 1. Structured text logs / JSON
    text_sample = (
        '{"timestamp": 1711500000, "level": "INFO", "source": "sensor_node_01", '
        '"event": "telemetry_batch", "measurements": [23.4, 23.5, 23.6, 23.8, 24.0], '
        '"status": "OK", "tags": ["prod", "edge", "cluster_us_east"]}\n'
    ).encode("utf-8")
    text_block = text_sample * (chunk_size // len(text_sample) + 1)
    buffer.extend(text_block[:chunk_size])

    # 2. Repeated byte sequences & constant zero blocks
    repeated_block = (b"\xAA\xBB\xCC\xDD" * 8192) + (b"\x00" * 32768)
    repeated_block = repeated_block * (chunk_size // len(repeated_block) + 1)
    buffer.extend(repeated_block[:chunk_size])

    # 3. Numerical sequence (sequential integers, delta-encoding friendly)
    num_ints = chunk_size // 4
    numeric_block = bytearray()
    val = 10000
    for i in range(num_ints):
        # Gradual walk with minor deltas
        val += (i % 7) - 3
        numeric_block.extend(struct.pack("<i", val))
    buffer.extend(numeric_block[:chunk_size])

    # 4. Pseudo-random high entropy bytes
    random_block = os.urandom(chunk_size)
    buffer.extend(random_block)

    # Trim or extend to target size
    final_data = bytes(buffer[: target_size_kb * 1024])
    print(f"Generated multimodal data: {format_size(len(final_data))} ({len(final_data):,} bytes)")
    return final_data


def demo_in_memory(test_data: bytes) -> None:
    """Demonstrate in-memory compression and decompression."""
    print_header("1. In-Memory Compression (compress_bytes & decompress_bytes)")

    start_c = time.perf_counter()
    compressed = compress_bytes(test_data)
    time_c = time.perf_counter() - start_c

    start_d = time.perf_counter()
    decompressed = decompress_bytes(compressed)
    time_d = time.perf_counter() - start_d

    # Integrity verification
    assert decompressed == test_data, "Decompressed bytes do not match original!"
    print("[OK] In-memory integrity check passed: decompressed == original")

    print_stats("In-Memory Default (BALANCED)", len(test_data), len(compressed), time_c, time_d)


def demo_file_compression(test_data: bytes, temp_dir: str) -> None:
    """Demonstrate file compression and decompression."""
    print_header("2. File-Based Compression (compress & decompress)")

    input_file = os.path.join(temp_dir, "multimodal_sample.bin")
    compressed_file = os.path.join(temp_dir, "multimodal_sample.hcz")
    restored_file = os.path.join(temp_dir, "multimodal_sample.restored")

    # Write initial test file
    with open(input_file, "wb") as f:
        f.write(test_data)

    orig_hash = hashlib.sha256(test_data).hexdigest()

    # Compress file
    start_c = time.perf_counter()
    compress(input_file, compressed_file)
    time_c = time.perf_counter() - start_c

    # Decompress file
    start_d = time.perf_counter()
    decompress(compressed_file, restored_file)
    time_d = time.perf_counter() - start_d

    # Verify integrity
    with open(restored_file, "rb") as f:
        restored_data = f.read()

    restored_hash = hashlib.sha256(restored_data).hexdigest()
    assert restored_hash == orig_hash, "Restored file hash mismatch!"
    print(f"[OK] File integrity verified: SHA-256 match ({restored_hash[:16]}...)")

    orig_size = os.path.getsize(input_file)
    comp_size = os.path.getsize(compressed_file)
    print_stats("File Compression (.hcz)", orig_size, comp_size, time_c, time_d)


def demo_compression_levels(test_data: bytes) -> None:
    """Demonstrate and compare all CompressionLevel presets."""
    print_header("3. Compression Levels Comparison")

    levels: List[Tuple[str, CompressionLevel]] = [
        ("FAST", CompressionLevel.FAST),
        ("BALANCED", CompressionLevel.BALANCED),
        ("HIGH", CompressionLevel.HIGH),
        ("MAXIMUM", CompressionLevel.MAXIMUM),
    ]

    print(f"{'Level':<12} {'Level Value':<12} {'Compressed':<15} {'Ratio':<10} {'Savings':<10} {'Comp (ms)':<12} {'Decomp (ms)':<12}")
    print("-" * 85)

    for name, lvl in levels:
        # Time compression
        t0 = time.perf_counter()
        comp = compress_bytes(test_data, level=lvl)
        t_comp = (time.perf_counter() - t0) * 1000

        # Time decompression
        t1 = time.perf_counter()
        decomp = decompress_bytes(comp)
        t_decomp = (time.perf_counter() - t1) * 1000

        assert decomp == test_data, f"Level {name} round-trip verification failed!"

        ratio = len(test_data) / len(comp) if len(comp) > 0 else 0.0
        savings = (1.0 - (len(comp) / len(test_data))) * 100.0

        print(
            f"{name:<12} {lvl.value:<12} {format_size(len(comp)):<15} {ratio:.2f}x{'':<5} "
            f"{savings:6.2f}%{'':<3} {t_comp:8.2f} ms{'':<2} {t_decomp:8.2f} ms"
        )


def demo_streaming(test_data: bytes, temp_dir: str) -> None:
    """Demonstrate streaming compression and decompression using context managers."""
    print_header("4. Streaming Compression (stream_compress & stream_decompress)")

    stream_file = os.path.join(temp_dir, "streamed_archive.hcz")
    chunk_size = 64 * 1024  # 64 KB chunks

    # 1. Streaming write (compression)
    print(f"Writing stream in {format_size(chunk_size)} chunks...")
    chunks_written = 0
    t0 = time.perf_counter()
    with stream_compress(stream_file, level=CompressionLevel.BALANCED) as writer:
        for offset in range(0, len(test_data), chunk_size):
            chunk = test_data[offset : offset + chunk_size]
            writer.write(chunk)
            chunks_written += 1
    stream_comp_time = time.perf_counter() - t0

    stream_file_size = os.path.getsize(stream_file)
    print(f"[OK] Stream compression completed: {chunks_written} chunks written to {os.path.basename(stream_file)}")

    # 2. Streaming read (decompression iterator)
    print(f"Reading stream iteratively chunk-by-chunk...")
    reconstructed = bytearray()
    chunks_read = 0
    t1 = time.perf_counter()
    with stream_decompress(stream_file) as reader:
        for chunk in reader:
            reconstructed.extend(chunk)
            chunks_read += 1
    stream_decomp_time = time.perf_counter() - t1

    reconstructed_bytes = bytes(reconstructed)
    assert reconstructed_bytes == test_data, "Streamed decompression output does not match original data!"
    print(f"[OK] Stream decompression completed: {chunks_read} chunks successfully read and verified")

    print_stats(
        "Streaming Pipeline",
        len(test_data),
        stream_file_size,
        stream_comp_time,
        stream_decomp_time,
    )


def main() -> None:
    """Run all HyperCompress demonstrations."""
    print("\n" + "#" * 70)
    print("  HYPERCOMPRESS COMPREHENSIVE FEATURE DEMONSTRATION")
    print("#" * 70)

    # 1. Generate multimodal test dataset (~1.5 MB)
    multimodal_data = generate_multimodal_data(target_size_kb=1536)

    with tempfile.TemporaryDirectory() as temp_dir:
        # 2. In-memory demo
        demo_in_memory(multimodal_data)

        # 3. File-based demo
        demo_file_compression(multimodal_data, temp_dir)

        # 4. Compression levels benchmark
        demo_compression_levels(multimodal_data)

        # 5. Streaming demo
        demo_streaming(multimodal_data, temp_dir)

    print("\n" + "=" * 70)
    print(" ALL HYPERCOMPRESS DEMO TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
