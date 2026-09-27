"""High-level public API for HyperCompress."""

import enum
import os
import io
import time
from typing import Optional, Dict, Any

from .engine import (Algorithm, Preprocess, compute_checksum, detect_data_type, select_algorithm,
                     apply_preprocess, reverse_preprocess, compress_block, decompress_block, trial_compress)
from .format import (FileHeader, ChunkMeta, write_file_header, read_file_header,
                    write_chunk, read_chunk, write_end_marker, VERSION, CHUNK_HEADER_SIZE)
from .parallel import parallel_compress, parallel_decompress
from .streaming import StreamCompressor, StreamDecompressor


class CompressionLevel(enum.IntEnum):
    FAST = 1
    BALANCED = 5
    HIGH = 7
    MAXIMUM = 9

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB


def compress(input_path: str, output_path: str, *, 
            level: int = CompressionLevel.BALANCED,
            workers: Optional[int] = None,
            chunk_size: int = DEFAULT_CHUNK_SIZE) -> Dict[str, Any]:
    """Compress a file to .hcz format.
    
    Returns dict with stats: original_size, compressed_size, ratio, time_seconds, chunks.
    """
    start = time.perf_counter()
    
    with open(input_path, 'rb') as f:
        data = f.read()
    
    original_size = len(data)
    use_trial = level >= CompressionLevel.MAXIMUM
    
    # Split into chunks
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Compress in parallel
    results = parallel_compress(chunks, level=level, workers=workers, trial=use_trial)
    
    # Deduplication pass
    dedup_table = {}  # checksum -> first chunk index
    final_results = []
    for idx, (algo, pp, compressed, orig_size, checksum) in enumerate(results):
        if checksum in dedup_table:
            # Store dedup reference
            ref_data = dedup_table[checksum].to_bytes(4, 'little')
            final_results.append((0, 0, ref_data, orig_size, checksum, True))  # is_dedup=True
        else:
            dedup_table[checksum] = idx
            final_results.append((algo, pp, compressed, orig_size, checksum, False))
    
    # Write output file
    compressed_size = 0
    with open(output_path, 'wb') as fp:
        header = FileHeader(version=VERSION, flags=1 if len(dedup_table) < len(results) else 0,
                          original_size=original_size, chunk_size=chunk_size)
        write_file_header(fp, header)
        
        for algo, pp, comp_data, orig_size, checksum, is_dedup in final_results:
            flags = 0x01 if is_dedup else 0x00
            meta = ChunkMeta(flags=flags, algorithm=int(algo), preprocess=int(pp),
                           original_size=orig_size, compressed_size=len(comp_data), checksum=checksum)
            write_chunk(fp, comp_data, meta)
            compressed_size += CHUNK_HEADER_SIZE + len(comp_data)
        
        write_end_marker(fp)
        compressed_size += CHUNK_HEADER_SIZE  # end marker
    
    elapsed = time.perf_counter() - start
    file_size = os.path.getsize(output_path)
    return {
        'original_size': original_size,
        'compressed_size': file_size,
        'ratio': original_size / file_size if file_size > 0 else 0,
        'time_seconds': round(elapsed, 3),
        'chunks': len(chunks),
        'dedup_savings': len(results) - len(dedup_table),
    }


def decompress(input_path: str, output_path: str, *,
              workers: Optional[int] = None) -> Dict[str, Any]:
    """Decompress a .hcz file.
    
    Returns dict with stats: original_size, compressed_size, time_seconds, chunks.
    """
    start = time.perf_counter()
    compressed_size = os.path.getsize(input_path)
    
    # Read all chunks
    chunk_data_list = []  # list of (algo, pp, data) or (ref_index,) for dedup
    dedup_refs = []  # (result_index, ref_chunk_index)
    
    with open(input_path, 'rb') as fp:
        header = read_file_header(fp)
        idx = 0
        while True:
            result = read_chunk(fp)
            if result is None:
                break
            meta, data = result
            if meta.flags & 0x01:  # dedup ref
                ref_index = int.from_bytes(data[:4], 'little')
                chunk_data_list.append(None)  # placeholder
                dedup_refs.append((idx, ref_index))
            else:
                chunk_data_list.append((int(meta.algorithm), int(meta.preprocess), data))
            idx += 1
    
    # Decompress non-dedup chunks in parallel
    to_decompress = [(algo, pp, d) for item in chunk_data_list if item is not None for algo, pp, d in [item]]
    if to_decompress:
        decompressed_list = parallel_decompress(to_decompress, workers=workers)
    else:
        decompressed_list = []
    
    # Rebuild full chunk list with dedup resolution
    all_chunks = []
    decomp_idx = 0
    for i, item in enumerate(chunk_data_list):
        if item is not None:
            all_chunks.append(decompressed_list[decomp_idx])
            decomp_idx += 1
        else:
            all_chunks.append(None)  # placeholder for dedup
    
    # Resolve dedup references
    for result_idx, ref_idx in dedup_refs:
        if ref_idx < len(all_chunks) and all_chunks[ref_idx] is not None:
            all_chunks[result_idx] = all_chunks[ref_idx]
        else:
            raise ValueError(f"Invalid dedup reference: chunk {result_idx} -> {ref_idx}")
    
    # Write output
    with open(output_path, 'wb') as f:
        for chunk in all_chunks:
            f.write(chunk)
    
    original_size = sum(len(c) for c in all_chunks)
    elapsed = time.perf_counter() - start
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'time_seconds': round(elapsed, 3),
        'chunks': len(all_chunks),
    }


def compress_bytes(data: bytes, *, 
                  level: int = CompressionLevel.BALANCED,
                  workers: Optional[int] = None,
                  chunk_size: int = DEFAULT_CHUNK_SIZE) -> bytes:
    """Compress bytes in memory, returns .hcz format bytes."""
    buf = io.BytesIO()
    use_trial = level >= CompressionLevel.MAXIMUM
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    results = parallel_compress(chunks, level=level, workers=workers, trial=use_trial)
    
    # Dedup
    dedup_table = {}
    header = FileHeader(version=VERSION, flags=0, original_size=len(data), chunk_size=chunk_size)
    write_file_header(buf, header)
    
    for idx, (algo, pp, compressed, orig_size, checksum) in enumerate(results):
        if checksum in dedup_table:
            ref_data = dedup_table[checksum].to_bytes(4, 'little')
            meta = ChunkMeta(flags=0x01, algorithm=0, preprocess=0,
                           original_size=orig_size, compressed_size=len(ref_data), checksum=checksum)
            write_chunk(buf, ref_data, meta)
        else:
            dedup_table[checksum] = idx
            meta = ChunkMeta(flags=0, algorithm=int(algo), preprocess=int(pp),
                           original_size=orig_size, compressed_size=len(compressed), checksum=checksum)
            write_chunk(buf, compressed, meta)
    
    write_end_marker(buf)
    return buf.getvalue()


def decompress_bytes(data: bytes, *, workers: Optional[int] = None) -> bytes:
    """Decompress .hcz format bytes in memory."""
    buf = io.BytesIO(data)
    header = read_file_header(buf)
    
    chunks_meta = []
    dedup_refs = []
    idx = 0
    while True:
        result = read_chunk(buf)
        if result is None:
            break
        meta, chunk_data = result
        if meta.flags & 0x01:
            ref_index = int.from_bytes(chunk_data[:4], 'little')
            chunks_meta.append(None)
            dedup_refs.append((idx, ref_index))
        else:
            chunks_meta.append((int(meta.algorithm), int(meta.preprocess), chunk_data))
        idx += 1
    
    to_decompress = [item for item in chunks_meta if item is not None]
    decompressed = parallel_decompress(to_decompress, workers=workers) if to_decompress else []
    
    all_chunks = []
    d_idx = 0
    for item in chunks_meta:
        if item is not None:
            all_chunks.append(decompressed[d_idx])
            d_idx += 1
        else:
            all_chunks.append(None)
    
    for res_idx, ref_idx in dedup_refs:
        all_chunks[res_idx] = all_chunks[ref_idx]
    
    return b''.join(all_chunks)


def stream_compress(output_path: str, *,
                   level: int = CompressionLevel.BALANCED,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   workers: Optional[int] = None) -> StreamCompressor:
    """Create a streaming compressor. Use as context manager."""
    return StreamCompressor(output_path, level=level, chunk_size=chunk_size, workers=workers)


def stream_decompress(input_path: str) -> StreamDecompressor:
    """Create a streaming decompressor. Use as context manager."""
    return StreamDecompressor(input_path)
