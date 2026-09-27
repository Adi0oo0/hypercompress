"""Streaming compression and decompression classes."""

from typing import Optional, Iterator
from .format import (
    FileHeader, ChunkMeta, write_file_header, read_file_header,
    write_chunk, read_chunk, write_end_marker, VERSION, CHUNK_HEADER_SIZE
)

class StreamCompressor:
    """Context manager for streaming compression to a .hcz file."""
    
    def __init__(self, filepath: str, level: int = 5, chunk_size: int = 4*1024*1024, workers: Optional[int] = None):
        self.filepath = filepath
        self.level = level
        self.chunk_size = chunk_size  # How much data to buffer before compressing a chunk
        self.workers = workers
        self._buffer = bytearray()
        self._fp = None
        self._total_original = 0
        self._total_compressed = 0
        self._chunk_count = 0
        self._dedup_table = {}  # checksum -> chunk_index for deduplication
    
    def __enter__(self):
        self._fp = open(self.filepath, 'wb')
        # Write file header with original_size=0 (unknown for streaming)
        header = FileHeader(version=VERSION, flags=0, original_size=0, chunk_size=self.chunk_size)
        write_file_header(self._fp, header)
        return self
    
    def __exit__(self, *exc):
        self.close()
        return False
    
    def write(self, data: bytes) -> None:
        """Write data to the compressor. Data is buffered and compressed in chunks."""
        self._buffer.extend(data)
        # Flush full chunks
        while len(self._buffer) >= self.chunk_size:
            chunk = bytes(self._buffer[:self.chunk_size])
            self._buffer = self._buffer[self.chunk_size:]
            self._flush_chunk(chunk)
    
    def _flush_chunk(self, chunk: bytes) -> None:
        """Compress and write a single chunk."""
        from .engine import compute_checksum, detect_data_type, select_algorithm, apply_preprocess, compress_block, Algorithm, Preprocess
        
        checksum = compute_checksum(chunk)
        original_size = len(chunk)
        self._total_original += original_size
        
        # Deduplication: check if we've seen this chunk before
        dedup_key = checksum  # Simple dedup by checksum
        if dedup_key in self._dedup_table:
            # Write dedup reference
            ref_index = self._dedup_table[dedup_key]
            ref_data = ref_index.to_bytes(4, 'little')
            meta = ChunkMeta(flags=0x01, algorithm=0, preprocess=0, 
                           original_size=original_size, compressed_size=len(ref_data), checksum=checksum)
            write_chunk(self._fp, ref_data, meta)
            self._total_compressed += CHUNK_HEADER_SIZE + len(ref_data)
            self._chunk_count += 1
            return
        
        self._dedup_table[dedup_key] = self._chunk_count
        
        # Detect and compress
        dtype = detect_data_type(chunk)
        algo, pp = select_algorithm(dtype, self.level)
        preprocessed = apply_preprocess(chunk, pp)
        compressed = compress_block(preprocessed, algo, self.level)
        
        # Check if compression helped
        if len(compressed) >= original_size:
            compressed = chunk
            algo = Algorithm.NONE
            pp = Preprocess.NONE
        
        meta = ChunkMeta(flags=0, algorithm=int(algo), preprocess=int(pp),
                        original_size=original_size, compressed_size=len(compressed), checksum=checksum)
        write_chunk(self._fp, compressed, meta)
        self._total_compressed += CHUNK_HEADER_SIZE + len(compressed)
        self._chunk_count += 1
    
    def close(self) -> None:
        if self._fp is None:
            return
        # Flush remaining buffer
        if self._buffer:
            self._flush_chunk(bytes(self._buffer))
            self._buffer = bytearray()
        # Write end marker
        write_end_marker(self._fp)
        # Seek back and update file header with total original size
        self._fp.seek(0)
        header = FileHeader(version=VERSION, flags=1 if self._dedup_table else 0,
                          original_size=self._total_original, chunk_size=self.chunk_size)
        write_file_header(self._fp, header)
        self._fp.close()
        self._fp = None


class StreamDecompressor:
    """Context manager for streaming decompression from a .hcz file."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._fp = None
        self.header = None
        self._dedup_cache = {}  # chunk_index -> decompressed_data
        self._chunk_index = 0
    
    def __enter__(self):
        self._fp = open(self.filepath, 'rb')
        self.header = read_file_header(self._fp)
        return self
    
    def __exit__(self, *exc):
        self.close()
        return False
    
    def __iter__(self):
        """Yield decompressed chunks one at a time."""
        while True:
            result = read_chunk(self._fp)
            if result is None:
                break
            meta, data = result
            
            if meta.flags & 0x01:  # is_dedup_ref
                ref_index = int.from_bytes(data[:4], 'little')
                if ref_index in self._dedup_cache:
                    chunk = self._dedup_cache[ref_index]
                else:
                    raise ValueError(f"Dedup reference to unknown chunk {ref_index}")
            else:
                from .engine import decompress_block, reverse_preprocess, Algorithm, Preprocess
                decompressed = decompress_block(data, Algorithm(meta.algorithm))
                chunk = reverse_preprocess(decompressed, Preprocess(meta.preprocess))
            
            self._dedup_cache[self._chunk_index] = chunk
            self._chunk_index += 1
            yield chunk
    
    def read_all(self) -> bytes:
        """Read and decompress the entire file."""
        return b''.join(self)
    
    def close(self) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None
            self._dedup_cache.clear()
