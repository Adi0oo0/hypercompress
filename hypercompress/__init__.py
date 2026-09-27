"""HyperCompress - High-performance parallel compression for multimodal data."""

__version__ = "1.0.0"

from .api import (
    CompressionLevel,
    compress,
    decompress,
    compress_bytes,
    decompress_bytes,
    stream_compress,
    stream_decompress,
    DEFAULT_CHUNK_SIZE,
)
from .streaming import StreamCompressor, StreamDecompressor
from .engine import Algorithm, Preprocess, DataType

__all__ = [
    'CompressionLevel',
    'compress',
    'decompress', 
    'compress_bytes',
    'decompress_bytes',
    'stream_compress',
    'stream_decompress',
    'StreamCompressor',
    'StreamDecompressor',
    'Algorithm',
    'Preprocess',
    'DataType',
    'DEFAULT_CHUNK_SIZE',
]
