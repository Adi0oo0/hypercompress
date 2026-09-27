import bz2
import lzma
import zlib
import math
from collections import Counter
from enum import IntEnum
from typing import List, Tuple

try:
    import zstandard
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

try:
    import lz4.frame
    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False

try:
    import brotli
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class Algorithm(IntEnum):
    NONE = 0
    ZSTD = 1
    LZ4 = 2
    BROTLI = 3
    ZLIB = 4
    BZ2 = 5
    LZMA = 6

class Preprocess(IntEnum):
    NONE = 0
    DELTA = 1
    SHUFFLE_4 = 2
    SHUFFLE_8 = 3

class DataType(IntEnum):
    BINARY = 0
    TEXT = 1
    NUMERIC = 2
    HIGH_ENTROPY = 3


def compress_block(data: bytes, algo: Algorithm, level: int = 5) -> bytes:
    if algo == Algorithm.NONE:
        return data
    elif algo == Algorithm.ZSTD and _HAS_ZSTD:
        native_level = min(22, level * 2 + 1)
        compressor = zstandard.ZstdCompressor(level=native_level, write_content_size=True)
        return compressor.compress(data)
    elif algo == Algorithm.LZ4 and _HAS_LZ4:
        native_level = min(16, level * 2)
        return lz4.frame.compress(data, compression_level=native_level)
    elif algo == Algorithm.BROTLI and _HAS_BROTLI:
        native_level = min(11, level)
        return brotli.compress(data, quality=native_level)
    elif algo == Algorithm.ZLIB:
        native_level = min(9, level)
        return zlib.compress(data, native_level)
    elif algo == Algorithm.BZ2:
        native_level = min(9, level)
        return bz2.compress(data, compresslevel=native_level)
    elif algo == Algorithm.LZMA:
        native_level = min(9, level)
        return lzma.compress(data, preset=native_level)
    else:
        raise ValueError(f"Algorithm {algo} is not supported or not available.")

def decompress_block(data: bytes, algo: Algorithm) -> bytes:
    if algo == Algorithm.NONE:
        return data
    elif algo == Algorithm.ZSTD and _HAS_ZSTD:
        decompressor = zstandard.ZstdDecompressor()
        return decompressor.decompress(data)
    elif algo == Algorithm.LZ4 and _HAS_LZ4:
        return lz4.frame.decompress(data)
    elif algo == Algorithm.BROTLI and _HAS_BROTLI:
        return brotli.decompress(data)
    elif algo == Algorithm.ZLIB:
        return zlib.decompress(data)
    elif algo == Algorithm.BZ2:
        return bz2.decompress(data)
    elif algo == Algorithm.LZMA:
        return lzma.decompress(data)
    else:
        raise ValueError(f"Algorithm {algo} is not supported or not available.")


def _delta_encode(data: bytes) -> bytes:
    if not data:
        return data
    if _HAS_NUMPY:
        arr = np.frombuffer(data, dtype=np.uint8)
        result = np.empty_like(arr)
        result[0] = arr[0]  # Store first byte unchanged
        result[1:] = np.subtract(arr[1:], arr[:-1], dtype=np.uint8)
        return result.tobytes()
    else:
        res = bytearray(len(data))
        res[0] = data[0]
        for i in range(1, len(data)):
            res[i] = (data[i] - data[i-1]) & 0xFF
        return bytes(res)

def _delta_decode(data: bytes) -> bytes:
    if not data:
        return data
    if _HAS_NUMPY:
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        # arr[0] is already the original first byte
        # Remaining elements are deltas; cumsum restores originals
        np.cumsum(arr, dtype=np.uint8, out=arr)
        return arr.tobytes()
    else:
        res = bytearray(len(data))
        res[0] = data[0]
        for i in range(1, len(data)):
            res[i] = (res[i-1] + data[i]) & 0xFF
        return bytes(res)

def _shuffle_encode(data: bytes, stride: int) -> bytes:
    n = len(data)
    if n < stride:
        return data
    blocks = n // stride
    rem = n % stride
    
    if _HAS_NUMPY:
        arr = np.frombuffer(data[:n-rem], dtype=np.uint8).reshape(blocks, stride).T.flatten()
        return arr.tobytes() + data[n-rem:]
    else:
        res = bytearray(n)
        for i in range(stride):
            for j in range(blocks):
                res[i * blocks + j] = data[j * stride + i]
        if rem > 0:
            res[n-rem:] = data[n-rem:]
        return bytes(res)

def _shuffle_decode(data: bytes, stride: int) -> bytes:
    n = len(data)
    if n < stride:
        return data
    blocks = n // stride
    rem = n % stride
    
    if _HAS_NUMPY:
        arr = np.frombuffer(data[:n-rem], dtype=np.uint8).reshape(stride, blocks).T.flatten()
        return arr.tobytes() + data[n-rem:]
    else:
        res = bytearray(n)
        for i in range(blocks):
            for j in range(stride):
                res[i * stride + j] = data[j * blocks + i]
        if rem > 0:
            res[n-rem:] = data[n-rem:]
        return bytes(res)

def apply_preprocess(data: bytes, pp: Preprocess) -> bytes:
    if pp == Preprocess.NONE:
        return data
    elif pp == Preprocess.DELTA:
        return _delta_encode(data)
    elif pp == Preprocess.SHUFFLE_4:
        return _shuffle_encode(data, 4)
    elif pp == Preprocess.SHUFFLE_8:
        return _shuffle_encode(data, 8)
    else:
        raise ValueError(f"Unknown preprocess {pp}")

def reverse_preprocess(data: bytes, pp: Preprocess) -> bytes:
    if pp == Preprocess.NONE:
        return data
    elif pp == Preprocess.DELTA:
        return _delta_decode(data)
    elif pp == Preprocess.SHUFFLE_4:
        return _shuffle_decode(data, 4)
    elif pp == Preprocess.SHUFFLE_8:
        return _shuffle_decode(data, 8)
    else:
        raise ValueError(f"Unknown preprocess {pp}")


def compute_entropy(data: bytes, sample_size: int = 8192) -> float:
    if not data:
        return 0.0
    sample = data[:sample_size]
    counts = Counter(sample)
    entropy = 0.0
    length = len(sample)
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def detect_data_type(data: bytes) -> DataType:
    if not data:
        return DataType.BINARY
    
    sample = data[:8192]
    ent = compute_entropy(sample)
    
    if ent > 7.9:
        return DataType.HIGH_ENTROPY
        
    printable = sum(1 for b in sample if 32 <= b <= 126 or b in (9, 10, 13))
    if printable / len(sample) > 0.85:
        return DataType.TEXT
        
    if len(sample) >= 32:
        ent4 = compute_entropy(sample[::4])
        ent8 = compute_entropy(sample[::8])
        if ent4 < ent - 0.5 or ent8 < ent - 0.5:
            return DataType.NUMERIC
            
    return DataType.BINARY

def available_algorithms() -> List[Algorithm]:
    algos = [Algorithm.NONE, Algorithm.ZLIB, Algorithm.BZ2, Algorithm.LZMA]
    if _HAS_ZSTD:
        algos.append(Algorithm.ZSTD)
    if _HAS_LZ4:
        algos.append(Algorithm.LZ4)
    if _HAS_BROTLI:
        algos.append(Algorithm.BROTLI)
    return sorted(algos)

def select_algorithm(data_type: DataType, level: int) -> Tuple[Algorithm, Preprocess]:
    if data_type == DataType.HIGH_ENTROPY:
        return Algorithm.NONE, Preprocess.NONE
    
    if data_type == DataType.TEXT:
        if level >= 7 and _HAS_BROTLI:
            return Algorithm.BROTLI, Preprocess.NONE
        if _HAS_ZSTD:
            return Algorithm.ZSTD, Preprocess.NONE
        return Algorithm.ZLIB, Preprocess.NONE
        
    if data_type == DataType.NUMERIC:
        if _HAS_ZSTD:
            return Algorithm.ZSTD, Preprocess.SHUFFLE_4
        return Algorithm.ZLIB, Preprocess.SHUFFLE_4
        
    if data_type == DataType.BINARY:
        algo = Algorithm.ZSTD if _HAS_ZSTD else Algorithm.ZLIB
        pp = Preprocess.DELTA if level >= 5 else Preprocess.NONE
        return algo, pp
        
    return Algorithm.NONE, Preprocess.NONE

def trial_compress(data: bytes, level: int) -> Tuple[Algorithm, Preprocess, bytes]:
    data_type = detect_data_type(data)
    if data_type == DataType.HIGH_ENTROPY:
        return Algorithm.NONE, Preprocess.NONE, data
        
    best_algo = Algorithm.NONE
    best_pp = Preprocess.NONE
    best_size = len(data)
    best_compressed = data
    
    algos = available_algorithms()
    algos.remove(Algorithm.NONE)
    
    pps = [Preprocess.NONE, Preprocess.DELTA]
    if data_type == DataType.NUMERIC:
        pps.extend([Preprocess.SHUFFLE_4, Preprocess.SHUFFLE_8])
        
    for pp in pps:
        try:
            pp_data = apply_preprocess(data, pp)
            for algo in algos:
                try:
                    comp = compress_block(pp_data, algo, level)
                    if len(comp) < best_size:
                        best_size = len(comp)
                        best_algo = algo
                        best_pp = pp
                        best_compressed = comp
                except Exception:
                    pass
        except Exception:
            pass
            
    return best_algo, best_pp, best_compressed

def compute_checksum(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF
