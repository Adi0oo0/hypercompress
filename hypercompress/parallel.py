import multiprocessing as mp
from typing import List, Tuple
from .engine import (Algorithm, Preprocess, compress_block, decompress_block, apply_preprocess, reverse_preprocess, 
                     detect_data_type, select_algorithm, trial_compress, compute_checksum)

def _compress_worker(args) -> Tuple[int, int, bytes, int, int]:
    chunk_bytes, level, trial_mode = args
    original_size = len(chunk_bytes)
    checksum = compute_checksum(chunk_bytes)
    
    if trial_mode:
        algo, pp, compressed_bytes = trial_compress(chunk_bytes, level)
    else:
        data_type = detect_data_type(chunk_bytes)
        algo, pp = select_algorithm(data_type, level)
        pp_data = apply_preprocess(chunk_bytes, pp)
        compressed_bytes = compress_block(pp_data, algo, level)
        
    if len(compressed_bytes) >= original_size:
        return Algorithm.NONE.value, Preprocess.NONE.value, chunk_bytes, original_size, checksum
        
    return algo.value, pp.value, compressed_bytes, original_size, checksum

def _decompress_worker(args) -> bytes:
    algo_int, pp_int, compressed_bytes = args
    algo = Algorithm(algo_int)
    pp = Preprocess(pp_int)
    
    decompressed = decompress_block(compressed_bytes, algo)
    return reverse_preprocess(decompressed, pp)

def parallel_compress(chunks: List[bytes], level: int = 5, workers: int = None, trial: bool = False) -> List[Tuple[Algorithm, Preprocess, bytes, int, int]]:
    if workers is None:
        workers = min(mp.cpu_count(), len(chunks)) if chunks else 1
        
    args_list = [(chunk, level, trial) for chunk in chunks]
    
    if workers <= 1 or len(chunks) <= 1:
        # Sequential processing
        results = [_compress_worker(args) for args in args_list]
    else:
        try:
            with mp.Pool(workers) as pool:
                results = pool.map(_compress_worker, args_list)
        except Exception:
            # Fallback to sequential on failure
            results = [_compress_worker(args) for args in args_list]
            
    # Convert ints back to Enums for returning
    return [(Algorithm(r[0]), Preprocess(r[1]), r[2], r[3], r[4]) for r in results]

def parallel_decompress(items: List[Tuple[int, int, bytes]], workers: int = None) -> List[bytes]:
    if workers is None:
        workers = min(mp.cpu_count(), len(items)) if items else 1
        
    if workers <= 1 or len(items) <= 1:
        # Sequential processing
        results = [_decompress_worker(item) for item in items]
    else:
        try:
            with mp.Pool(workers) as pool:
                results = pool.map(_decompress_worker, items)
        except Exception:
            # Fallback to sequential on failure
            results = [_decompress_worker(item) for item in items]
            
    return results
