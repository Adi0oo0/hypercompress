import struct
from dataclasses import dataclass
from typing import Optional, Tuple

MAGIC = b'HCMP'
VERSION = 1
FILE_HEADER_SIZE = 32
CHUNK_HEADER_SIZE = 18
CHUNK_MARKER = 0xCBCB

@dataclass
class FileHeader:
    version: int
    flags: int
    original_size: int
    chunk_size: int

@dataclass
class ChunkMeta:
    flags: int
    algorithm: int
    preprocess: int
    original_size: int
    compressed_size: int
    checksum: int

# format string: '<4sHHQI12x' (32 bytes)
FILE_HEADER_FMT = '<4sHHQI12x'

# format string: '<HBBBxIII' (18 bytes)
CHUNK_HEADER_FMT = '<HBBBxIII'

def write_file_header(fp, header: FileHeader) -> None:
    data = struct.pack(FILE_HEADER_FMT, MAGIC, header.version, header.flags, header.original_size, header.chunk_size)
    fp.write(data)

def read_file_header(fp) -> FileHeader:
    data = fp.read(FILE_HEADER_SIZE)
    if len(data) < FILE_HEADER_SIZE:
        raise EOFError("Unexpected EOF reading file header")
    magic, version, flags, original_size, chunk_size = struct.unpack(FILE_HEADER_FMT, data)
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic}")
    return FileHeader(version=version, flags=flags, original_size=original_size, chunk_size=chunk_size)

def write_chunk(fp, data: bytes, meta: ChunkMeta) -> None:
    header = struct.pack(CHUNK_HEADER_FMT, CHUNK_MARKER, meta.flags, meta.algorithm, meta.preprocess, meta.original_size, meta.compressed_size, meta.checksum)
    fp.write(header)
    if data:
        fp.write(data)

def read_chunk(fp) -> Optional[Tuple[ChunkMeta, bytes]]:
    header_data = fp.read(CHUNK_HEADER_SIZE)
    if not header_data:
        return None
    if len(header_data) < CHUNK_HEADER_SIZE:
        raise EOFError("Unexpected EOF reading chunk header")
    marker, flags, algo, pp, orig_size, comp_size, checksum = struct.unpack(CHUNK_HEADER_FMT, header_data)
    if marker != CHUNK_MARKER:
        raise ValueError(f"Invalid chunk marker: {marker}")
        
    meta = ChunkMeta(flags=flags, algorithm=algo, preprocess=pp, original_size=orig_size, compressed_size=comp_size, checksum=checksum)
    
    if flags & 0x02: # is_last flag bit1
        return None
        
    if comp_size > 0:
        data = fp.read(comp_size)
        if len(data) < comp_size:
            raise EOFError("Unexpected EOF reading chunk data")
    else:
        data = b''
        
    return meta, data

def write_end_marker(fp) -> None:
    meta = ChunkMeta(flags=0x02, algorithm=0, preprocess=0, original_size=0, compressed_size=0, checksum=0)
    write_chunk(fp, b'', meta)
