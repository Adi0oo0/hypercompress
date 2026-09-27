import io
import struct
import pytest
from hypercompress.format import (
    FileHeader, ChunkMeta, write_file_header, read_file_header,
    write_chunk, read_chunk, write_end_marker,
    MAGIC, VERSION, FILE_HEADER_SIZE, CHUNK_HEADER_SIZE
)
from hypercompress import Algorithm, Preprocess

def test_file_header_roundtrip():
    fp = io.BytesIO()
    header = FileHeader(version=VERSION, flags=0, original_size=1000, chunk_size=4096)
    write_file_header(fp, header)
    fp.seek(0)
    read_h = read_file_header(fp)
    assert read_h.version == header.version
    assert read_h.flags == header.flags
    assert read_h.original_size == header.original_size
    assert read_h.chunk_size == header.chunk_size

def test_chunk_roundtrip():
    fp = io.BytesIO()
    data = b"compresseddata"
    meta = ChunkMeta(flags=0, algorithm=Algorithm.ZLIB, preprocess=Preprocess.NONE,
                     original_size=20, compressed_size=len(data), checksum=12345)
    write_chunk(fp, data, meta)
    fp.seek(0)
    read_m, read_d = read_chunk(fp)
    assert read_m.algorithm == meta.algorithm
    assert read_m.preprocess == meta.preprocess
    assert read_m.original_size == meta.original_size
    assert read_m.compressed_size == meta.compressed_size
    assert read_m.checksum == meta.checksum
    assert read_d == data

def test_end_marker():
    fp = io.BytesIO()
    write_end_marker(fp)
    fp.seek(0)
    result = read_chunk(fp)
    assert result is None

def test_invalid_magic_raises():
    fp = io.BytesIO(b"BADM" + b"\x00" * 28)
    with pytest.raises(ValueError, match="Invalid magic"):
        read_file_header(fp)

def test_truncated_header_raises():
    fp = io.BytesIO(b"HCMP\x00")
    with pytest.raises(EOFError):
        read_file_header(fp)

def test_multiple_chunks_sequential():
    fp = io.BytesIO()
    data1, data2, data3 = b"chunk1", b"chunk2", b"chunk3"
    m1 = ChunkMeta(0, Algorithm.ZLIB, Preprocess.NONE, 10, len(data1), 1)
    m2 = ChunkMeta(0, Algorithm.LZMA, Preprocess.DELTA, 20, len(data2), 2)
    m3 = ChunkMeta(0, Algorithm.NONE, Preprocess.NONE, 30, len(data3), 3)
    
    write_chunk(fp, data1, m1)
    write_chunk(fp, data2, m2)
    write_chunk(fp, data3, m3)
    write_end_marker(fp)
    
    fp.seek(0)
    rm1, rd1 = read_chunk(fp)
    rm2, rd2 = read_chunk(fp)
    rm3, rd3 = read_chunk(fp)
    end = read_chunk(fp)
    
    assert rd1 == data1 and rm1.original_size == 10
    assert rd2 == data2 and rm2.algorithm == Algorithm.LZMA
    assert rd3 == data3
    assert end is None

def test_dedup_chunk_flag():
    fp = io.BytesIO()
    meta = ChunkMeta(flags=0x01, algorithm=Algorithm.NONE, preprocess=Preprocess.NONE,
                     original_size=0, compressed_size=0, checksum=123)
    write_chunk(fp, b"", meta)
    fp.seek(0)
    rm, rd = read_chunk(fp)
    assert rm.flags == 0x01

def test_header_size_constant():
    # Verify the hardcoded sizes match struct.calcsize behavior for safety
    assert struct.calcsize("<4sBxxxQQQ") == FILE_HEADER_SIZE
    assert struct.calcsize("<BBBxQQL") == CHUNK_HEADER_SIZE
