import os
import pytest
from hypercompress import stream_compress, stream_decompress
from hypercompress.format import MAGIC

class TestStreamCompressor:
    def test_streaming_roundtrip(self, tmp_dir, multimodal_data):
        filepath = str(tmp_dir / "stream.hcmp")
        with stream_compress(filepath, workers=1) as comp:
            comp.write(multimodal_data)
        
        with stream_decompress(filepath) as decomp:
            result = decomp.read_all()
        
        assert result == multimodal_data

    def test_streaming_small_chunks(self, tmp_dir, multimodal_data):
        filepath = str(tmp_dir / "stream_small.hcmp")
        with stream_compress(filepath, workers=1, chunk_size=1024) as comp:
            for i in range(0, len(multimodal_data), 1024):
                comp.write(multimodal_data[i:i+1024])
        
        with stream_decompress(filepath) as decomp:
            result = decomp.read_all()
            
        assert result == multimodal_data

    def test_streaming_single_write(self, tmp_dir, multimodal_data):
        filepath = str(tmp_dir / "stream_single.hcmp")
        with stream_compress(filepath, workers=1) as comp:
            comp.write(multimodal_data)
            
        with stream_decompress(filepath) as decomp:
            result = decomp.read_all()
            
        assert result == multimodal_data

    def test_streaming_chunk_iteration(self, tmp_dir, multimodal_data):
        filepath = str(tmp_dir / "stream_iter.hcmp")
        with stream_compress(filepath, workers=1) as comp:
            comp.write(multimodal_data)
            
        chunks = []
        with stream_decompress(filepath) as decomp:
            for chunk in decomp:
                chunks.append(chunk)
                
        result = b"".join(chunks)
        assert result == multimodal_data

    def test_streaming_creates_valid_file(self, tmp_dir, text_data):
        filepath = str(tmp_dir / "stream_valid.hcmp")
        with stream_compress(filepath, workers=1) as comp:
            comp.write(text_data)
            
        with open(filepath, 'rb') as f:
            assert f.read(4) == MAGIC

    def test_streaming_empty(self, tmp_dir):
        filepath = str(tmp_dir / "stream_empty.hcmp")
        with stream_compress(filepath, workers=1) as comp:
            pass # write nothing
            
        with stream_decompress(filepath) as decomp:
            result = decomp.read_all()
            
        assert result == b''

    def test_streaming_dedup(self, tmp_dir):
        filepath = str(tmp_dir / "stream_dedup.hcmp")
        chunk = b"x" * (4 * 1024 * 1024)
        with stream_compress(filepath, workers=1) as comp:
            comp.write(chunk)
            comp.write(chunk)
            
        assert filepath.stat().st_size < len(chunk) * 1.5 if hasattr(filepath, 'stat') else os.path.getsize(filepath) < len(chunk) * 1.5
