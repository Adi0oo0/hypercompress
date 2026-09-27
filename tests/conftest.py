import os
import json
import struct
import pytest

@pytest.fixture
def text_data():
    line = json.dumps({"timestamp": "2023-10-25T12:00:00Z", "level": "INFO", "message": "Server started successfully."}) + "\n"
    return line.encode('utf-8') * 1000

@pytest.fixture
def binary_data():
    return bytes(i % 256 for i in range(100000))

@pytest.fixture
def numeric_data():
    return b''.join(struct.pack('<i', i) for i in range(25000))

@pytest.fixture
def random_data():
    return os.urandom(100000)

@pytest.fixture
def multimodal_data(text_data, binary_data, numeric_data, random_data):
    return text_data + binary_data + numeric_data + random_data

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def input_file(tmp_dir, multimodal_data):
    path = tmp_dir / "input.dat"
    path.write_bytes(multimodal_data)
    return str(path)
