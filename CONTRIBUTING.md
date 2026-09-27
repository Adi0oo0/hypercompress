# Contributing to HyperCompress

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/hypercompress.git
cd hypercompress
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check hypercompress/
ruff format hypercompress/
```

## Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests for your changes
4. Ensure all tests pass: `pytest tests/ -v`
5. Ensure code passes lint: `ruff check hypercompress/`
6. Submit a pull request

## Reporting Issues

Please include:
- Python version
- OS
- Installed optional packages (zstandard, lz4, brotli, numpy)
- Minimal reproduction code
- Full traceback
