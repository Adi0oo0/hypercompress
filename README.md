# hypercompress

Query-aware context compression for LLM applications. Given a long piece of text and a question, hypercompress keeps the parts relevant to the question and drops the rest, so you send fewer tokens to the model.

> **This is not a general-purpose data compressor.** It is lossy and question-driven. There is no `compress()`, `decompress()` or `stream_compress()`. Dropped text is not recoverable. For lossless compression use `zlib`, `gzip`, `lzma` or `zstandard`.

- **Version tested:** 0.7.3
- **Python:** 3.10+
- **License:** MIT
- **Core dependencies:** none

## Installation

```bash
pip install hypercompress                      # core only
pip install "hypercompress[server]"            # + FastAPI/uvicorn HTTP server
pip install "hypercompress[semantic]"          # + embedding tier for paraphrased questions
pip install "hypercompress[documents]"         # + PDF/Word/Excel upload support in the demo
pip install "hypercompress[mcp]"               # + MCP server for coding agents
pip install "hypercompress[exact-tokens]"      # + tiktoken for exact token counts
pip install "hypercompress[server,documents,semantic]"   # common combination
```

The plain install only gives you the core library. Importing `hypercompress.server` or `hypercompress.mcp_server` without the matching extra raises `ModuleNotFoundError`.

## Quick start

```python
from hypercompress import compress_context

result = compress_context(long_text, "What was the error code for incident INC-4821?")

print(result.compressed_text)    # send this to your LLM
print(result.tokens_saved_pct)   # e.g. 92.05
print(result.compression_risk)   # "low" | "medium" | "high"
```

Fixed budget instead of adaptive mode (keep about 30%):

```python
result = compress_context(long_text, question, 0.3)
```

### Always check `compression_risk`

When the question shares little vocabulary with the text, the engine reports `"high"` risk rather than guessing. Fall back to the original context in that case:

```python
result = compress_context(context, question)
text = context if result.compression_risk == "high" else result.compressed_text
```

## Public API

| Function / class | Purpose |
|---|---|
| `compress_context(context, question, ratio=None)` | Main entry point. Returns a `CompressionResult`. |
| `compress_context_hybrid(context, question)` | Fast lexical path, with a semantic tier for weak or paraphrased matches. |
| `compress_for_turn(messages, query=None, budget_ratio=None)` | Compresses older turns of an OpenAI/Anthropic-style message list. |
| `compact_session(transcript, current_task, budget_ratio=None)` | Compacts a long transcript for the current task. |
| `CacheStableSession(budget_ratio, drift_frac, min_history_tokens)` | Multi-turn sessions that keep the prompt prefix stable for caching. Use `.prepare(messages)`. |
| `CompressionResult` | Return type. See fields below. |

### `CompressionResult` fields

| Field | Meaning |
|---|---|
| `compressed_text` | The trimmed text, with `[...]` marking elided sections |
| `tokens_saved_pct` | Percentage of tokens removed |
| `compression_risk` | `"low"`, `"medium"` or `"high"` |
| `kept_blocks` | Audit trail: every block with its score and reasons |

## Chat message lists

```python
from hypercompress import compress_for_turn

messages = compress_for_turn(messages)
```

Rules:

- System messages pass through unchanged.
- The latest user message passes through unchanged. It is used as the query.
- Only earlier history is compressed.
- The compressed history is returned as a **single extra system message**, so the output can contain fewer messages than the input (for example, 4 in and 3 out). This is intended behavior.
- The input list is not modified.

## Multi-turn sessions with prompt caching

```python
from hypercompress import CacheStableSession

session = CacheStableSession()
messages = session.prepare(messages)
```

`session.recompressions` and `session.stable_len` show how the session is behaving.

## HTTP server

```bash
pip install "hypercompress[server,documents]"
uvicorn hypercompress.server:app --port 8765
```

| Route | Description |
|---|---|
| `GET /health` | Health check |
| `POST /api/v1/compress` | Compress text |
| `GET /demo` | Interactive step-by-step demo |
| `GET /` | Overview page |
| `GET /docs` | OpenAPI docs |

```bash
curl -X POST http://localhost:8765/api/v1/compress \
  -H "Content-Type: application/json" \
  -d '{"context": "...", "query": "what is the invoice total?"}'
```

**Note:** the HTTP field is `query`, while the Python function parameter is called `question`. Sending `question` to the server returns HTTP 422.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError` / `ImportError` on `compress`, `decompress`, `stream_compress` | These functions do not exist | Use `compress_context` (see API table) |
| `ModuleNotFoundError: fastapi` (or `mcp`) | Extra not installed | Install `[server]` or `[mcp]` |
| Server returns 422 | Sent `question` instead of `query` | Rename the field to `query` |
| Output barely shrinks | Input too small, or weak match to the question | Check `compression_risk`; try `compress_context_hybrid` |
| Semantic tier fails offline | It downloads a model from Hugging Face on first use | Allow network access to Hugging Face, or use the lexical path only |
| Fewer messages returned from `compress_for_turn` | History is merged into one system message | Expected behavior |

## Limitations

- Lossy: removed content cannot be restored.
- The core scorer is lexical (BM25 plus structural boosts). Questions phrased with none of the document's vocabulary may score poorly. The hybrid tier with `[semantic]` is designed for that case.
- Validate savings and answer retention on your own data before relying on it in production.

## Links

- Repository: https://github.com/ashy092000-cell/hypercompress
- Package: https://pypi.org/project/hypercompress/
