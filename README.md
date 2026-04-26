# SensorDoc-AI

SensorDoc-AI is a hardware-document RAG project for parsing sensor datasheets, preserving technical tables, and answering questions from retrieved context only.

The RAG pipeline now supports a provider switch with `ollama` as the default runtime and `openai` as an optional fallback.

## Features

- PDF ingestion with PyMuPDF
- Table preservation for technical datasheets
- ChromaDB vector store
- Hybrid retrieval with semantic search plus BM25
- Local-first inference with Ollama at `http://localhost:11434`
- Prompt-injection guard to block secret-exfiltration attempts and sanitize retrieved context

## Requirements

- Python 3.10+
- Ollama installed locally if using the default configuration

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want the default local model, make sure Ollama is running and the models are available:

```bash
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

## Configuration

The default configuration in [config.yaml](config.yaml) uses Ollama:

```yaml
rag_module:
  provider: "ollama"
  embedding_provider: "ollama"
  ollama:
    base_url: "http://localhost:11434"
    llm_model: "llama3"
    embedding_model: "nomic-embed-text"
```

To switch to OpenAI, change the providers in [config.yaml](config.yaml):

```yaml
rag_module:
  provider: "openai"
  embedding_provider: "openai"
```

Then set your API key in `.env`:

```bash
OPENAI_API_KEY=your_key_here
```

## Ingest Datasheets

Source PDFs are read from [Document](Document).

```bash
python rag_pipeline.py --ingest
```

This builds the ChromaDB store under `./vector_store/chroma_db`.

## Run Interactive Queries

```bash
python rag_pipeline.py
```

Example:

```python
from rag_pipeline import query_sensor_info

answer = query_sensor_info("What is the I2C address?")
print(answer)
```

## Prompt-Injection Protection

The pipeline includes a guardrail layer that:

- blocks suspicious user requests that try to reveal secrets, prompts, or environment variables
- removes suspicious instruction-like lines from retrieved context
- blocks responses that appear to contain API keys or secret names

This does not replace proper secret management. Keep credentials in environment variables and do not store them in PDFs or source files.

## Secret Management

- Never commit API keys to Git or upload them to GitHub.
- Use a `.env` file with `python-dotenv` for local development.
- In cloud environments such as AWS, Azure, or GCP, use a managed secret store like Secrets Manager or Key Vault.
- In CI/CD systems such as GitHub Actions or Jenkins, use built-in secret variables instead of hardcoding credentials.

## Notes

- If you use `openai`, `OPENAI_API_KEY` is required.
- If you use `ollama`, the project works without an OpenAI key.
- BM25 is rebuilt from the source PDFs when loading an existing vector store.