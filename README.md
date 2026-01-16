# Visual Text Benchmark Green Agent

A green agent that evaluates a purple agent on visual text recognition tasks
across three dataset types:

- `flip_overlay` (EN/ZH/JA)
- `gradient_overlay` (EN/ZH/JA)
- `cin_fusion` (ZH only)

The green agent does not call any LLM directly. It sends the image + prompt
to the purple agent and scores the returned text against ground truth.
The purple agent is responsible for calling Gemini (or another model) and
returning plain text.

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Evaluation logic (scoring + manifest)
├─ dataset.py     # Manifest building + dataset parsing
└─ messenger.py   # A2A messaging utilities
datasets/
└─ ...            # Images + labels.jsonl (see layout below)
tests/
└─ test_agent.py  # A2A conformance tests
Dockerfile        # Docker configuration
pyproject.toml    # Python dependencies
```

## Requirements

- Python >= 3.13
- `uv` (recommended) or `pip`

## Dataset Layout

Place datasets under `datasets/` with their `labels.jsonl` files. Example:

```
datasets/
  flip_overlay/
    en/ labels.jsonl + *.png
    zh/ labels.jsonl + *.png
    ja/ labels.jsonl + *.png
  gradient_overlay/
    en/ labels.jsonl + *.png
    zh/ labels.jsonl + *.png
    ja/ labels.jsonl + *.png
  cin_fusion/
    zh/ labels.jsonl + *.png
```

## Manifest Generation

`manifest.jsonl` is generated automatically the first time you send an
evaluation request. It is written to:

```
datasets/manifest.jsonl
```

Manual generation (optional):

```bash
python -c "import sys; from pathlib import Path; sys.path.append('src'); \
from dataset import build_manifest; root=Path('datasets').resolve(); \
print(build_manifest(root, root/'manifest.jsonl'))"
```

## Prompt Versions

Set `prompt_version` in config to one of:

- `basic`
- `detailed`
- `contextual` (used by `cin_fusion` only)

You can override prompts with:

- `prompt` (global)
- `prompt_by_lang`
- `prompt_by_task`
- `prompt_by_task_lang`

## Running Locally (uv)

```bash
uv sync
uv run src/server.py
```

## Running Locally (pip)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
python src/server.py
```

## Running with Docker

The Docker image includes `datasets/` by default.

```bash
docker build -t visual-text-green-agent .
docker run -p 9009:9009 visual-text-green-agent
```

To override datasets at runtime:

```bash
docker run -p 9009:9009 -v /path/to/datasets:/home/agent/datasets visual-text-green-agent
```

## Evaluation Request (A2A)

Example payload:

```json
{
  "participants": {
    "agent": "http://localhost:9019"
  },
  "config": {
    "dataset_root": "datasets",
    "model": "gemini-2.0-flash",
    "prompt_version": "basic",
    "tasks": ["flip_overlay", "gradient_overlay", "cin_fusion"],
    "langs": ["en", "zh", "ja"],
    "shuffle": false,
    "seed": 42
  }
}
```

The green agent will:

- send image + prompt to the purple agent,
- normalize the response,
- compare with ground truth,
- return accuracy + per-task breakdown,
- write a response log under `logs/`.

### Purple Agent Expectations

The green agent sends a JSON payload that includes a `gemini` field, e.g.:

```json
{
  "id": "flip_overlay_en_00001",
  "task": "flip_overlay",
  "lang": "en",
  "gemini": {
    "model": "gemini-2.0-flash",
    "contents": [
      {
        "role": "user",
        "parts": [
          { "text": "Read the text in this image and reply with only the text. Do not add explanations or extra punctuation." },
          { "inlineData": { "mimeType": "image/png", "data": "<base64>" } }
        ]
      }
    ]
  }
}
```

The purple agent should call Gemini with `payload.gemini` and return **only the
recognized text**. The green agent does **not** require `GEMINI_API_KEY`; the
purple agent does.

## Testing

Start the server, then run:

```bash
uv sync --extra test
uv run pytest --agent-url http://localhost:9009
```

## Troubleshooting

- If `pytest` returns `502`, your HTTP proxy settings might be routing
  localhost traffic. Clear them before testing:

```bash
set HTTP_PROXY=
set HTTPS_PROXY=
set NO_PROXY=127.0.0.1,localhost
```

## Publishing

This repository includes a GitHub Actions workflow to build and publish the
Docker image to GitHub Container Registry. Configure secrets as needed in your
repository settings.
