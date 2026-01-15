# AGENTS.md

## Build & Run
- Install: `pip install -r requirements.txt` or `pip install -e .` (dev mode)
- Run WebUI: `python examples/web/webui.py`
- Run CLI: `python examples/cmd/run.py "Your text"`
- Run all tests: `sh tests/testall.sh`
- Run single test: `python tests/#511.py`

## Architecture
- `ChatTTS/` - Main library: `core.py` (Chat class entry point), `model/` (DVAE, GPT, Tokenizer), `config/`, `utils/`, `norm.py`
- `ChatTTS-weights/` - Model weights and YAML configs (decoder.yaml, dvae.yaml, gpt.yaml, etc.)
- `examples/` - Usage examples: `web/` (Gradio WebUI), `cmd/` (CLI), `api/` (API server), `ipynb/`, `onnx/`
- `tests/` - Issue-specific test scripts (named by issue number like `#511.py`)

## Code Style
- Python 3.11+, type hints with `typing` module (Literal, Optional, List, Dict, Union)
- Use `dataclasses` for parameter structs (e.g., InferCodeParams, RefineTextParams)
- Logging via `logging` module; avoid print statements
- PyTorch for ML; torchaudio for audio I/O at 24kHz sample rate
- Imports: stdlib first, then third-party (torch, numpy, vocos), then local (`.config`, `.model`)
- No trailing semicolons; use pathlib.Path for file paths where appropriate
