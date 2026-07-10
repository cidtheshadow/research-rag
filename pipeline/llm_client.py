"""
Thin wrapper around a local Ollama server (free, no API key).

Setup (run once on your machine, not in this repo):
    1. Install Ollama: https://ollama.com/download
    2. ollama pull llama3.1        (or qwen2.5:7b, mistral, etc.)
    3. Ollama runs a local server at http://localhost:11434 automatically.

If you'd rather use a hosted free-tier API (e.g. Groq), swap the `generate`
function body for that provider's HTTP call -- everything else in the
pipeline is provider-agnostic.
"""
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "huihui_ai/qwen2.5-coder-abliterate:latest"



def generate(prompt: str, model: str = MODEL_NAME, timeout: int = 120) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()
