# Aero the Assistant

A fully local voice assistant using:

- **Moonshine** — speech-to-text
- **Ollama** — local LLM
- **Kokoro** — text-to-speech

Speak naturally, pause, and the assistant responds. You can also interrupt it while it is talking.

## Setup

```bash
pip install moonshine-voice kokoro-onnx sounddevice numpy
ollama serve
```

Pull a model:

```bash
ollama pull gemma3:4b-it-qat
```

Run:

```bash
python assistant.py
```

## Switch LLM Models

The LLM can be swapped without changing the voice system. Change:

```python
MODEL = "gemma3:4b-it-qat"
```

to any model installed in Ollama, for example:

```python
MODEL = "llama3.2:3b"
MODEL = "qwen2.5:3b"
MODEL = "phi4-mini"
```

See installed models with:

```bash
ollama list
```

This lets you experiment with different models for **speed, reasoning, memory, and hardware requirements** while keeping the same assistant interface.
