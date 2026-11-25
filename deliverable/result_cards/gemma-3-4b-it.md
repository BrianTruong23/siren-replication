# Result Card — google/gemma-3-4b-it

**Status: NOT RUN — blocked (gated model).**

| Field | Value |
|---|---|
| Decoding | greedy (planned) |
| Attack applied | none (clean baseline, planned) |
| Semantic scoring | embeddings: all-MiniLM-L6-v2 (threshold 0.6) |
| Overall | — (not run) |

## Blocker

`google/gemma-3-4b-it` is a **gated** Hugging Face repo. The download failed with:

```
OSError: You are trying to access a gated repo.
401 Client Error ... Access to model google/gemma-3-4b-it is restricted.
You must have access to it and be authenticated to access it.
```

Running it requires, from an account that owns the access:
1. Accept the Gemma license at <https://huggingface.co/google/gemma-3-4b-it>.
2. Authenticate the environment, e.g. `huggingface-cli login` or set `HF_TOKEN`.

## Note on architecture

Gemma 3 is a **multimodal** (image-text-to-text) model, not a plain text causal LM. The
baseline runner handles this (it falls back to `AutoModelForImageTextToText` for text-only
generation), but the adversarial-suffix optimizer in `swhisper_stage1.py` does **not** support
Gemma's architecture — only the harmless baseline would run here.

## Command to run once access is granted

```bash
. .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python scripts/run_baseline.py \
  --model google/gemma-3-4b-it --out runs/baseline_gemma.json --batch-size 8
python scripts/make_deliverable.py   # regenerates this card + the comparison table
```

## Ungated alternatives (~3–4B, text causal LMs) if Gemma access is not available
- `Qwen/Qwen2.5-3B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
