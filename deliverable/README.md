# Member 1 Deliverable — Prompt Research & Testing

Harmless prompt library and baseline evaluation for an authorized voice-driven LLM,
measuring **instruction-following** and **response consistency**. No attack content;
all prompts are benign and speakable (TTS-ready for Member 2).

## Contents

| File | Part | What it is |
|---|---|---|
| [`1_prompt_library.md`](1_prompt_library.md) | A | The 30 test prompts, grouped by category (human-readable) |
| [`2_expected_criteria.md`](2_expected_criteria.md) | B | The pass/fail rule for every category |
| [`3_baseline_observations.md`](3_baseline_observations.md) | C | Methodology + findings from running the library clean |
| [`result_cards/`](result_cards/) | — | One scorecard per model tested |
| [`data/`](data/) | — | Machine-readable library + raw baseline results (JSON) |

## Scope & safety

- 6 categories × 5 prompts = **30 harmless prompts**.
- Content is mundane (science/geography/cooking); nothing a model would refuse, so any
  failure is an *instruction-following* failure, not a safety refusal.
- Sourced from the **IFEval** verifiable-instruction benchmark (format/keyword/length
  constraints) plus the *structure* of prompt-injection techniques (prefix/suffix/override)
  filled with benign content. **AdvBench and other harmful sets are deliberately excluded**
  and reserved for a separate, authorization-gated evaluation harness.

## Reproduce

```bash
. .venv/bin/activate
python scripts/build_library.py                 # (re)write examples/prompt_library.json
python scripts/run_baseline.py --model <HF_ID> --out runs/baseline_<name>.json
python scripts/make_deliverable.py              # refresh library listing + result cards
```

Baseline runs use **greedy decoding, no attack**, with generation batched on one GPU
(the concurrency/speed-up). Semantic categories (suffix-robustness, reword-consistency)
are scored with sentence-embedding cosine similarity (`all-MiniLM-L6-v2`, threshold 0.6).
