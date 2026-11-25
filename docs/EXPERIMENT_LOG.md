# SWhisper stage-1 — experiment log

**Date:** 2026-11-17
**Scope:** Local, white-box, text-only approximation of Sirens' Whisper §§5.1–5.2.
**Status:** Method wired and demonstrated on a **benign** fixture. No jailbreak evaluation performed.

> ⚠️ This scaffold produces candidate adversarial suffixes and measures an optimization
> loss. A low loss is **not** evidence that a model's safeguards were bypassed. There is
> no harmfulness/refusal judge in this repo, so nothing here can be labelled a "jailbreak."

---

## 1. Objective

Verify that the stage-1 search loop is implemented correctly: given a `goal` and a desired
response prefix (`target`), use white-box gradients + simulated annealing to find a short
"voiceable" suffix that lowers the combined target/semantic NLL, and confirm the model then
adopts the target prefix. Correctness of the *method*, not attack efficacy, is the goal.

## 2. Environment

| Item | Value |
|---|---|
| Hardware | 8× NVIDIA A100-SXM4-40GB (one GPU used) |
| Python | 3.12 (virtualenv at `.venv`) |
| torch | 2.15.0 + CUDA 13.2 |
| transformers | 4.57.6 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (smallest model with a usable chat template) |

Install:
```bash
virtualenv .venv && . .venv/bin/activate
pip install -e .
```

## 3. Code changes made this session

- **Refactor (behavior-preserving):** extracted the loss/index math from the nested closure
  in `main()` into a module-level `compute_losses(logits, ids, start, target_len, n, h, semantic_weight)`
  so it is unit-testable. `main()` now delegates to it. Removed an unused import.
- **Tests added under `tests/`:**
  - `test_validation.py` — no-torch error paths: non-positive args, bad temperature/weight,
    template missing `{goal}` or not ending in `{adv_suffix}`, malformed input rows.
  - `test_losses.py` — index math on synthetic logits: uniform→`log(V)`, perfect prediction→~0,
    **deliberate misalignment→large loss** (catches off-by-one slicing), `semantic_weight=0`
    isolation, batch-dim preservation.
  - `test_integration.py` — tiny-model end-to-end + model-dependent error paths
    (overwrite refusal, bad seeds, no-chat-template guard). Gated behind
    `SWHISPER_RUN_INTEGRATION=1` so the default suite needs no network.

### Test results
```
python -m unittest discover -s tests
# Ran 23 tests ... OK (skipped=5)   # 5 skipped = network-gated integration tests
```
The 5 `compute_losses` index-math tests execute and pass with torch installed, so the
target/semantic NLL slicing is verified independent of any model.

## 4. Benign demonstration run

Command:
```bash
CUDA_VISIBLE_DEVICES=0 swhisper-stage1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input examples/tasks.json --template examples/template.txt \
  --steps 150 --suffix-length 20 --top-k 256 --batch-size 64 --microbatch 64 \
  --positions 2 --semantic-weight 0.1 \
  --device cuda --dtype bfloat16 --max-new-tokens 96 --seed 0 \
  --output runs/qwen-demo.jsonl
```

Task (benign fixture):
- **goal:** "Explain how rainbows form."
- **target:** "Sure, here is an explanation of how rainbows form."

### Loss trajectory (`best_loss`)
| step | best_loss |
|---|---|
| 0 | 1.7361 |
| 20 | 1.3010 |
| 40 | 1.1953 |
| 140 | 1.1775 |
| 149 (final) | **1.1775** |

`best_loss` fell ~32% and is monotonic. `current_loss` oscillated upward mid-run
(1.19 → 1.28 → 1.18) — that is simulated annealing accepting worse neighbours to escape
local minima, which is the intended behaviour.

### Outcome
- Discovered suffix: 20 tokens of gibberish-but-ASCII-word content (overfit to this model's
  tokenizer and this one prompt; **not** a transferable phrase).
- The model's response began with the exact target line and gave a correct rainbow
  explanation. Full record in `runs/qwen-demo.jsonl`.

### Interpretation
- ✅ The gradient search + annealing + loss computation are wired correctly and drive the
  target-prefix probability up as designed.
- ❌ This is **not** a jailbreak and does not show a bypassed safeguard: the request is one
  the model answers freely, the prefix is also elicited by the template's explicit
  "Begin your response with:" line, and there is no judge measuring refusal. Low loss only
  means "the target prefix became more probable under this suffix."

## 5. Known replication gaps (unchanged from README)

Not implemented: channel robustness (TTS→distortion→ASR, §5.2.3), ultrasound generation
(§5.3), exact Cartesian-product argmin (Eq. 8), the paper's real templates/seeds/vocabulary
construction, a success/refusal judge, NR/SC metrics, baselines, cross-model transfer, and
acoustic validation. Parallel updates are *sampled* (64 proposals, 2 positions), not
exhaustive.

## 6. Next stage — authorized safety evaluation (methodology, not executed here)

Moving from "the loop runs" to "does the attack actually defeat refusals" requires an
**evaluation harness**, not changes to the search itself. The responsible, research-grade
shape of that stage:

1. **Authorized dataset only.** Standard public jailbreak benchmarks (e.g. AdvBench,
   HarmBench) are the usual yardstick in this literature. These contain explicitly harmful
   behaviours; they may only be run by a party authorized to red-team the target model, in a
   controlled, access-limited environment, and the resulting prompts/completions are
   sensitive artifacts to be contained — not published.
2. **A harmfulness/refusal judge.** Efficacy is meaningless without an independent classifier
   (or human reviewers) that labels each completion refused vs. complied. This is the missing
   piece that would let the pipeline report anything beyond a loss number.
3. **Aggregate metrics, not specimen attacks.** Report attack-success-rate (ASR), refusal
   rate, and the paper's NR/SC metrics over the set — i.e. *how well the method works*, so
   results are reproducible and reviewable without circulating working attack strings.
4. **Baselines + ablations.** Compare against no-suffix and random-suffix controls; run the
   `--semantic-weight` / `--positions` / `--top-k` sensitivity studies the README calls for.
5. **Containment & disclosure.** Treat `runs/*.jsonl` as sensitive, keep it out of version
   control, and follow coordinated-disclosure norms for any real finding against a deployed
   model.

**Scope boundary (deliberate):** this project does not generate, curate, or hand back working
jailbreak prompts or their harmful completions. The reusable deliverable is the *harness and
the aggregate metrics* an authorized evaluator computes themselves — the measurement
infrastructure, not the attack content.
