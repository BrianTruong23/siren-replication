# SWhisper: stage-1 text-search scaffold

Local, white-box prompt/suffix search based on [Sirens’ Whisper, arXiv:2603.13847](https://arxiv.org/abs/2603.13847), §§5.1–5.2 and §6.1.5. Use only for models and evaluations you are authorized to test.

**This is an approximation, not an exact replication or a verified jailbreak.** It produces candidate prompts and responses; a low optimization loss is not evidence that a model's safeguards were bypassed. The included task and template are benign smoke-test fixtures, not the paper's attack templates or dataset.

## Member 1 deliverable — harmless prompt library & baseline

A documented library of **30 harmless prompts** (6 categories × 5) that measure
**instruction-following** and **response consistency** on authorized models, plus clean
baselines across three models. No attack content; prompts are benign and TTS-ready.
Full write-up and data: **[`deliverable/`](deliverable/)**.

| Document | What it is |
|---|---|
| [deliverable/README.md](deliverable/README.md) | Overview + how to reproduce |
| [1_prompt_library.md](deliverable/1_prompt_library.md) | **(A)** the 30 prompts, by category |
| [2_expected_criteria.md](deliverable/2_expected_criteria.md) | **(B)** pass rule per category |
| [3_baseline_observations.md](deliverable/3_baseline_observations.md) | **(C)** methodology + findings |
| [result_cards/](deliverable/result_cards/) | one scorecard per model |

Baseline pass rates (clean, greedy, no attack; semantic categories scored with `all-MiniLM-L6-v2`):

| Model | Overall | format | keyword | prefix | suffix | override | reword |
|---|---|---|---|---|---|---|---|
| [Llama-3.1-8B](deliverable/result_cards/meta-llama-3.1-8b-instruct.md) (ungated mirror) | **28/30 (93%)** | 5/5 | 5/5 | 4/5 | 5/5 | 4/5 | 5/5 |
| [Mistral-7B-v0.3](deliverable/result_cards/mistral-7b-instruct-v0.3.md) | **25/30 (83%)** | 5/5 | 5/5 | 5/5 | 5/5 | 1/5 | 4/5 |
| [Qwen2.5-0.5B](deliverable/result_cards/qwen2.5-0.5b-instruct.md) | **21/30 (70%)** | 2/5 | 3/5 | 2/5 | 5/5 | 4/5 | 5/5 |
| [Gemma-3-4B](deliverable/result_cards/gemma-3-4b-it.md) | not run (gated) | — | — | — | — | — | — |

Reproduce: `python scripts/run_baseline.py --model <HF_ID> --out runs/baseline_<name>.json`
then `python scripts/make_deliverable.py`. Requires the `eval` extra (`pip install -e .[eval]`).

## Install

Python 3.10+, PyTorch-compatible GPU recommended. Install a CUDA-compatible PyTorch build for your machine, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests
```

## Smoke test

```bash
swhisper-stage1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --steps 2 --top-k 16 --batch-size 4 --microbatch 1 \
  --output runs/smoke.jsonl
```

The small model is for testing wiring, not reproducing paper results. For CPU add `--device cpu --dtype float32` (slow). Downloads need network access; gated models require separately accepting the model license and Hugging Face authentication.

## Run a paper-listed surrogate

```bash
swhisper-stage1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input examples/tasks.json \
  --template examples/template.txt \
  --steps 500 --suffix-length 20 --top-k 256 --batch-size 64 \
  --microbatch 1 --output runs/qwen-stage1.jsonl
```

The paper also lists Meta-Llama-3.1-8B-Instruct, Gemma-3-4B, and Mistral-7B-Instruct-v0.3. This scaffold supports **text causal LMs with chat templates**; it does not support Gemma's multimodal conditional-generation architecture or audio-native models. GPU requirements depend on model, sequence length and microbatch; no quantization or automatic multi-GPU sharding is implemented. Start with microbatch 1 and increase only if memory permits.

### Input/output

Supply a JSON array of `{ "goal": "...", "target": "..." }` objects from your authorized evaluation set. `target` is a desired response prefix, not a complete target answer. The paper uses “Sure, here is + Rephrase(Q)”; supply this explicitly rather than silently generating it.

Templates must contain `{goal}` and end with `{adv_suffix}`; `{guiding_str}` expands to the target prefix. Optional `--seeds seeds.json` accepts a JSON array of previously obtained suffix strings. Seeds must tokenize to exactly 20 tokens (or your configured length) and pass the vocabulary filter. Without seeds, initialization is random voiceable tokens, **not** the paper's pretrained successful suffix set.

Output is JSONL, one record per task, including prompt, token IDs, suffix, model response, losses, settings, and limitations. Existing files are never overwritten. Exact `prompt_ids` are saved because token insertion at a template boundary can differ from retokenizing the displayed prompt. Do not treat the displayed prompt as a verified transfer attack.

## Method and replication gaps

| Component | Implementation |
|---|---|
| Target response objective | Mean teacher-forced target negative log likelihood |
| Semantic objective, Eq. 7 | Mean suffix next-token NLL conditioned on its context |
| Combined objective, Eq. 6 | Target NLL + `--semantic-weight` × semantic NLL |
| Gradient search | Input-embedding gradients rank replacement tokens per suffix position |
| Parallel updates | Sample a bounded batch of joint replacements at `--positions` positions |
| Annealing | Accept improving proposals; accept worsening ones with exp(-delta/T) |
| Reported settings | 20 suffix tokens, top-k 256, 64 proposals, 500 iterations |
| Voiceable vocabulary | ASCII word-token heuristic; **not** a semantic/phonetic guarantee |
| Initialization | Random or user-supplied seeds; no pretrained seed corpus included |
| Channel robustness, §5.2.3 | **Not implemented**; requires TTS → distortion → ASR evaluations |
| Ultrasound generation, §5.3 | Out of scope |

The paper describes enumerating a Cartesian product across suffix positions. Exhaustive `256^20` enumeration is infeasible; this implementation instead samples 64 proposals and defaults to updating two positions together. This is an explicit approximation, not a claimed reproduction of Eq. 8's exact argmin. Gradient steps are computed on text token IDs only.

The paper does not specify a reproducible vocabulary construction, semantic weight, temperature schedule, pretrained seed corpus, or full practical Cartesian-product search procedure. Here `--semantic-weight 0.1`, `--positions 2`, and geometric annealing from 0.1 to 0.001 are implementation choices, not reported paper settings. Use sensitivity studies before drawing conclusions. The supplied template is deliberately not Appendix B's jailbreak template.

Full stage-1 replication still requires the paper's templates/seeds, documented vocabulary construction, the chosen evaluation subset, and TTS/ASR distortion-in-the-loop scoring. Non-differentiable TTS/ASR cannot simply be backpropagated through this text gradient step; use a documented proposal/rescoring scheme and report that methodological choice. No independent refusal/harmfulness judge, NR/SC metrics, baseline comparison, cross-model transfer test, or acoustic validation is included.

## Validation

The lightweight tests check input validation and the vocabulary heuristic. A full GPU/model smoke test must be run separately; unit tests do not validate attack efficacy. Fix `--seed` for repeatable sampling, but GPU kernels/model versions may still affect reproducibility. Treat generated evaluation artifacts as potentially sensitive.
