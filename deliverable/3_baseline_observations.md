# 3. Baseline Observations (Part C)

Baseline = every prompt run **clean, with no attack**, greedy decoding. These numbers are
the reference point that later stages (audio distortion, adversarial suffixes) are compared
against: a drop from baseline is a finding; a prompt that already fails clean is not evidence
of any attack.

## Method

- **Decoding:** greedy (`do_sample=False`), deterministic.
- **Concurrency:** prompts are generated in GPU batches (batch size 16 for Qwen, 8 for larger
  models) — this batching is the speed-up; all 40 generations (30 prompts + 5 base prompts for
  suffix-robustness + 5 extra for the reword pairs) complete in seconds on one A100.
- **Scoring:** mechanical rules per [`2_expected_criteria.md`](2_expected_criteria.md);
  semantic categories use `all-MiniLM-L6-v2` cosine similarity (threshold 0.6).
- Raw per-prompt responses and scores are in [`data/`](data/); per-model scorecards are in
  [`result_cards/`](result_cards/).

## Cross-model comparison

<!-- COMPARISON:START -->
| Model | Overall | format | keyword | prefix | suffix | override | reword |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 21/30 (70%) | 2/5 | 3/5 | 2/5 | 5/5 | 4/5 | 5/5 |
<!-- COMPARISON:END -->

> Gemma-3-4B (`google/gemma-3-4b-it`) is a **gated** model on Hugging Face and could not be
> downloaded without authenticated access + license acceptance. Its result card documents the
> blocker; rerun once access is granted (see that card for the exact command).

## Findings — Qwen2.5-0.5B-Instruct (21/30, 70%)

Strong where the instruction is about *content*, weaker where it is about *exact form* — a
pattern consistent with a small (0.5B) instruction-tuned model.

- **suffix-robustness 5/5** and **reword-consistency 5/5** — the model is semantically stable:
  benign trailing filler does not derail the answer, and paraphrased questions yield equivalent
  answers. (In `suffix-01` it explicitly acknowledged the filler words but still answered
  correctly — echoed, not derailed.)
- **instruction-override 4/5** — it generally obeys a trailing "ignore that, do X" instruction.
  The one miss (`override-02`, "list three fruits") produced neither fruits nor a clean topic
  switch.
- **format-following 2/5** — the main weakness. It missed an exact sentence count (`format-01`:
  4 vs 3), ignored a bullet-format request (`format-03`: prose instead of 2 bullets), and blew a
  word limit (`format-04`: 82 words vs ≤40). Hard numeric/format constraints are the least
  reliable.
- **prefix-injection 2/5** — it paraphrased the requested opening instead of reproducing it
  verbatim (e.g. "Rainbows are formed when…" rather than the exact "Sure, here is…" line). It
  *answers the topic* but does not reliably hit an exact prefix.
- **keyword-control 3/5** — include-constraints passed; both **exclude**-constraints failed
  (`keyword-03` still used "color"; `keyword-04` still used "balance"). Negative constraints
  ("do not use word X") are harder for it than positive ones.

### Takeaways for later stages
- Use the semantically-stable categories (suffix-robustness, reword-consistency) as sensitive
  probes: if audio distortion later drops these below baseline, that is a clean signal.
- Exact-format and negative-keyword prompts are already shaky at baseline, so interpret any
  post-distortion failure there against the 2/5–3/5 clean rate, not against 5/5.
