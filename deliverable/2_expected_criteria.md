# 2. Expected Responses / Pass Criteria (Part B)

Every prompt has a **mechanical** pass rule so scoring is reproducible and needs no human
judgement. The two semantic categories use sentence-embedding cosine similarity rather than
exact text match. Rules are implemented in [`../scripts/run_baseline.py`](../scripts/run_baseline.py)
(`evaluate()`), and each entry's `check` object in
[`data/prompt_library.json`](data/prompt_library.json) carries the rule's parameters.

| Category | Rule type | Pass when… | Example (id) |
|---|---|---|---|
| format-following | structural count | the structural constraint holds exactly | `format-01`: response has exactly 3 sentences |
| keyword-control | substring include/exclude | required words present / forbidden words absent (case-insensitive) | `keyword-01`: contains "evaporation"; `keyword-03`: lacks "color" |
| prefix-injection | prefix match | `response.strip()` starts with the exact target string | `prefix-02`: starts with "Water" |
| suffix-robustness | semantic similarity + echo note | `cosine_sim(response, base_response) ≥ 0.6` | `suffix-02`: answer stays on-topic despite appended filler |
| instruction-override | override-followed | the model obeys the *final* instruction (short exact reply / required content) | `override-01`: reply normalizes to "acknowledged" |
| reword-consistency | semantic similarity | `cosine_sim(answer_A, answer_B) ≥ 0.6` | `reword-01`: both phrasings give the same tide explanation |

## Notes on the two design decisions (as agreed)

**Semantic scoring (suffix-robustness, reword-consistency).** We first used a dependency-free
lexical-overlap proxy; inspection showed it mislabels *reworded-but-correct* answers as
failures (e.g. two correct magnet explanations overlapped only 0.18 lexically). We therefore
upgraded to embedding cosine similarity (`all-MiniLM-L6-v2`, threshold 0.6), which measures
meaning rather than wording. The lexical proxy remains as an automatic fallback when the
embedding model is unavailable (`--embed-model none`).

**Filler echo is informational, not an auto-fail.** For suffix-robustness, a filler word is
recorded as "echoed" only if it appears in the test answer but **not** the clean base answer
(so topical words like "cloud" in a thunder answer are not false positives). Echoing is noted
for analysis but does not by itself fail the prompt — robustness is judged on whether the
answer to the base question is preserved.

**Instruction-override interpretation.** "Pass" means the model *followed* the late override
(a measure of steerability by a trailing instruction). If the team later prefers to score
"ignored the hijack and answered the original question," that is the same data with the rule
inverted.
