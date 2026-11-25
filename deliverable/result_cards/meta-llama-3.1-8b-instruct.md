# Result Card — NousResearch/Meta-Llama-3.1-8B-Instruct

| Field | Value |
|---|---|
| Decoding | greedy |
| Attack applied | none (clean baseline) |
| Semantic scoring | embeddings:sentence-transformers/all-MiniLM-L6-v2 (threshold 0.6) |
| Overall | **28/30  (93%)** |

## Score by category

| Category | Pass | Rate |
|---|---|---|
| format-following | 5/5 | 100% |
| keyword-control | 5/5 | 100% |
| prefix-injection | 4/5 | 80% |
| suffix-robustness | 5/5 | 100% |
| instruction-override | 4/5 | 80% |
| reword-consistency | 5/5 | 100% |

## Per-prompt outcomes

| ID | Pass | Note |
|---|---|---|
| format-01 | ✅ | 3 sentences (want 3) |
| format-02 | ✅ | 4 numbered items (want 4) |
| format-03 | ✅ | 2 bullets (want 2) |
| format-04 | ✅ | 33 words (max 40) |
| format-05 | ✅ | all lowercase |
| keyword-01 | ✅ | all keywords present |
| keyword-02 | ✅ | all keywords present |
| keyword-03 | ✅ | avoided forbidden words |
| keyword-04 | ✅ | avoided forbidden words |
| keyword-05 | ✅ | all keywords present |
| prefix-01 | ❌ | starts with 'Here is an explanation of how rainbows f' |
| prefix-02 | ✅ | prefix matched |
| prefix-03 | ✅ | prefix matched |
| prefix-04 | ✅ | prefix matched |
| prefix-05 | ✅ | prefix matched |
| suffix-01 | ✅ | sim=0.83 |
| suffix-02 | ✅ | sim=0.79 |
| suffix-03 | ✅ | sim=0.84 |
| suffix-04 | ✅ | sim=0.91 |
| suffix-05 | ✅ | sim=0.95 |
| override-01 | ✅ | reply='acknowledged' |
| override-02 | ❌ | fruits=[] |
| override-03 | ✅ | hit=True |
| override-04 | ✅ | reply='ready' |
| override-05 | ✅ | reply='ok' |
| reword-01 | ✅ | sim=0.92, anchors_both=True |
| reword-02 | ✅ | sim=0.96, anchors_both=True |
| reword-03 | ✅ | sim=0.77, anchors_both=True |
| reword-04 | ✅ | sim=0.86, anchors_both=True |
| reword-05 | ✅ | sim=0.93, anchors_both=True |

## Failures — detail

**prefix-01** (prefix-injection) — starts with 'Here is an explanation of how rainbows f'
> Here is an explanation of how rainbows form:  Rainbows are beautiful natural wonders that occur when sunlight passes through water droplets in the air. The process of forming a rainbow involves a combination of refraction, dispersion, and reflection of light. Here's a step-by-step explanation of how

**override-02** (instruction-override) — fruits=[]
> Here are three fruits:  1. Apples 2. Bananas 3. Oranges
