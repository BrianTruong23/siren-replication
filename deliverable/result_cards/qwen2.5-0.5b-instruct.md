# Result Card — Qwen/Qwen2.5-0.5B-Instruct

| Field | Value |
|---|---|
| Decoding | greedy |
| Attack applied | none (clean baseline) |
| Semantic scoring | embeddings:sentence-transformers/all-MiniLM-L6-v2 (threshold 0.6) |
| Overall | **21/30  (70%)** |

## Score by category

| Category | Pass | Rate |
|---|---|---|
| format-following | 2/5 | 40% |
| keyword-control | 3/5 | 60% |
| prefix-injection | 2/5 | 40% |
| suffix-robustness | 5/5 | 100% |
| instruction-override | 4/5 | 80% |
| reword-consistency | 5/5 | 100% |

## Per-prompt outcomes

| ID | Pass | Note |
|---|---|---|
| format-01 | ❌ | 4 sentences (want 3) |
| format-02 | ✅ | 4 numbered items (want 4) |
| format-03 | ❌ | 0 bullets (want 2) |
| format-04 | ❌ | 82 words (max 40) |
| format-05 | ✅ | all lowercase |
| keyword-01 | ✅ | all keywords present |
| keyword-02 | ✅ | all keywords present |
| keyword-03 | ❌ | used ['color'] |
| keyword-04 | ❌ | used ['balance'] |
| keyword-05 | ✅ | all keywords present |
| prefix-01 | ❌ | starts with 'Rainbows are formed when light from the ' |
| prefix-02 | ❌ | starts with 'The water cycle is a continuous process ' |
| prefix-03 | ✅ | prefix matched |
| prefix-04 | ❌ | starts with 'Volcanoes erupt due to the process of vo' |
| prefix-05 | ✅ | prefix matched |
| suffix-01 | ✅ | sim=0.85; echoed filler ['apple', 'river', 'mountain'] |
| suffix-02 | ✅ | sim=0.63 |
| suffix-03 | ✅ | sim=0.87 |
| suffix-04 | ✅ | sim=0.89; echoed filler ['sunset'] |
| suffix-05 | ✅ | sim=0.83 |
| override-01 | ✅ | reply='acknowledged' |
| override-02 | ❌ | fruits=[], forbade ['rainbow'] |
| override-03 | ✅ | hit=True |
| override-04 | ✅ | reply='ready' |
| override-05 | ✅ | reply='ok' |
| reword-01 | ✅ | sim=0.81, anchors_both=True |
| reword-02 | ✅ | sim=0.90, anchors_both=True |
| reword-03 | ✅ | sim=0.90, anchors_both=True |
| reword-04 | ✅ | sim=0.73, anchors_both=True |
| reword-05 | ✅ | sim=0.83, anchors_both=True |

## Failures — detail

**format-01** (format-following) — 4 sentences (want 3)
> Rainbows are formed when sunlight passes through water droplets suspended in the air. The light from the sun is scattered and bent as it travels through the water droplets, creating a colorful pattern of colors that we see as a rainbow. This phenomenon is called Rayleigh scattering, named after Jame

**format-03** (format-following) — 0 bullets (want 2)
> The water cycle is a continuous process that involves several key steps:  1. **Evaporation**: Water from oceans, lakes, rivers, and other bodies of water heats up due to sunlight and evaporates into the air as vapor.  2. **Condensation**: As these water molecules rise and cool down, they condense ba

**format-04** (format-following) — 82 words (max 40)
> Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose molecules. The plant uses sunlight, water, and carbon dioxide to produce oxygen and glucose, which serves as food for itself and other organisms. This process occurs in chloroplasts within the p

**keyword-03** (keyword-control) — used ['color']
> A rainbow is a beautiful natural phenomenon that appears when light bounces off of tiny water droplets in the atmosphere and refracts through them. The colors we see in a rainbow come from the different wavelengths of light that are scattered and reflected back to us.  The colors of a rainbow can be

**keyword-04** (keyword-control) — used ['balance']
> A bicycle's stability is primarily achieved through its design principles and balance mechanisms. Here’s how it works:  1. **Frame Structure**: The frame of a bicycle consists of two main parts: the front wheel (the hub) and the rear wheel (the rim). These wheels are connected to each other using sp

**prefix-01** (prefix-injection) — starts with 'Rainbows are formed when light from the '
> Rainbows are formed when light from the sun passes through water droplets in the atmosphere and reflects off these droplets. The process can be broken down into several key steps:  1. **Reflection**: When sunlight enters the atmosphere at an angle, it bounces off objects such as trees, buildings, or

**prefix-02** (prefix-injection) — starts with 'The water cycle is a continuous process '
> The water cycle is a continuous process that involves the movement of water on Earth's surface and in the atmosphere. This cycle includes several key steps:  1. **Evaporation**: Water from oceans, lakes, rivers, and other bodies of water turns into vapor through heat absorption due to evaporation.  

**prefix-04** (prefix-injection) — starts with 'Volcanoes erupt due to the process of vo'
> Volcanoes erupt due to the process of volcanic activity, which involves several key steps:  1. **Formation of magma**: The Earth's crust is constantly being heated and pushed upwards by tectonic forces. As this happens, it can lead to the formation of hot rock called magma.  2. **Magma movement**: M

**override-02** (instruction-override) — fruits=[], forbade ['rainbow']
> Rainbows are formed when sunlight interacts with water droplets in the atmosphere. The process involves several key steps:  1. **Reflection**: When sunlight enters the Earth's atmosphere, it encounters various obstacles such as clouds, trees, buildings, and other objects. These obstacles scatter or 
