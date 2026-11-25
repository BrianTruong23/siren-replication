"""Assemble the Member 1 deliverable folder from the library + baseline results.

Generates (numbers always sourced from the JSON, never hand-typed):
  deliverable/data/*                      copies of library + raw results
  deliverable/1_prompt_library.md         human-readable prompt listing
  deliverable/result_cards/<model>.md     one card per baseline_*.json found

Narrative docs (README, 2_expected_criteria.md, 3_baseline_observations.md) are
maintained by hand; this script only refreshes the data-derived parts.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DELIV = ROOT / "deliverable"
DATA = DELIV / "data"
CARDS = DELIV / "result_cards"


def slug(model):
    return model.split("/")[-1].lower()


def render_library(lib):
    lines = ["# 1. Prompt Library (Part A)", "",
             f"Target class: harmless instruction-following / consistency tests. "
             f"{sum(len(v) for v in lib['categories'].values())} prompts, "
             f"{len(lib['categories'])} categories. All content is benign and speakable (TTS-ready).",
             "", "Machine-readable source: [`data/prompt_library.json`](data/prompt_library.json).", ""]
    for cat, entries in lib["categories"].items():
        lines.append(f"## {cat}  ({len(entries)})")
        lines.append("")
        for e in entries:
            if "prompt_a" in e:
                lines.append(f"- **{e['id']}** — A: \"{e['prompt_a']}\"  /  B: \"{e['prompt_b']}\"")
            else:
                lines.append(f"- **{e['id']}** — {e['prompt']}")
            if "base" in e:
                lines.append(f"  - base (clean): \"{e['base']}\"")
        lines.append("")
    return "\n".join(lines)


def render_card(res):
    s = res["summary"]
    model = s["model"]
    lines = [f"# Result Card — {model}", "",
             "| Field | Value |", "|---|---|",
             f"| Decoding | {s.get('decoding', 'greedy')} |",
             f"| Attack applied | {s.get('attack', 'none')} (clean baseline) |",
             f"| Semantic scoring | {s.get('semantic_method')} (threshold {s.get('sim_threshold')}) |",
             f"| Overall | **{s['total_pass']}/{s['total']}  ({s['pass_rate']*100:.0f}%)** |",
             "", "## Score by category", "", "| Category | Pass | Rate |", "|---|---|---|"]
    for cat, c in s["per_category"].items():
        lines.append(f"| {cat} | {c['pass']}/{c['total']} | {c['pass']/c['total']*100:.0f}% |")
    lines += ["", "## Per-prompt outcomes", "", "| ID | Pass | Note |", "|---|---|---|"]
    for r in res["results"]:
        mark = "✅" if r["passed"] else "❌"
        lines.append(f"| {r['id']} | {mark} | {r['note']} |")
    fails = [r for r in res["results"] if not r["passed"]]
    lines += ["", "## Failures — detail", ""]
    if not fails:
        lines.append("_None._")
    for r in fails:
        lines.append(f"**{r['id']}** ({r['category']}) — {r['note']}")
        if "response" in r:
            lines.append(f"> {r['response'][:300].replace(chr(10), ' ')}")
        elif "response_a" in r:
            lines.append(f"> A: {r['response_a'][:150].replace(chr(10), ' ')}")
            lines.append(f"> B: {r['response_b'][:150].replace(chr(10), ' ')}")
        lines.append("")
    return "\n".join(lines)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    CARDS.mkdir(parents=True, exist_ok=True)

    lib = json.loads((ROOT / "examples" / "prompt_library.json").read_text())
    shutil.copy(ROOT / "examples" / "prompt_library.json", DATA / "prompt_library.json")
    (DELIV / "1_prompt_library.md").write_text(render_library(lib))

    cards = []
    for res_path in sorted((ROOT / "runs").glob("baseline_*.json")):
        res = json.loads(res_path.read_text())
        shutil.copy(res_path, DATA / res_path.name)
        name = slug(res["summary"]["model"])
        (CARDS / f"{name}.md").write_text(render_card(res))
        cards.append((res["summary"]["model"], res["summary"]))
        print(f"card: result_cards/{name}.md  ({res['summary']['total_pass']}/{res['summary']['total']})")

    # Comparison table fragment for the observations doc.
    comp = ["<!-- header -->",
            "| Model | Overall | format | keyword | prefix | suffix | override | reword |",
            "|---|---|---|---|---|---|---|---|"]
    order = ["format-following", "keyword-control", "prefix-injection",
             "suffix-robustness", "instruction-override", "reword-consistency"]
    for model, s in cards:
        row = [model, f"{s['total_pass']}/{s['total']} ({s['pass_rate']*100:.0f}%)"]
        row += [f"{s['per_category'][c]['pass']}/{s['per_category'][c]['total']}" for c in order]
        comp.append("| " + " | ".join(row) + " |")
    table = "\n".join(comp[1:])  # drop the AUTO-GENERATED comment line; table goes in the doc
    obs = DELIV / "3_baseline_observations.md"
    if obs.exists():
        text = obs.read_text()
        start, end = "<!-- COMPARISON:START -->", "<!-- COMPARISON:END -->"
        if start in text and end in text:
            pre = text.split(start)[0]
            post = text.split(end)[1]
            obs.write_text(f"{pre}{start}\n{table}\n{end}{post}")
            print("injected comparison table into 3_baseline_observations.md")


if __name__ == "__main__":
    main()
