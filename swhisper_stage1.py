"""Bounded, text-only approximation of SWhisper §§5.1–5.2 for local models."""
import argparse
import json
import math
import random
import re
from pathlib import Path


def voiceable(text):
    """Conservative heuristic, NOT the paper's unspecified semantic vocabulary."""
    return bool(re.fullmatch(r"[A-Za-z]+(?:['-][A-Za-z]+)*", text.strip()))


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input must be a nonempty JSON array")
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("goal", "target")):
            raise ValueError("Each row needs nonempty goal and target strings")


def compute_losses(logits, ids, start, target_len, n, h, semantic_weight):
    """Eq. 3: mean target NLL; Eq. 7: mean suffix NLL including its first token.

    Layout of each row of `ids` is [prefix(n) | suffix(h) | tail | target(target_len)],
    so `start = n + h + len(tail)` and `ids[:, start:]` is the target region. The logits
    at position p-1 predict the token at position p, hence the shifted slices below.
    """
    import torch.nn.functional as F
    b = ids.shape[0]
    adv = F.cross_entropy(
        logits[:, start-1:start+target_len-1].float().reshape(-1, logits.shape[-1]),
        ids[:, start:].reshape(-1), reduction="none").reshape(b, -1).mean(1)
    sem = F.cross_entropy(
        logits[:, n-1:n+h-1].float().reshape(-1, logits.shape[-1]),
        ids[:, n:n+h].reshape(-1), reduction="none").reshape(b, h).mean(1)
    return adv + semantic_weight * sem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local path or Hugging Face causal LM ID")
    p.add_argument("--input", default="examples/tasks.json")
    p.add_argument("--output", default="runs/stage1.jsonl")
    p.add_argument("--template", default="examples/template.txt")
    p.add_argument("--seeds", help="Optional JSON array of candidate initial suffix strings")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--suffix-length", type=int, default=20)
    p.add_argument("--top-k", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64, help="Proposals per step, NOT GPU microbatch")
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--positions", type=int, default=2, help="Positions updated together (approximation)")
    p.add_argument("--semantic-weight", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--final-temperature", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = p.parse_args()
    for key in ("steps", "suffix_length", "top_k", "batch_size", "microbatch", "positions", "max_new_tokens"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if min(args.temperature, args.final_temperature) <= 0 or args.semantic_weight < 0:
        p.error("Temperatures must be positive; semantic weight must be nonnegative")
    rows = json.loads(Path(args.input).read_text())
    validate_rows(rows)
    template = Path(args.template).read_text().strip()
    if "{goal}" not in template or not template.endswith("{adv_suffix}"):
        p.error("Template must contain {goal} and end with {adv_suffix}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if not tok.chat_template:
        p.error("Model needs a chat template; use an instruction-tuned causal LM")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), trust_remote_code=False
    ).to(args.device).eval()
    model.requires_grad_(False)
    embedding = model.get_input_embeddings()
    vocab_size = min(embedding.num_embeddings, model.config.vocab_size, len(tok))
    allowed = torch.tensor([
        i for i in range(vocab_size)
        if i not in tok.all_special_ids and voiceable(tok.decode([i]))
    ], device=args.device)
    if not len(allowed):
        raise ValueError("No voiceable vocabulary tokens found")
    allowed_set = set(allowed.tolist())
    seeds = json.loads(Path(args.seeds).read_text()) if args.seeds else []
    if not isinstance(seeds, list) or any(not isinstance(s, str) for s in seeds):
        raise ValueError("Seeds must be a JSON array of strings")
    seed_ids = [tok.encode(s, add_special_tokens=False) for s in seeds]
    seed_ids = [s for s in seed_ids if len(s) == args.suffix_length and set(s) <= allowed_set]
    if args.seeds and not seed_ids:
        raise ValueError("No seed has the required token length and voiceable vocabulary")

    def tensor(ids):
        return torch.tensor(ids, dtype=torch.long, device=args.device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new --output")
    with out.open("x") as output:
        for row_index, row in enumerate(rows):
            # Render with an empty suffix, then insert token IDs before the user-turn ending.
            marker = "SWHISPER_SUFFIX_INSERTION_MARKER"
            user = template.format(goal=row["goal"], guiding_str=row["target"], adv_suffix=marker)
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
            if rendered.count(marker) != 1:
                raise ValueError("Template must preserve exactly one insertion marker")
            before, after = rendered.split(marker)
            prefix = tensor(tok.encode(before, add_special_tokens=False))
            tail = tensor(tok.encode(after, add_special_tokens=False))
            target = tensor(tok.encode(row["target"], add_special_tokens=False))
            n, h = len(prefix), args.suffix_length
            start = n + h + len(tail)
            context_limit = getattr(model.config, "max_position_embeddings", None)
            if context_limit and start + len(target) + args.max_new_tokens > context_limit:
                raise ValueError("Prompt/target/generation exceeds model context window")

            def assemble(suffixes):
                b = suffixes.shape[0]
                return torch.cat((prefix.expand(b, -1), suffixes,
                                  tail.expand(b, -1), target.expand(b, -1)), dim=1)

            def losses(logits, ids):
                return compute_losses(logits, ids, start, len(target), n, h, args.semantic_weight)

            @torch.no_grad()
            def score(candidates):
                scores = []
                for batch in candidates.split(args.microbatch):
                    ids = assemble(batch)
                    scores.append(losses(model(input_ids=ids, use_cache=False).logits, ids))
                return torch.cat(scores)

            current = tensor(random.choice(seed_ids)) if seed_ids else allowed[
                torch.randint(len(allowed), (h,), device=args.device)]
            current_loss = score(current[None])[0].item()
            best, best_loss = current.clone(), current_loss
            history = []
            for step in range(args.steps):
                ids = assemble(current[None])
                inputs = embedding(ids).detach().requires_grad_(True)
                loss = losses(model(inputs_embeds=inputs, use_cache=False).logits, ids).sum()
                grad = torch.autograd.grad(loss, inputs)[0][0, n:n+h].float()
                with torch.no_grad():
                    # First-order replacement ranking restricted to voiceable token IDs.
                    ranks = grad @ embedding.weight[allowed].float().T
                    choices = allowed[(-ranks).topk(min(args.top_k, len(allowed)), dim=1).indices]
                    candidates = current.repeat(args.batch_size, 1)
                    for b in range(args.batch_size):
                        positions = torch.randperm(h, device=args.device)[:min(args.positions, h)]
                        for j in positions.tolist():
                            candidates[b, j] = choices[j, random.randrange(choices.shape[1])]
                    # Ensure decoded candidates correspond to the token sequences being optimized.
                    valid = [c for c in candidates if tok.encode(tok.decode(c.tolist()),
                             add_special_tokens=False) == c.tolist()]
                    if not valid:
                        continue
                    candidates = torch.stack(valid)
                    scores = score(candidates)
                    index = scores.argmin().item()
                    proposal_loss = scores[index].item()
                    temperature = args.temperature * (args.final_temperature / args.temperature) ** (
                        step / max(1, args.steps - 1))
                    if proposal_loss <= current_loss or random.random() < math.exp(
                            min(0, (current_loss - proposal_loss) / temperature)):
                        current, current_loss = candidates[index].clone(), proposal_loss
                    if proposal_loss < best_loss:
                        best, best_loss = candidates[index].clone(), proposal_loss
                    if step % 10 == 0 or step == args.steps - 1:
                        history.append({"step": step, "current_loss": current_loss, "best_loss": best_loss})
                        print(f"task={row_index} step={step} best_loss={best_loss:.4f}", flush=True)
                del inputs, grad, loss
            prompt_ids = assemble(best[None])[:, :start]
            with torch.no_grad():
                response_ids = model.generate(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                    do_sample=False, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.eos_token_id)
            suffix = tok.decode(best.tolist())
            record = {"task": row, "suffix": suffix, "suffix_ids": best.tolist(),
                      "prompt": template.format(goal=row["goal"], guiding_str=row["target"], adv_suffix=suffix),
                      "prompt_ids": prompt_ids[0].tolist(), "best_loss": best_loss,
                      "response": tok.decode(response_ids[0, start:], skip_special_tokens=True),
                      "history": history, "config": vars(args),
                      "limitations": ["text-only", "heuristic vocabulary", "sampled parallel proposals",
                                      "no TTS/ASR distortion", "no success judge"]}
            output.write(json.dumps(record) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
