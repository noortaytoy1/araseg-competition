"""BoundRL Stage-2 RLVR: GRPO with intermediate candidates (Sec 4.3 + 5.2).

TRL GRPOTrainer (trl==0.21.0, version-pinned; the hook below reads that
version's internals), m=4 generations, rollout temperature 1.2, NO standard
deviation reward scaling (scale_rewards=False, the paper's choice), KL
regularization to the SFT reference via --kl-coeff (GRPOConfig.beta), reward
from boundrl_reward (r = rho_rec * (EM + F1char) / 2), loss_type="grpo".

Batch layout mirrors the paper's "6 input documents, m=4 candidates":
per_device_train_batch_size=4 (= m, one group), steps_per_generation=6 and
gradient_accumulation_steps=6, so each optimizer step sees 6 documents x 4
candidates and generation happens once per optimizer step.  Because
gradient_accumulation_steps % (steps_per_generation * num_iterations) == 0,
TRL keeps old_per_token_logps = None (fully on-policy).

Reference model: pass --adapter (the SFT LoRA); it is merged into the base
(merge_and_unload) and GRPO trains a FRESH LoRA r=16 on the merged model, so
TRL's disable_adapter() reference — the KL anchor — is exactly the SFT-tuned
policy, as in the paper.

INTERMEDIATE-CANDIDATE HOOK (Sec 4.3) — exactly where it plugs into TRL
========================================================================
TRL 0.21.0 exposes no rollout post-processing API, but the whole rollout ->
reward -> advantage pipeline lives in one method,
`GRPOTrainer._generate_and_score_completions(inputs)`.  We subclass it:

    1. `super()._generate_and_score_completions(inputs)` runs generation,
       reward and advantage computation unchanged and returns the training
       dict {prompt_ids, prompt_mask, completion_ids, completion_mask,
       advantages, [ref_per_token_logps], [old_per_token_logps]}.
    2. The hook then re-derives each candidate's reward from
       completion_ids/completion_mask (our reward is a deterministic pure
       function of the completion text, so these rewards are identical to the
       ones the parent computed via `_calculate_rewards`), ranks each
       m-candidate group by descending reward, takes the median candidate
       (0-based index (m-1)//2, i.e. rank m/2 for even m), builds the
       perturbation pool (each segment start shifted LEFT/RIGHT by one word —
       the single-class analogue of Sec 4.3; label replacement does not
       exist with one class), scores the pool, and keeps the best candidate
       only if its reward exceeds the median's.
    3. Selective replacement: across the generation batch, at most --k
       groups (paper: k=2 for Qwen3-1.7b), largest reward gain first, get
       their median candidate REPLACED in-place: the perturbed completion is
       re-tokenized (+EOS), completion_ids/completion_mask re-padded, the
       group's advantages recomputed from the post-swap rewards (respecting
       scale_rewards=False), and ref/old per-token logps recomputed for the
       whole batch with the same code path the parent used (they were
       computed BEFORE the swap and would otherwise be stale).
    4. Everything else (loss, clipping, KL) proceeds in TRL unchanged, so
       the swapped candidate is treated exactly like a generated rollout —
       this IS the paper's replacement strategy, not the "extra scored
       sequence" fallback.  Deviation from the paper worth noting: TRL
       computes ref logps before rewards, so a swap costs one extra
       forward pass per swapped batch; and A.2's neighbour-sequence overlap
       repair after perturbation is not implemented — an overlapping
       perturbed start simply scores lower and is not selected.

If the hook ever fails it prints the error and returns the parent's output
unchanged (rollouts then proceed as plain GRPO for that batch); failures are
counted in the `boundrl/hook_errors` metric.

Usage (CPU smoke):
  CUDA_VISIBLE_DEVICES=-1 python src/boundrl_grpo.py --model sshleifer/tiny-gpt2 \
    --adapter runs/boundrl_pilot_2026-07-10/smoke_sft --max-steps 1 --m 4 \
    --per-device-batch 4 --steps-per-gen 1 --grad-accum 1 --k 1 \
    --window-tokens 256 --max-prompt 512 --max-completion 48 --subset 1.0 \
    --max-docs 5 --out runs/boundrl_pilot_2026-07-10/smoke_grpo
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boundrl_data import build_examples, format_target, locate_starts  # noqa: E402
from boundrl_reward import completion_reward  # noqa: E402
from boundrl_sft import LR_RANGE, load_model_and_tokenizer, lora_config_for  # noqa: E402
from data import load_jsonl  # noqa: E402

# chunk_id -> {"words": [...], "gold_spans": [...]} for reward lookups
REGISTRY: Dict[str, dict] = {}


def boundrl_reward_fn(prompts=None, completions=None, completion_ids=None, **kwargs):
    """TRL reward function; extra dataset column `chunk_id` arrives in kwargs."""
    out = []
    for text, cid in zip(completions, kwargs["chunk_id"]):
        rec = REGISTRY[cid]
        out.append(float(completion_reward(rec["words"], rec["gold_spans"], text)["reward"]))
    return out


boundrl_reward_fn.__name__ = "boundrl_reward"


# --------------------------------------------------------------------------
# Sec 4.3 perturbations (single-class: boundary shifts only)
# --------------------------------------------------------------------------

def perturbation_pool(
    words: Sequence[str], start_seqs: List[List[str]]
) -> List[List[List[str]]]:
    """All single-perturbation variants of a candidate segmentation: each
    located segment start shifted one word right (drop first word of its
    start sequence; skipped for one-word sequences per A.2) or one word left
    (prepend the document word immediately before its located position)."""
    positions = locate_starts(words, start_seqs)
    pool: List[List[List[str]]] = []
    for i, (seq, pos) in enumerate(zip(start_seqs, positions)):
        if pos is None:
            continue
        if len(seq) >= 2:  # shift boundary right by one word
            variant = [list(s) for s in start_seqs]
            variant[i] = list(seq[1:])
            pool.append(variant)
        if pos > 0:  # shift boundary left by one word
            variant = [list(s) for s in start_seqs]
            variant[i] = [words[pos - 1]] + list(seq)
            pool.append(variant)
    return pool


def best_intermediate(
    words: Sequence[str], gold_spans: Sequence[Tuple[int, int]], completion_text: str
) -> Optional[Tuple[str, float]]:
    """Best-reward single-perturbation intermediate candidate (Sec 4.3), or
    None if the completion parses to no locatable segments / empty pool."""
    parsed = completion_reward(words, gold_spans, completion_text)
    seqs = parsed["start_seqs"]
    if not seqs:
        return None
    best_text, best_r = None, float("-inf")
    for variant in perturbation_pool(words, seqs):
        if any(len(s) == 0 for s in variant):
            continue
        text = format_target(variant)
        r = float(completion_reward(words, gold_spans, text)["reward"])
        if r > best_r:
            best_text, best_r = text, r
    if best_text is None:
        return None
    return best_text, best_r


def make_trainer_class(GRPOTrainer):
    """Factory so trl import stays inside main() for fast --help/tests."""
    import torch

    class BoundRLGRPOTrainer(GRPOTrainer):
        """GRPOTrainer + Sec 4.3 intermediate-candidate replacement hook."""

        boundrl_k: int = 2
        boundrl_max_completion: int = 256

        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            mode = "train" if self.model.training else "eval"
            try:
                n_swapped = self._boundrl_swap(inputs, out)
                self._metrics[mode]["boundrl/n_swapped"].append(float(n_swapped))
            except Exception as e:  # noqa: BLE001 — pilot safety net, loudly logged
                print(f"[boundrl_grpo] HOOK ERROR (batch used unmodified): {e!r}", file=sys.stderr)
                self._metrics[mode]["boundrl/hook_errors"].append(1.0)
            return out

        def _boundrl_swap(self, inputs, out) -> int:
            assert self.accelerator.num_processes == 1, "hook assumes single process (scope guard: no multi-GPU)"
            tok = self.processing_class
            G = self.num_generations
            ids, mask = out["completion_ids"], out["completion_mask"]
            n = ids.size(0)
            if n % G != 0:
                return 0
            texts = [
                tok.decode(row[m.bool()], skip_special_tokens=True)
                for row, m in zip(ids, mask)
            ]
            chunk_ids = [inp["chunk_id"] for inp in inputs]
            rewards = [
                float(
                    completion_reward(
                        REGISTRY[cid]["words"], REGISTRY[cid]["gold_spans"], t
                    )["reward"]
                )
                for t, cid in zip(texts, chunk_ids)
            ]

            # rank each group, perturb the median candidate (Sec 4.3)
            proposals = []  # (gain, row_index, new_text, new_reward)
            for g in range(n // G):
                rows = list(range(g * G, (g + 1) * G))
                order = sorted(rows, key=lambda r: rewards[r], reverse=True)
                med_row = order[(G - 1) // 2]
                rec = REGISTRY[chunk_ids[med_row]]
                cand = best_intermediate(rec["words"], rec["gold_spans"], texts[med_row])
                if cand is None:
                    continue
                new_text, new_r = cand
                if new_r > rewards[med_row]:
                    proposals.append((new_r - rewards[med_row], med_row, new_text, new_r))

            proposals.sort(reverse=True)  # largest gain first
            chosen = proposals[: self.boundrl_k]
            swapped_rows = []
            new_ids_list = [
                [int(t) for t, m in zip(row, mrow) if m] for row, mrow in zip(ids, mask)
            ]
            for _, row, new_text, new_r in chosen:
                enc = tok(new_text, add_special_tokens=False).input_ids + [self.eos_token_id]
                if len(enc) > self.boundrl_max_completion:
                    continue  # would exceed the completion budget; skip this swap
                new_ids_list[row] = enc
                rewards[row] = new_r
                swapped_rows.append(row)
            if not swapped_rows:
                return 0

            # rebuild padded completion tensors
            device = ids.device
            width = max(len(r) for r in new_ids_list)
            pad_id = self.pad_token_id if self.pad_token_id is not None else self.eos_token_id
            new_ids = torch.full((n, width), pad_id, dtype=ids.dtype, device=device)
            new_mask = torch.zeros((n, width), dtype=mask.dtype, device=device)
            for i, r in enumerate(new_ids_list):
                new_ids[i, : len(r)] = torch.tensor(r, dtype=ids.dtype, device=device)
                new_mask[i, : len(r)] = 1
            out["completion_ids"], out["completion_mask"] = new_ids, new_mask

            # recompute advantages from post-swap rewards (paper: no std scaling)
            rew = torch.tensor(rewards, dtype=torch.float32, device=device)
            mean_g = rew.view(-1, G).mean(dim=1).repeat_interleave(G)
            adv = rew - mean_g
            if self.scale_rewards:
                std_g = rew.view(-1, G).std(dim=1).repeat_interleave(G)
                adv = adv / (std_g + 1e-4)
            out["advantages"] = adv

            # ref/old per-token logps were computed pre-swap -> recompute
            prompt_completion_ids = torch.cat([out["prompt_ids"], new_ids], dim=1)
            attention_mask = torch.cat([out["prompt_mask"], new_mask], dim=1)
            logits_to_keep = new_ids.size(1)
            bs = (
                self.args.per_device_train_batch_size
                if self.model.training
                else self.args.per_device_eval_batch_size
            )
            with torch.no_grad():
                if out.get("old_per_token_logps") is not None:
                    out["old_per_token_logps"], _ = self._get_per_token_logps_and_entropies(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, bs
                    )
                if out.get("ref_per_token_logps") is not None:
                    if self.ref_model is not None:
                        out["ref_per_token_logps"], _ = self._get_per_token_logps_and_entropies(
                            self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, bs
                        )
                    else:
                        with self.accelerator.unwrap_model(self.model).disable_adapter():
                            out["ref_per_token_logps"], _ = self._get_per_token_logps_and_entropies(
                                self.model, prompt_completion_ids, attention_mask, logits_to_keep, bs
                            )
            return len(swapped_rows)

    return BoundRLGRPOTrainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--fallback-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="SFT LoRA adapter dir (merged into the base -> KL reference)")
    ap.add_argument("--train-jsonl", default="data/NoPnx-NP_train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-6, help="paper: 1e-6 for Qwen3 RLVR")
    ap.add_argument("--kl-coeff", type=float, default=0.04, help="GRPOConfig.beta (KL regularization)")
    ap.add_argument("--m", type=int, default=4, help="candidate segmentations per input")
    ap.add_argument("--temp", type=float, default=1.2)
    ap.add_argument("--k", type=int, default=2, help="max intermediate-candidate replacements per batch")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=6)
    ap.add_argument("--steps-per-gen", type=int, default=6)
    ap.add_argument("--subset", type=float, default=0.25, help="paper: RLVR on a random 25%% subset")
    ap.add_argument("--window-tokens", type=int, default=2048)
    ap.add_argument("--max-prompt", type=int, default=4096)
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--save-steps", type=int, default=0, help="0 = save only at the end")
    args = ap.parse_args()

    import datasets as hf_datasets
    import peft as peft_pkg
    import transformers
    import trl
    from trl import GRPOConfig, GRPOTrainer

    model, tok, used_model = load_model_and_tokenizer(args.model, args.fallback_model, args.dtype)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[boundrl_grpo] merged SFT adapter {args.adapter} into {used_model} (KL reference = SFT policy)")

    count_tokens = lambda s: len(tok(s, add_special_tokens=False).input_ids)  # noqa: E731
    docs = load_jsonl(args.train_jsonl)
    if args.max_docs:
        docs = docs[: args.max_docs]
    examples = build_examples(docs, seed=args.seed, count_tokens=count_tokens, budget=args.window_tokens)
    rng = random.Random(args.seed)
    if args.subset < 1.0:
        n_keep = max(1, int(round(args.subset * len(examples))))
        examples = rng.sample(examples, n_keep)
    for ex in examples:
        REGISTRY[ex["chunk_id"]] = {"words": ex["words"], "gold_spans": ex["gold_spans"]}
    ds = hf_datasets.Dataset.from_list(
        [{"prompt": ex["prompt"], "chunk_id": ex["chunk_id"]} for ex in examples]
    )
    print(f"[boundrl_grpo] model={used_model} docs={len(docs)} rl-examples={len(examples)}")

    os.makedirs(args.out, exist_ok=True)
    cfg = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        steps_per_generation=args.steps_per_gen,
        num_generations=args.m,
        temperature=args.temp,
        beta=args.kl_coeff,
        scale_rewards=False,  # paper: GRPO without std-based reward scaling
        loss_type="grpo",
        epsilon=0.2,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.1,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_completion,
        logging_steps=1,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        report_to=[],
        seed=args.seed,
        bf16=args.dtype == "bfloat16",
        use_cpu=os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1",
    )
    TrainerCls = make_trainer_class(GRPOTrainer)
    trainer = TrainerCls(
        model=model,
        reward_funcs=boundrl_reward_fn,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora_config_for(model, 16, 32),
    )
    trainer.boundrl_k = args.k
    trainer.boundrl_max_completion = args.max_completion
    result = trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    swaps = trainer._metrics["train"].get("boundrl/n_swapped", [])
    record = {
        "model_used": used_model,
        "adapter_merged": args.adapter,
        "train_jsonl": args.train_jsonl,
        "n_docs": len(docs),
        "n_rl_examples": len(examples),
        "subset": args.subset,
        "lr": args.lr,
        "kl_coeff_beta": args.kl_coeff,
        "m": args.m,
        "temperature": args.temp,
        "k_replacements": args.k,
        "scale_rewards": False,
        "loss_type": "grpo",
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "train_loss": float(result.training_loss),
        "n_generation_batches": len(swaps),
        "total_swaps": float(sum(swaps)),
        "hook_errors": float(sum(trainer._metrics["train"].get("boundrl/hook_errors", []))),
        "versions": {
            "trl": trl.__version__,
            "peft": peft_pkg.__version__,
            "transformers": transformers.__version__,
            "datasets": hf_datasets.__version__,
        },
    }
    with open(os.path.join(args.out, "boundrl_grpo_config.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[boundrl_grpo] done; swaps={record['total_swaps']} hook_errors={record['hook_errors']} -> {args.out}")


if __name__ == "__main__":
    main()
