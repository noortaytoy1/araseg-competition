"""GPU smoke test for byol_pretrain.py.

Proves, on a tiny UNLABELED Arabic corpus and a real GPU, that:
  (a) the BYOL loss DECREASES over steps,
  (b) the online representation std stays AWAY from zero (no collapse),
  (c) the EMA target weights actually CHANGE and are DETACHED (no grad), and
  (d) the saved encoder RELOADS with AutoModel / AutoModelForTokenClassification.

All Arabic is written to files, never printed, to avoid Windows cp1252 stdout
errors. Numbers are printed (ASCII) and also dumped to a JSON report.

Run:  python smoke_byol.py
"""
from __future__ import annotations

import json
import os
import random
import sys

import torch
from transformers import (AutoModel, AutoModelForTokenClassification,
                          AutoTokenizer)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import byol_pretrain as B  # noqa: E402


# A small pool of Arabic words + the prepositions, to synthesize a few hundred
# passages that actually contain the connectives the preposition-aware view
# targets. Written to a UTF-8 corpus file; never printed.
_WORDS = [
    "المدرسة", "الطالب", "الكتاب", "المعلم", "البيت", "الشارع", "المدينة",
    "الحديقة", "الشمس", "القمر", "البحر", "الجبل", "النهر", "السماء", "الطريق",
    "الرجل", "المرأة", "الطفل", "العمل", "الوقت", "اليوم", "الليل", "الصباح",
    "الماء", "الهواء", "الشجرة", "الزهرة", "الطائر", "الحصان", "السيارة",
    "كبير", "صغير", "جميل", "سريع", "بطيء", "قديم", "جديد", "بعيد", "قريب",
    "يذهب", "يكتب", "يقرأ", "يلعب", "يجلس", "يمشي", "ينظر", "يفكر", "يعمل",
]
_PREPS = B.PREPOSITIONS


def make_corpus(path: str, n_passages: int = 400, seed: int = 7) -> int:
    rng = random.Random(seed)
    lines = []
    for _ in range(n_passages):
        n_words = rng.randint(8, 30)
        toks = []
        for j in range(n_words):
            # sprinkle prepositions so view B has something to hide
            if j > 0 and rng.random() < 0.35:
                toks.append(rng.choice(_PREPS))
            toks.append(rng.choice(_WORDS))
        lines.append(" ".join(toks))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


class Args:
    """Mirror of byol_pretrain's argparse Namespace for the smoke config."""
    def __init__(self, corpus, out):
        self.corpus = corpus
        self.out = out
        self.model_name = B.DEFAULT_MODEL
        self.max_length = 64
        self.batch_size = 16
        self.epochs = 4          # enough batches (25/epoch) to reach max_steps
        self.max_steps = 60
        self.lr = 3e-4          # a bit hot so the loss moves visibly in 60 steps
        self.weight_decay = 1e-6
        self.warmup_frac = 0.1
        self.clip_norm = 1.0
        self.ema_decay = 0.99   # faster target drift so (c) is unambiguous
        self.ema_ramp = False
        self.mask_prob = 0.15
        self.proj_hidden = 1024
        self.proj_dim = 128
        self.log_every = 5
        self.collapse_std = 1e-3
        self.seed = 42
        self.cpu = False
        self.no_bf16 = False


def main():
    report = {}
    corpus = os.path.join(HERE, os.pardir, "data", "byol_smoke_corpus.txt")
    out = os.path.join(HERE, os.pardir, "runs", "byol_smoke2")
    os.makedirs(os.path.dirname(corpus), exist_ok=True)

    n = make_corpus(corpus)
    report["n_passages"] = n
    report["cuda"] = torch.cuda.is_available()
    report["device_name"] = (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else "CPU")

    args = Args(corpus, out)

    # ---- Snapshot a target-encoder parameter BEFORE training to prove EMA drift
    # We rebuild the nets here mirroring train(), snapshot, then let train() run
    # a fresh pair — simpler: capture from inside via a hook is overkill, so we
    # instead reload the saved target-equivalent is impossible (target isn't
    # saved). Cleanest: monkeypatch to grab handles. We instead re-run the core
    # by calling train() and separately verifying EMA on a fresh mini-instance.

    # (c) EMA CHANGE + DETACH check on a fresh tiny instance (few steps) ------
    from transformers import set_seed
    set_seed(123)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    online = B.OnlineNet(args.model_name, args.proj_hidden, args.proj_dim).to(dev)
    target = B.TargetNet(online).to(dev)
    # requires_grad must be False on every target param (detached / no grad)
    report["target_all_requires_grad_false"] = all(
        (not p.requires_grad) for p in target.parameters())
    # snapshot a representative encoder weight of the target
    ref_name = None
    for nme, p in target.encoder.named_parameters():
        if p.dtype.is_floating_point and p.numel() > 10:
            ref_name = nme
            before = p.detach().clone()
            break
    # nudge the ONLINE weights (simulate an optimizer step), then EMA-update
    with torch.no_grad():
        for p in online.encoder.parameters():
            if p.dtype.is_floating_point:
                p.add_(torch.randn_like(p) * 0.05)
    target.update_ema(online, decay=0.99)
    after = dict(target.encoder.named_parameters())[ref_name].detach()
    ema_delta = (after - before).abs().mean().item()
    report["ema_ref_param"] = ref_name
    report["ema_mean_abs_change"] = ema_delta
    report["ema_changed"] = ema_delta > 0.0
    # target has no .grad after a backward on the online loss (verify detach):
    # run one loss.backward on the online path and confirm no target grad.
    tok = AutoTokenizer.from_pretrained(args.model_name)
    ids = tok(["الطالب و المعلم في المدرسة"], return_tensors="pt",
              padding=True).to(dev)
    att = ids["attention_mask"]
    inp = ids["input_ids"]
    valid = att.bool() & ~(inp == tok.cls_token_id) & ~(inp == tok.sep_token_id)
    loss = B.byol_token_loss(online, target, inp, inp, att, valid)
    loss.backward()
    report["target_grads_all_none"] = all(
        (p.grad is None) for p in target.parameters())
    del online, target
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- (a)+(b) run the real training loop -------------------------------
    result = B.train(args)
    hist = result["history"]  # list of (step, loss, std)
    report["history"] = hist
    losses = [h[1] for h in hist]
    stds = [h[2] for h in hist if h[2] == h[2]]  # drop NaN
    # early vs late loss (average first third vs last third for robustness)
    k = max(1, len(losses) // 3)
    report["loss_first"] = sum(losses[:k]) / k
    report["loss_last"] = sum(losses[-k:]) / k
    report["loss_decreased"] = report["loss_last"] < report["loss_first"]
    report["min_online_std"] = min(stds) if stds else None
    report["final_online_std"] = stds[-1] if stds else None
    report["std_off_zero"] = (min(stds) > args.collapse_std) if stds else False

    # ---- (d) reload the saved encoder -------------------------------------
    reload_ok = True
    reload_err = None
    try:
        enc = AutoModel.from_pretrained(out)
        _ = enc.config.hidden_size
        clf = AutoModelForTokenClassification.from_pretrained(out, num_labels=2)
        _ = clf.config.num_labels
        # a forward pass to be sure the weights are usable
        tok2 = AutoTokenizer.from_pretrained(out)
        b = tok2(["الطالب في المدرسة"], return_tensors="pt")
        with torch.no_grad():
            o = enc(**b)
        report["reload_hidden_shape"] = list(o.last_hidden_state.shape)
    except Exception as e:  # noqa: BLE001
        reload_ok = False
        reload_err = repr(e)
    report["reload_ok"] = reload_ok
    report["reload_err"] = reload_err

    # ---- verdict ----------------------------------------------------------
    passed = (report["loss_decreased"] and report["std_off_zero"]
              and report["ema_changed"] and report["target_grads_all_none"]
              and report["target_all_requires_grad_false"] and report["reload_ok"])
    report["ALL_PASS"] = passed

    rpath = os.path.join(HERE, os.pardir, "runs", "byol_smoke2_report.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== BYOL SMOKE REPORT =====")
    for key in ["cuda", "device_name", "n_passages",
                "loss_first", "loss_last", "loss_decreased",
                "min_online_std", "final_online_std", "std_off_zero",
                "ema_ref_param", "ema_mean_abs_change", "ema_changed",
                "target_all_requires_grad_false", "target_grads_all_none",
                "reload_ok", "reload_hidden_shape", "reload_err",
                "ALL_PASS"]:
        print("  %-32s %s" % (key, report.get(key)))
    print("report ->", rpath)
    print("=============================")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
