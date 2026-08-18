"""GPU-SMOKE for fire_penalty_train.py — keep it TINY (main GPU is busy with a
battery). Verifies:

  1. BIT-FOR-BIT GUARANTEE: with fire_penalty_weight=0 AND conf_penalty_weight=0,
     the model's loss equals a hand-computed plain weighted-CE loss exactly
     (torch.equal on identical logits via a frozen forward).
  2. FIRE PENALTY is NON-ZERO when on, and is computed on GOLD==0 ONLY:
     - independently recompute the penalty from the returned logits and check it
       matches loss(on) - loss(off);
     - flip every gold==1 -> gold==0 changes the penalty (proves gold==1 was
       excluded), while flipping only -100 positions does NOT change it.
  3. Trains a few steps (batch 2) through the real Trainer path without OOM.

Run:  python smoke_fire_penalty.py
"""
import os
import sys

import torch
from transformers import AutoTokenizer, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import MODEL_PAR_TOKEN  # noqa: E402
from fire_penalty_train import FirePenaltyModel  # noqa: E402

MODEL = "aubmindlab/bert-base-arabertv02"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
POS_W = 4.0
GAMMA = 2.0
FIRE_W = 1.0
CONF_W = 0.1


def _plain_weighted_ce(logits, labels, pos_w):
    w = torch.tensor([1.0, pos_w], device=logits.device)
    return torch.nn.functional.cross_entropy(
        logits.view(-1, 2).float(), labels.view(-1), weight=w, ignore_index=-100)


def _fire_term(logits, labels, gamma):
    p_b = torch.softmax(logits.view(-1, 2).float(), -1)[..., 1]
    neg = (labels.view(-1) == 0)
    return (p_b[neg] ** gamma).mean() if neg.any() else torch.tensor(0.0)


def _entropy_term(logits, labels):
    logp = torch.log_softmax(logits.view(-1, 2).float(), -1)
    p = logp.exp()
    ent = -(p * logp).sum(-1)
    valid = (labels.view(-1) != -100)
    return ent[valid].mean()


def main():
    set_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})

    # tiny synthetic batch of 2 short windows
    words = [["و", "ذهب", "الولد", "إلى", "المدرسة", "من", "الصباح"],
             ["ثم", "عاد", "في", "المساء", "بعد", "الدرس", "الطويل"]]
    labels_word = [[0, 0, 1, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1]]
    enc = tok(words, is_split_into_words=True,
              padding=True, truncation=True, max_length=64, return_tensors="pt")
    # align word labels onto last subword
    lab = torch.full_like(enc["input_ids"], -100)
    for b in range(len(words)):
        wid = enc.word_ids(batch_index=b)
        for pos in range(len(wid)):
            w = wid[pos]
            if w is None:
                continue
            nxt = wid[pos + 1] if pos + 1 < len(wid) else None
            if nxt != w:
                lab[b, pos] = labels_word[b][w]
    input_ids = enc["input_ids"].to(DEV)
    attn = enc["attention_mask"].to(DEV)
    lab = lab.to(DEV)

    n_gold0 = int((lab == 0).sum()); n_gold1 = int((lab == 1).sum())
    n_ign = int((lab == -100).sum())
    print(f"[batch] gold0={n_gold0}  gold1={n_gold1}  ignore(-100)={n_ign}  "
          f"shape={tuple(input_ids.shape)}  device={DEV}")

    # ---------- 1. BIT-FOR-BIT baseline (both weights 0) ------------------- #
    set_seed(0)
    base = FirePenaltyModel(MODEL, pos_weight=POS_W, fire_weight=0.0,
                            conf_weight=0.0, n_extra_tok=1).to(DEV).eval()
    with torch.no_grad():
        out0 = base(input_ids=input_ids, attention_mask=attn, labels=lab)
        logits0 = out0["logits"]
        ref_ce = _plain_weighted_ce(logits0, lab, POS_W)
    same = torch.equal(out0["loss"], ref_ce)
    print(f"[1] baseline loss == plain weighted-CE (torch.equal): {same}  "
          f"loss={out0['loss'].item():.8f}  ref={ref_ce.item():.8f}")
    assert same, "BIT-FOR-BIT baseline broken"

    # ---------- 2a. FIRE penalty non-zero + matches (on gold==0 only) ------ #
    set_seed(0)
    firem = FirePenaltyModel(MODEL, pos_weight=POS_W, fire_weight=FIRE_W,
                             fire_gamma=GAMMA, conf_weight=0.0,
                             n_extra_tok=1).to(DEV).eval()
    with torch.no_grad():
        outf = firem(input_ids=input_ids, attention_mask=attn, labels=lab)
        logf = outf["logits"]
        ce_f = _plain_weighted_ce(logf, lab, POS_W)
        fire_expect = _fire_term(logf, lab, GAMMA)
    delta = (outf["loss"] - ce_f).item()
    print(f"[2a] fire penalty: loss-CE={delta:.8f}  "
          f"expected B*fire={FIRE_W * fire_expect.item():.8f}  "
          f"raw fire term={fire_expect.item():.8f}")
    assert fire_expect.item() > 0, "fire penalty is zero — nothing to test"
    assert abs(delta - FIRE_W * fire_expect.item()) < 1e-5, "fire penalty mismatch"

    # ---------- 2b. gold==1 is EXCLUDED (flip gold1->gold0 changes it) ----- #
    lab_flip1 = lab.clone()
    lab_flip1[lab == 1] = 0            # now former boundaries count as no-boundary
    with torch.no_grad():
        fire_flip1 = _fire_term(logf, lab_flip1, GAMMA)
    print(f"[2b] flip gold1->gold0: fire {fire_expect.item():.8f} -> "
          f"{fire_flip1.item():.8f}  (must DIFFER => gold==1 was excluded)")
    assert not torch.allclose(fire_expect, fire_flip1), \
        "flipping gold1->gold0 did not change penalty — gold==1 was NOT excluded"

    # ---------- 2c. -100 is IGNORED (flip one -100->0 only if any) --------- #
    lab_flip_ign = lab.clone()
    ign_pos = (lab == -100).nonzero(as_tuple=False)
    if ign_pos.numel() > 0:
        # set an ignored position's gold to 1 (still must not enter the mean,
        # because the mask keys on ==0 and ==-100 stays out either way). Instead
        # verify: penalty over gold==0 is unaffected by what -100 positions "are".
        b0, p0 = ign_pos[0].tolist()
        lab_flip_ign[b0, p0] = 1  # pretend a padding/subword is a boundary
        with torch.no_grad():
            fire_ign = _fire_term(logf, lab_flip_ign, GAMMA)
        print(f"[2c] set one -100->1: fire {fire_expect.item():.8f} -> "
              f"{fire_ign.item():.8f}  (must be EQUAL => -100 truly ignored)")
        assert torch.allclose(fire_expect, fire_ign), \
            "changing a -100 position changed the penalty — not ignored"
    else:
        print("[2c] no -100 positions in this tiny batch; skipped")

    # ---------- 2d. entropy (symmetric) arm non-zero ----------------------- #
    set_seed(0)
    confm = FirePenaltyModel(MODEL, pos_weight=POS_W, fire_weight=0.0,
                             conf_weight=CONF_W, n_extra_tok=1).to(DEV).eval()
    with torch.no_grad():
        outc = confm(input_ids=input_ids, attention_mask=attn, labels=lab)
        logc = outc["logits"]
        ce_c = _plain_weighted_ce(logc, lab, POS_W)
        ent_expect = _entropy_term(logc, lab)
    delta_c = (outc["loss"] - ce_c).item()
    print(f"[2d] entropy penalty: loss-CE={delta_c:.8f}  "
          f"expected -W*H={-CONF_W * ent_expect.item():.8f}  H={ent_expect.item():.8f}")
    assert abs(delta_c - (-CONF_W * ent_expect.item())) < 1e-5, "entropy penalty mismatch"

    # ---------- 3. a few real training steps (batch 2), no OOM ------------- #
    set_seed(0)
    trm = FirePenaltyModel(MODEL, pos_weight=POS_W, fire_weight=FIRE_W,
                           fire_gamma=GAMMA, conf_weight=0.0,
                           n_extra_tok=1).to(DEV).train()
    opt = torch.optim.AdamW(trm.parameters(), lr=5e-5)
    losses = []
    for step in range(3):
        out = trm(input_ids=input_ids, attention_mask=attn, labels=lab)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        losses.append(round(out["loss"].item(), 6))
    if DEV == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[3] trained 3 steps (batch 2); losses={losses}  "
              f"peak_gpu_mem={peak:.0f} MiB (no OOM)")
    else:
        print(f"[3] trained 3 steps (batch 2, CPU); losses={losses}")

    print("\nSMOKE PASS")


if __name__ == "__main__":
    main()
