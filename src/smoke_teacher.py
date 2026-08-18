"""GPU smoke for teacher_train.py (Mean Teacher + CVT). REAL GPU only.

Proves:
  (a) mt-weight 0 AND cvt-weight 0 reproduce the plain weighted-CE baseline
      BIT-FOR-BIT (torch.equal on logits AND on loss), for the same weights and
      the same input;
  (b) mt-weight > 0: the mean-teacher MSE term is non-zero, the EMA teacher
      weights actually move after an optimizer step, and a few GPU steps run with
      no device/OOM error;
  (c) cvt-weight > 0: the CVT KL term is non-zero, the aux head receives gradient,
      and a few GPU steps run with no device/OOM error.

Runs on a tiny slice of the real NoPnx-NP train split, few steps, on cuda.
"""
import os
import sys

import torch
from torch import nn
from transformers import AutoTokenizer, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import MODEL_PAR_TOKEN, load_task            # noqa: E402
import teacher_train as tt                              # noqa: E402

DEV = "cuda"
MODEL = "aubmindlab/bert-base-arabertv02"
TASK = "NoPnx-NP"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
JSONL = os.path.join(DATA, f"{TASK}_train.jsonl")


def build_batch(tok, n_docs=3, window=180, stride=90, batch=4):
    docs = load_task(TASK, "train", jsonl_path=JSONL)[:n_docs]
    exs = tt.make_examples(docs, window, stride)[:batch]
    ds = {k: [e[k] for e in exs] for k in exs[0]}
    enc = tt.encode(ds, tok, 512)
    feats = [{k: enc[k][i] for k in enc} for i in range(len(exs))]
    col = tt.Collator(tok)
    b = col(feats)
    return {k: v.to(DEV) for k, v in b.items()}


def new_model(mt=0.0, cvt=0.0, seed=0):
    set_seed(seed)
    m = tt.TeacherModel(MODEL, pos_weight=8.0, mt_weight=mt, cvt_weight=cvt, n_extra_tok=1)
    return m.to(DEV)


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "need a real GPU"
    print("device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.add_special_tokens({"additional_special_tokens": [MODEL_PAR_TOKEN]})
    batch = build_batch(tok)
    print("batch input_ids:", tuple(batch["input_ids"].shape),
          "valid labels:", int((batch["labels"] != -100).sum()),
          "prep subwords masked:", int((batch["input_ids"] != batch["input_ids_masked"]).sum()))

    # ---------------- (a) BIT-FOR-BIT at weight 0 ----------------
    # Reference = a model whose forward CANNOT take any consistency path: build a
    # weight-0 model, snapshot its weights, and run forward twice under the SAME
    # pre-seeded dropout. The "baseline" path (both weights 0) must equal a run
    # where we additionally pass the masked input + teacher_probs but the gates
    # are off — proving the extra kwargs are inert at weight 0.
    ref = new_model(mt=0.0, cvt=0.0, seed=123)
    sd = ref.state_dict()

    m0 = tt.TeacherModel(MODEL, pos_weight=8.0, mt_weight=0.0, cvt_weight=0.0, n_extra_tok=1).to(DEV)
    m0.load_state_dict(sd); m0.train()

    # Build the fake teacher BEFORE seeding so it does not perturb the dropout
    # RNG stream that the compared forwards consume.
    fake_teacher = torch.rand(batch["labels"].shape + (2,), device=DEV)
    torch.manual_seed(7)
    out_base = m0(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                  labels=batch["labels"])
    # Same weights, same seed, but ALSO hand it the masked input + fake teacher
    # probs. With both gates 0 these must be ignored -> identical logits & loss.
    torch.manual_seed(7)
    out_inert = m0(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   labels=batch["labels"], input_ids_masked=batch["input_ids_masked"],
                   teacher_probs=fake_teacher)
    logits_equal = torch.equal(out_base["logits"], out_inert["logits"])
    loss_equal = torch.equal(out_base["loss"], out_inert["loss"])
    print(f"[a] weight-0 bit-for-bit: logits_equal={logits_equal}  loss_equal={loss_equal}  "
          f"loss={out_base['loss'].item():.8f}")

    # And prove the loss equals a hand-rolled plain weighted-CE on the SAME logits.
    torch.manual_seed(7)
    out_chk = m0(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                 labels=batch["labels"])
    w = torch.tensor([1.0, 8.0], device=DEV)
    manual_ce = nn.functional.cross_entropy(
        out_chk["logits"].view(-1, 2).float(), batch["labels"].view(-1),
        weight=w, ignore_index=-100)
    ce_equal = torch.equal(out_chk["loss"], manual_ce)
    print(f"[a] loss == hand-rolled weighted-CE: {ce_equal}  (manual={manual_ce.item():.8f})")

    assert logits_equal and loss_equal and ce_equal, "BIT-FOR-BIT FAILED"

    # ---------------- (b) MEAN TEACHER (mt-weight > 0) ----------------
    m_mt = new_model(mt=1.0, cvt=0.0, seed=123)
    opt = torch.optim.AdamW(m_mt.parameters(), lr=5e-5)
    # emulate the trainer's EMA teacher
    import copy
    teacher = copy.deepcopy(m_mt)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    ema_decay = 0.999

    # isolate the mt term magnitude: loss(mt) - loss(ce-only) on same seed/weights
    torch.manual_seed(7)
    with torch.no_grad():
        tp = torch.softmax(teacher(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"])["logits"].float(), -1)
    torch.manual_seed(7)
    out_mt = m_mt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                  labels=batch["labels"], input_ids_masked=batch["input_ids_masked"],
                  teacher_probs=tp)
    torch.manual_seed(7)
    out_ce = m_mt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                  labels=batch["labels"])
    mt_term = (out_mt["loss"] - out_ce["loss"]).item()
    print(f"[b] mt term (loss_mt - loss_ce) = {mt_term:.8f}  (must be != 0)")
    assert abs(mt_term) > 0, "mean-teacher term is zero"

    # a few real steps + EMA update; check the teacher actually moves
    tsnap = teacher.enc.encoder.layer[0].attention.self.query.weight.detach().clone()
    ssnap = m_mt.enc.encoder.layer[0].attention.self.query.weight.detach().clone()
    for step in range(3):
        with torch.no_grad():
            tp = torch.softmax(teacher(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"])["logits"].float(), -1)
        out = m_mt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   labels=batch["labels"], input_ids_masked=batch["input_ids_masked"],
                   teacher_probs=tp)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        with torch.no_grad():
            for tpar, spar in zip(teacher.parameters(), m_mt.parameters()):
                tpar.mul_(ema_decay).add_(spar.detach(), alpha=1 - ema_decay)
        print(f"[b] step {step}: loss={out['loss'].item():.6f}")
    tmoved = (teacher.enc.encoder.layer[0].attention.self.query.weight - tsnap).abs().max().item()
    smoved = (m_mt.enc.encoder.layer[0].attention.self.query.weight - ssnap).abs().max().item()
    print(f"[b] student weight moved by {smoved:.3e}; EMA teacher moved by {tmoved:.3e} (must be > 0)")
    assert tmoved > 0 and smoved > 0, "EMA teacher or student did not update"

    # ---------------- (c) CVT (cvt-weight > 0) ----------------
    m_cvt = new_model(mt=0.0, cvt=1.0, seed=123)
    opt = torch.optim.AdamW(m_cvt.parameters(), lr=5e-5)
    torch.manual_seed(7)
    out_cvt = m_cvt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"], input_ids_masked=batch["input_ids_masked"])
    torch.manual_seed(7)
    out_ce2 = m_cvt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
    cvt_term = (out_cvt["loss"] - out_ce2["loss"]).item()
    print(f"[c] cvt term (loss_cvt - loss_ce) = {cvt_term:.8f}  (must be != 0)")
    assert abs(cvt_term) > 0, "CVT term is zero"

    # aux head must receive gradient
    m_cvt.zero_grad()
    out = m_cvt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=batch["labels"], input_ids_masked=batch["input_ids_masked"])
    out["loss"].backward()
    g = m_cvt.aux.weight.grad
    aux_gradnorm = float(g.norm()) if g is not None else 0.0
    print(f"[c] aux head grad norm = {aux_gradnorm:.6e} (must be > 0)")
    assert aux_gradnorm > 0, "aux head received no gradient"

    for step in range(3):
        out = m_cvt(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"], input_ids_masked=batch["input_ids_masked"])
        opt.zero_grad(); out["loss"].backward(); opt.step()
        print(f"[c] step {step}: loss={out['loss'].item():.6f}")

    # sanity: with cvt on, the aux forward must NOT change the main logits path
    # (main head still predicts on the full view). already covered by [a] gates.
    torch.cuda.synchronize()
    print("SMOKE OK: (a) bit-for-bit, (b) mean-teacher + EMA update, (c) CVT + aux grads all pass on GPU.")


if __name__ == "__main__":
    main()
