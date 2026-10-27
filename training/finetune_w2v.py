"""
Fine-tune a pretrained wav2vec2-style model (MMS / XLS-R) with CTC on the project's manifests.

Reuses the project's CharTokenizer (blank id = 0) and CSV manifests
(audio_path, transcript, language, speaker, duration).
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForCTC, get_linear_schedule_with_warmup

from evaluation.metrics import compute_batch_cer, compute_batch_wer
from tokenizer.tokenizer import CharTokenizer

SR = 16000


class ASRData(Dataset):
    def __init__(self, manifest, tok, max_dur=20.0, min_dur=0.5):
        self.rows, missing = [], 0
        with open(manifest, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                path, text = r["audio_path"].strip(), r["transcript"].strip()
                if not text:
                    continue
                if not os.path.exists(path):
                    missing += 1
                    continue
                dur = float(r["duration"])
                if dur > max_dur or dur < min_dur:
                    continue
                self.rows.append(dict(path=path, text=text, ids=tok.encode(text),
                                      lang=r.get("language", "?"), dur=dur))
        print(f"[data] {manifest}: {len(self.rows)} usable, {missing} missing audio files")
        self.durs = [r["dur"] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav, sr = sf.read(r["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        return wav, r["ids"], r["text"], r["lang"]


def collate(batch):
    max_len = max(len(b[0]) for b in batch)
    max_lab = max(len(b[1]) for b in batch)
    x = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_lab), -100, dtype=torch.long)
    for i, (w, ids, _, _) in enumerate(batch):
        x[i, : len(w)] = torch.from_numpy(w)
        mask[i, : len(w)] = 1
        labels[i, : len(ids)] = torch.tensor(ids)
    return dict(x=x, mask=mask, labels=labels,
                texts=[b[2] for b in batch], langs=[b[3] for b in batch])


def make_batches(durs, max_sec, shuffle, chunk=2000):
    """Length-bucketed batches whose padded audio length stays under max_sec."""
    idx = list(range(len(durs)))
    if shuffle:
        random.shuffle(idx)
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: durs[i])
        cur, mx = [], 0.0
        for i in part:
            if cur and max(mx, durs[i]) * (len(cur) + 1) > max_sec:
                batches.append(cur)
                cur, mx = [], 0.0
            cur.append(i)
            mx = max(mx, durs[i])
        if cur:
            batches.append(cur)
    if shuffle:
        random.shuffle(batches)
    return batches


class BucketBatches:
    def __init__(self, durs, max_sec, shuffle):
        self.durs, self.max_sec, self.shuffle = durs, max_sec, shuffle
        self.n = len(make_batches(durs, max_sec, False))

    def __iter__(self):
        return iter(make_batches(self.durs, self.max_sec, self.shuffle))

    def __len__(self):
        return self.n


@torch.no_grad()
def evaluate(model, loader, tok, device, max_print=3):
    model.eval()
    refs, hyps, langs, total_loss, n = [], [], [], 0.0, 0
    for b in loader:
        x, mask, labels = b["x"].to(device), b["mask"].to(device), b["labels"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(input_values=x, attention_mask=mask, labels=labels)
        total_loss += out.loss.item()
        n += 1
        out_lens = model._get_feat_extract_output_lengths(mask.sum(-1)).tolist()
        pred = out.logits.argmax(-1).cpu()
        for i in range(pred.shape[0]):
            hyps.append(tok.decode_ctc(pred[i, : out_lens[i]].tolist()))
        refs.extend(b["texts"])
        langs.extend(b["langs"])
    res = dict(val_loss=total_loss / max(n, 1),
               wer=compute_batch_wer(refs, hyps), cer=compute_batch_cer(refs, hyps))
    print(f"  [val] loss={res['val_loss']:.3f} WER={res['wer']*100:.1f}% CER={res['cer']*100:.1f}%")
    for lang in sorted(set(langs)):
        ix = [i for i, l in enumerate(langs) if l == lang]
        r, h = [refs[i] for i in ix], [hyps[i] for i in ix]
        print(f"    [{lang}] n={len(ix)} WER={compute_batch_wer(r, h)*100:.1f}% CER={compute_batch_cer(r, h)*100:.1f}%")
    for i in random.sample(range(len(refs)), min(max_print, len(refs))):
        print(f"    REF: {refs[i][:90]}\n    HYP: {hyps[i][:90]}")
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", default="data/metadata/combined_train.csv")
    ap.add_argument("--val_manifest", default="data/metadata/combined_val.csv")
    ap.add_argument("--model", default="facebook/mms-300m")
    ap.add_argument("--out_dir", default="checkpoints_w2v")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_batch_sec", type=float, default=80.0, help="padded audio seconds per micro-batch (lower if OOM)")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max_dur", type=float, default=20.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    tok = CharTokenizer()
    tok.load_vocab("tokenizer/vocab.json")
    print(f"[vocab] {tok.vocab_size} tokens, blank id = {tok.blank_id}")

    train_ds = ASRData(args.train_manifest, tok, args.max_dur)
    val_ds = ASRData(args.val_manifest, tok, args.max_dur)
    train_bs = BucketBatches(train_ds.durs, args.max_batch_sec, True)
    val_bs = BucketBatches(val_ds.durs, args.max_batch_sec, False)
    train_loader = DataLoader(train_ds, batch_sampler=train_bs, collate_fn=collate,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_sampler=val_bs, collate_fn=collate,
                            num_workers=args.num_workers)

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        vocab_size=tok.vocab_size,
        pad_token_id=tok.blank_id,          # HF uses pad_token_id as the CTC blank
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0,
        layerdrop=0.0, mask_time_prob=0.05,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    model.to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_bs) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    print(f"[train] {len(train_bs)} micro-batches/epoch, {steps_per_epoch} optimizer steps/epoch, {total_steps} total")

    start_epoch, best_cer = 1, float("inf")
    last_path = os.path.join(args.out_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_cer = ck["epoch"] + 1, ck["best_cer"]
        print(f"[resume] from epoch {ck['epoch']} (best CER {best_cer*100:.1f}%)")

    model.train()
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        ep0, run, cnt = time.time(), 0.0, 0
        opt.zero_grad()
        for step, b in enumerate(train_loader):
            x, mask, labels = b["x"].to(device), b["mask"].to(device), b["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(input_values=x, attention_mask=mask, labels=labels)
            loss = out.loss / args.accum
            scaler.scale(loss).backward()
            run += out.loss.item(); cnt += 1
            if (step + 1) % args.accum == 0 or (step + 1) == len(train_bs):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            if (step + 1) % 100 == 0:
                print(f"  ep {epoch} [{step+1}/{len(train_bs)}] loss={run/cnt:.3f} lr={sched.get_last_lr()[0]:.2e}")
                run, cnt = 0.0, 0
        print(f"Epoch {epoch}/{args.epochs} done in {(time.time()-ep0)/60:.1f} min (total {(time.time()-t0)/60:.1f} min)")
        res = evaluate(model, val_loader, tok, device)
        if res["cer"] < best_cer:
            best_cer = res["cer"]
            model.save_pretrained(os.path.join(args.out_dir, "best"))
            tok.save_vocab(os.path.join(args.out_dir, "best", "vocab.json"))
            print(f"  ✓ new best (CER {best_cer*100:.1f}%) saved")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, best_cer=best_cer), last_path)
    print(f"Done. Best CER: {best_cer*100:.1f}%")


if __name__ == "__main__":
    main()
