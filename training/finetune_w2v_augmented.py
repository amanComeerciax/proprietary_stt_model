"""
Augmented Fine-Tuning of Wav2Vec2 / MMS-300M for Real-World Microphone Robustness.

Adds dynamic online acoustic augmentations:
1. Room Impulse Response (Reverberation / Echo)
2. Additive Background Noise (Fan / AC / Room hum)
3. Volume / Gain Perturbation

Safety & Non-Destructive:
- Does NOT overwrite existing checkpoints.
- Continues fine-tuning from our current best model checkpoint.
- Outputs exclusively to a separate directory (e.g., checkpoints_augmented/).
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
from scipy.signal import fftconvolve
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForCTC, get_linear_schedule_with_warmup

from evaluation.metrics import compute_batch_cer, compute_batch_wer
from tokenizer.tokenizer import CharTokenizer

SR = 16000


def apply_room_reverb(wav: np.ndarray, sr: int = 16000, rt60: float = 0.25) -> np.ndarray:
    """Simulate realistic room impulse response (reverberation) using an exponential decay model."""
    length = max(int(sr * rt60), 100)
    t = np.linspace(0, rt60, length)
    decay = np.exp(-6.9 * t / max(rt60, 0.01))
    noise = np.random.randn(length)
    rir = decay * noise
    rir[0] = 1.0  # Direct path impulse
    rir = rir / np.sqrt(np.sum(rir ** 2) + 1e-9)
    reverbed = fftconvolve(wav, rir, mode="full")[: len(wav)]
    return (0.6 * wav + 0.4 * reverbed).astype(np.float32)


def apply_noise(wav: np.ndarray, min_snr: float = 12.0, max_snr: float = 25.0) -> np.ndarray:
    """Add ambient background noise at a random Signal-to-Noise Ratio (SNR)."""
    snr = random.uniform(min_snr, max_snr)
    signal_power = np.mean(wav ** 2) + 1e-9
    noise_power = signal_power / (10 ** (snr / 10.0))
    noise = np.random.randn(len(wav)) * np.sqrt(noise_power)
    return (wav + noise).astype(np.float32)


def apply_gain(wav: np.ndarray, min_gain: float = 0.75, max_gain: float = 1.25) -> np.ndarray:
    """Apply random volume scaling jitter to simulate varying speaker distance."""
    gain = random.uniform(min_gain, max_gain)
    return (wav * gain).astype(np.float32)


class ASRDataAugmented(Dataset):
    """Dataset with dynamic online acoustic augmentations for microphone adaptation."""

    def __init__(self, manifest, tok, max_dur=20.0, min_dur=0.5, augment=False):
        self.rows, missing = [], 0
        self.augment = augment
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
                self.rows.append(
                    dict(
                        path=path,
                        text=text,
                        ids=tok.encode(text),
                        lang=r.get("language", "?"),
                        dur=dur,
                    )
                )
        print(f"[data] {manifest}: {len(self.rows)} usable, {missing} missing (augment={augment})")
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

        # Dynamic online augmentations (applied ONLY during training)
        if self.augment:
            # 1. Volume Jitter (50% probability)
            if random.random() < 0.5:
                wav = apply_gain(wav, 0.75, 1.25)
            # 2. Room Reverberation / Echo (40% probability)
            if random.random() < 0.4:
                rt60 = random.uniform(0.15, 0.35)
                wav = apply_room_reverb(wav, sr=SR, rt60=rt60)
            # 3. Additive Background Noise (50% probability)
            if random.random() < 0.5:
                wav = apply_noise(wav, min_snr=12.0, max_snr=25.0)

        # Standardize matching Wav2Vec2 training
        wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)
        return wav.astype(np.float32), r["ids"], r["text"], r["lang"]


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
    return dict(
        x=x,
        mask=mask,
        labels=labels,
        texts=[b[2] for b in batch],
        langs=[b[3] for b in batch],
    )


def make_batches(durs, max_sec, shuffle, chunk=2000):
    """Length-bucketed batches whose padded audio length stays under max_sec."""
    idx = list(range(len(durs)))
    if shuffle:
        random.shuffle(idx)
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s : s + chunk], key=lambda i: durs[i])
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
    res = dict(
        val_loss=total_loss / max(n, 1),
        wer=compute_batch_wer(refs, hyps),
        cer=compute_batch_cer(refs, hyps),
    )
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
    # By default, fine-tune from our best checkpoint
    default_model = "checkpoints/stt_w2v/best" if Path("checkpoints/stt_w2v/best").exists() else "facebook/mms-300m"
    ap.add_argument("--model", default=default_model, help="Base checkpoint to continue training from")
    ap.add_argument("--out_dir", default="checkpoints_augmented", help="Separate destination directory for new checkpoints")
    ap.add_argument("--epochs", type=int, default=3, help="Focused fine-tuning epochs (2-3 recommended)")
    ap.add_argument("--lr", type=float, default=3e-5, help="Gentle learning rate to preserve existing phonetic capabilities")
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--max_batch_sec", type=float, default=80.0, help="Padded audio seconds per micro-batch")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max_dur", type=float, default=20.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_augment", action="store_true", help="Disable online noise/reverb augmentations if set")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    tok = CharTokenizer()
    tok.load_vocab("tokenizer/vocab.json")
    print(f"[vocab] {tok.vocab_size} tokens, blank id = {tok.blank_id}")

    # Train dataset has augment=True, validation dataset has augment=False (clean evaluation)
    train_ds = ASRDataAugmented(
        args.train_manifest, tok, args.max_dur, augment=not args.no_augment
    )
    val_ds = ASRDataAugmented(
        args.val_manifest, tok, args.max_dur, augment=False
    )
    train_bs = BucketBatches(train_ds.durs, args.max_batch_sec, True)
    val_bs = BucketBatches(val_ds.durs, args.max_batch_sec, False)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_bs,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_bs, collate_fn=collate, num_workers=args.num_workers
    )

    print(f"[init] Loading base checkpoint from: {args.model}")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        vocab_size=tok.vocab_size,
        pad_token_id=tok.blank_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        mask_time_prob=0.08,  # SpecAugment time masking
        mask_time_length=10,
        mask_feature_prob=0.04,  # SpecAugment frequency masking
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    model.to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters loaded on {device}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train_bs) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(args.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    print(
        f"[train] {len(train_bs)} batches/epoch, {steps_per_epoch} opt steps/epoch, {total_steps} total"
    )

    # Initial baseline check before training
    print("\n--- Initial Model Validation (Before Augmentation Training) ---")
    initial_res = evaluate(model, val_loader, tok, device)
    best_cer = initial_res["cer"]
    print(f"--- Baseline Validation CER: {best_cer*100:.1f}% ---\n")

    model.train()
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep0, run, cnt = time.time(), 0.0, 0
        opt.zero_grad()
        for step, b in enumerate(train_loader):
            x, mask, labels = (
                b["x"].to(device),
                b["mask"].to(device),
                b["labels"].to(device),
            )
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(input_values=x, attention_mask=mask, labels=labels)
            loss = out.loss / args.accum
            scaler.scale(loss).backward()
            run += out.loss.item()
            cnt += 1
            if (step + 1) % args.accum == 0 or (step + 1) == len(train_bs):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
            if (step + 1) % 100 == 0:
                print(
                    f"  ep {epoch} [{step+1}/{len(train_bs)}] loss={run/cnt:.3f} lr={sched.get_last_lr()[0]:.2e}"
                )
                run, cnt = 0.0, 0
        print(
            f"Epoch {epoch}/{args.epochs} done in {(time.time()-ep0)/60:.1f} min (total {(time.time()-t0)/60:.1f} min)"
        )
        res = evaluate(model, val_loader, tok, device)
        if res["cer"] < best_cer:
            best_cer = res["cer"]
            best_dir = os.path.join(args.out_dir, "best")
            model.save_pretrained(best_dir)
            tok.save_vocab(os.path.join(best_dir, "vocab.json"))
            print(f"  ✓ New Best Model saved to {best_dir} (CER {best_cer*100:.1f}%)")

        torch.save(
            dict(
                model=model.state_dict(),
                opt=opt.state_dict(),
                sched=sched.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                best_cer=best_cer,
            ),
            os.path.join(args.out_dir, "last.pt"),
        )
    print(f"\n[Finished] Augmented Fine-Tuning Complete! Best CER: {best_cer*100:.1f}%")
    print(f"Model saved in: {os.path.join(args.out_dir, 'best')}")


if __name__ == "__main__":
    main()
