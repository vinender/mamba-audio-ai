"""
Fine-tune MambaAudioClassifier on RAVDESS emotion dataset.

RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song
Download: https://zenodo.org/record/1188976
Labels: neutral, calm, happy, sad, angry, fearful, disgust, surprised

Usage:
    python train.py --data_dir ./data/ravdess --epochs 30 --output ./checkpoints
"""

import os
import argparse
import glob
import time
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from app.models.mamba_audio import MambaAudioClassifier, train_one_epoch
from app.config import settings


# ─── RAVDESS Dataset ──────────────────────────────────────────────────────────

# RAVDESS filename format: 03-01-06-01-02-01-12.wav
# Position 3 (index 2) = emotion: 01=neutral, 02=calm, 03=happy, 04=sad,
#                                  05=angry, 06=fearful, 07=disgust, 08=surprised
RAVDESS_EMOTION_MAP = {
    "01": 0,  # neutral
    "02": 0,  # calm → neutral
    "03": 1,  # happy
    "04": 2,  # sad
    "05": 3,  # angry
    "06": 4,  # fearful
    "07": 3,  # disgust → angry
    "08": 4,  # surprised → fearful
}

CREMAD_EMOTION_MAP = {
    "NEU": 0,  # neutral
    "HAP": 1,  # happy
    "SAD": 2,  # sad
    "ANG": 3,  # angry
    "FEA": 4,  # fearful
    "DIS": 3,  # disgust → angry
}


class RAVDESSDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str],
        chunk_size: int = 32000,   # 2 seconds at 16kHz
        augment: bool = False,
    ):
        self.files = file_paths
        self.chunk_size = chunk_size
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        filename = os.path.basename(path)

        # Detect dataset by filename format
        # RAVDESS: 03-01-06-01-02-01-12.wav (dash-separated, numeric)
        # CREMA-D: 1001_DFA_ANG_XX.wav (underscore-separated, alphanumeric)

        if "_" in filename:
            # CREMA-D format
            parts = filename.split("_")
            emotion_code = parts[2] if len(parts) > 2 else "NEU"
            label = CREMAD_EMOTION_MAP.get(emotion_code, 0)
        else:
            # RAVDESS format
            parts = filename.split("-")
            emotion_code = parts[2] if len(parts) > 2 else "01"
            label = RAVDESS_EMOTION_MAP.get(emotion_code, 0)

        # Load audio
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != settings.SAMPLE_RATE:
            import torchaudio
            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, settings.SAMPLE_RATE)
            audio = t.squeeze(0).numpy()

        # Pad or crop to chunk_size
        if len(audio) < self.chunk_size:
            audio = np.pad(audio, (0, self.chunk_size - len(audio)))
        else:
            # Random crop during training
            if self.augment and len(audio) > self.chunk_size:
                start = np.random.randint(0, len(audio) - self.chunk_size)
                audio = audio[start:start + self.chunk_size]
            else:
                audio = audio[:self.chunk_size]

        # Normalize
        peak = np.abs(audio).max()
        audio = audio / (peak + 1e-9)

        # Data augmentation
        if self.augment:
            audio = self._augment(audio)

        return torch.from_numpy(audio), label

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """Simple augmentations for robustness."""
        # Random volume scale
        audio = audio * np.random.uniform(0.7, 1.3)

        # Additive Gaussian noise
        if np.random.rand() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise

        # Random time shift
        if np.random.rand() < 0.3:
            shift = np.random.randint(-1600, 1600)  # ±100ms
            audio = np.roll(audio, shift)

        return np.clip(audio, -1.0, 1.0)


# ─── Training loop ────────────────────────────────────────────────────────────

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for waveform, labels in dataloader:
            waveform, labels = waveform.to(device), labels.to(device)
            logits = model(waveform)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Load dataset
    all_files = glob.glob(os.path.join(args.data_dir, "**/*.wav"), recursive=True)
    print(f"Found {len(all_files)} audio files")

    if not all_files:
        print(f"No WAV files found in {args.data_dir}")
        print("Download RAVDESS: https://zenodo.org/record/1188976")
        return

    train_files, val_files = train_test_split(all_files, test_size=0.15, random_state=42)
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    train_ds = RAVDESSDataset(train_files, augment=True)
    val_ds = RAVDESSDataset(val_files, augment=False)

    # Balance dataset with weighted sampling
    from torch.utils.data import WeightedRandomSampler

    # Count samples per class
    class_counts = [0] * 5
    for f in train_files:
        fname = os.path.basename(f)
        if "_" in fname:
            code = fname.split("_")[2]
            label = CREMAD_EMOTION_MAP.get(code, 0)
        else:
            code = fname.split("-")[2]
            label = RAVDESS_EMOTION_MAP.get(code, 0)
        class_counts[label] += 1

    print(f"Class distribution: {dict(zip(settings.EMOTION_LABELS, class_counts))}")

    # Weight = inverse of class frequency
    weights = [1.0 / max(class_counts[i], 1) for i in range(5)]
    sample_weights = []
    for f in train_files:
        fname = os.path.basename(f)
        if "_" in fname:
            code = fname.split("_")[2]
            label = CREMAD_EMOTION_MAP.get(code, 0)
        else:
            code = fname.split("-")[2]
            label = RAVDESS_EMOTION_MAP.get(code, 0)
        sample_weights.append(weights[label])

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = MambaAudioClassifier(
        n_classes=5,
        d_model=args.d_model,
        d_state=64,
        n_layers=args.n_layers,
    ).float().to(device)  # force float32 on all weights

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train_loss = train_one_epoch(model, train_dl, optimizer, device)
        val_acc = evaluate(model, val_dl, device)
        scheduler.step()

        elapsed = time.time() - t
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val acc: {val_acc:.3f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.output, "mamba_audio_best.pt")
            model.save_checkpoint(ckpt_path, epoch=epoch, loss=train_loss)
            print(f"  ✓ New best: {best_acc:.3f} → saved to {ckpt_path}")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.3f}")
    print(f"Update MAMBA_CHECKPOINT in .env or config.py to use the trained model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/ravdess", help="Path to RAVDESS dataset")
    parser.add_argument("--output", default="./checkpoints", help="Output dir for checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    args = parser.parse_args()
    main(args)
