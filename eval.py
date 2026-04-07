"""
Evaluate trained MambaAudioClassifier on RAVDESS test set.
Shows per-emotion accuracy and overall metrics.

Usage:
    python eval.py --data_dir ./data/ravdess --checkpoint ./checkpoints/mamba_audio_best.pt
"""

import os
import argparse
import glob
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict

from app.models.mamba_audio import MambaAudioClassifier
from app.config import settings


# ─── RAVDESS emotion codes ────────────────────────────────────────────────────

EMOTION_LABELS = settings.EMOTION_LABELS
RAVDESS_EMOTION_MAP = {
    "01": 0, "02": 1, "03": 2, "04": 3,
    "05": 4, "06": 5, "07": 6, "08": 7,
}


class RAVDESSEvalDataset(Dataset):
    """Read-only evaluation dataset."""

    def __init__(self, file_paths: list[str], chunk_size: int = 32000):
        self.files = file_paths
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]

        # Extract label from filename
        parts = os.path.basename(path).split("-")
        emotion_code = parts[2]
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

        # Pad or crop
        if len(audio) < self.chunk_size:
            audio = np.pad(audio, (0, self.chunk_size - len(audio)))
        else:
            audio = audio[:self.chunk_size]

        # Normalize
        peak = np.abs(audio).max()
        audio = audio / (peak + 1e-9)

        return torch.from_numpy(audio).float(), label


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for waveform, labels in dataloader:
            waveform = waveform.float().to(device)
            logits = model(waveform)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/ravdess", help="Path to RAVDESS dataset")
    parser.add_argument("--checkpoint", default="./checkpoints/mamba_audio_best.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Find all WAV files
    wav_files = glob.glob(os.path.join(args.data_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No WAV files found in {args.data_dir}")
        return

    print(f"Found {len(wav_files)} files")

    # Load model
    model = MambaAudioClassifier(
        n_classes=len(EMOTION_LABELS),
        d_model=settings.MAMBA_D_MODEL,
        d_state=settings.MAMBA_D_STATE,
        n_layers=settings.MAMBA_N_LAYERS,
    ).float().to(device)

    if os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    # Create dataset and dataloader
    dataset = RAVDESSEvalDataset(wav_files, chunk_size=settings.CHUNK_SIZE)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Evaluate
    print(f"\nEvaluating on {len(dataset)} samples...\n")
    preds, labels = evaluate(model, dataloader, device)

    # Overall accuracy
    overall_acc = accuracy_score(labels, preds)
    print(f"Overall Accuracy: {overall_acc:.1%}\n")

    # Per-emotion accuracy
    print("Per-Emotion Accuracy:")
    print("-" * 50)
    per_emotion_acc = {}
    for emotion_idx, emotion_name in enumerate(EMOTION_LABELS):
        mask = labels == emotion_idx
        if mask.sum() == 0:
            acc = 0.0
            count = 0
        else:
            acc = (preds[mask] == emotion_idx).mean()
            count = mask.sum()
        per_emotion_acc[emotion_name] = (acc, count)

        # Bar chart
        bar_len = int(acc * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"{emotion_name:12} {bar} {acc:5.1%}  ({count} samples)")

    print("\n" + "=" * 50)
    print("Confusion Matrix:")
    print("-" * 50)
    cm = confusion_matrix(labels, preds)
    print(cm)

    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(labels, preds, target_names=EMOTION_LABELS))


if __name__ == "__main__":
    main()
