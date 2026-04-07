"""
Auto-label audio clips by emotion using trained MambaAudioClassifier.
Splits audio into chunks, predicts emotion for each, saves to labeled folders.

Usage:
    python prepare_serial.py \
      --input ./data/serial_audio.wav \
      --checkpoint ./checkpoints/mamba_audio_best.pt \
      --output ./data/serial/labeled
"""

import os
import argparse
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from app.models.mamba_audio import MambaAudioClassifier
from app.config import settings


def prepare_serial(input_file, checkpoint, output_dir, chunk_duration_sec=5, confidence_threshold=0.0):
    """
    Process long audio file, label each chunk by emotion, save to folders.

    Args:
        input_file: Path to input WAV file
        checkpoint: Path to trained model checkpoint
        output_dir: Root directory to save labeled chunks
        chunk_duration_sec: Duration of each chunk (5 seconds = 80,000 samples at 16kHz)
        confidence_threshold: Only save if confidence > this (0.0 = save all)
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load audio
    print(f"\nLoading audio: {input_file}")
    audio, sr = sf.read(input_file, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        print(f"Resampling from {sr}Hz to 16000Hz...")
        import torchaudio
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, 16000)
        audio = t.squeeze(0).numpy()

    duration_sec = len(audio) / 16000
    print(f"Duration: {duration_sec:.1f} seconds ({len(audio)} samples)")

    # Load model
    print(f"\nLoading checkpoint: {checkpoint}")
    model = MambaAudioClassifier(
        n_classes=len(settings.EMOTION_LABELS),
        d_model=settings.MAMBA_D_MODEL,
        d_state=settings.MAMBA_D_STATE,
        n_layers=settings.MAMBA_N_LAYERS,
    ).float().to(device)

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    emotion_dirs = {}
    for emotion in settings.EMOTION_LABELS:
        emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        emotion_dirs[emotion] = emotion_dir

    # Process chunks
    chunk_samples = int(chunk_duration_sec * 16000)
    n_chunks = len(audio) // chunk_samples
    print(f"\nProcessing {n_chunks} chunks ({chunk_duration_sec}s each)...\n")

    chunk_idx = 0
    saved_count = {e: 0 for e in settings.EMOTION_LABELS}

    with torch.no_grad():
        for start in range(0, len(audio) - chunk_samples, chunk_samples):
            chunk = audio[start:start + chunk_samples]

            # Normalize
            peak = np.abs(chunk).max()
            chunk = chunk / (peak + 1e-9)

            # Predict
            chunk_t = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
            probs = model.predict_proba(chunk_t)
            probs_np = probs[0].cpu().numpy()

            emotion_idx = probs_np.argmax()
            emotion_name = settings.EMOTION_LABELS[emotion_idx]
            confidence = float(probs_np[emotion_idx])

            # Save if above threshold
            if confidence >= confidence_threshold:
                output_path = os.path.join(
                    emotion_dirs[emotion_name],
                    f"chunk_{chunk_idx:06d}_{emotion_name}_{confidence:.2f}.wav"
                )
                sf.write(output_path, chunk, 16000)
                saved_count[emotion_name] += 1

                # Progress bar
                if (chunk_idx + 1) % 50 == 0:
                    print(f"Processed {chunk_idx + 1}/{n_chunks} chunks")

            chunk_idx += 1

    # Summary
    print("\n" + "=" * 60)
    print("Labeling Complete")
    print("=" * 60)
    total_saved = sum(saved_count.values())
    print(f"Total chunks saved: {total_saved}")
    print("\nBreakdown:")
    for emotion in settings.EMOTION_LABELS:
        count = saved_count[emotion]
        pct = 100 * count / total_saved if total_saved > 0 else 0
        print(f"  {emotion:12} {count:4d}  ({pct:.1f}%)")

    print(f"\nOutput directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--checkpoint", default="./checkpoints/mamba_audio_best.pt")
    parser.add_argument("--output", default="./data/serial/labeled")
    parser.add_argument("--chunk_duration", type=float, default=5.0, help="Chunk duration in seconds")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Min confidence to save")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    prepare_serial(
        input_file=args.input,
        checkpoint=args.checkpoint,
        output_dir=args.output,
        chunk_duration_sec=args.chunk_duration,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
