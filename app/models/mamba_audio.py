"""
Mamba-based audio sequence model.

Architecture:
  Raw waveform → linear projection → N x Mamba2 blocks → global pool → classifier head

Why Mamba for audio:
  - Audio at 16kHz = 16,000 samples/sec. A 30-sec clip = 480,000 tokens.
  - Transformers hit OOM at ~4,000 tokens for a mid-GPU.
  - Mamba is O(n) in time AND memory — processes the full waveform, no chunking hacks.
  - At inference it runs as a recurrent model: O(1) memory per step. Perfect for streaming.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[MambaAudio] mamba-ssm not installed. Using fallback GRU model for development.")


class MambaBlock(nn.Module):
    """Single Mamba2 block with pre-norm and residual."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if MAMBA_AVAILABLE:
            self.ssm = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # CPU-friendly fallback for dev without CUDA
            self.ssm = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if MAMBA_AVAILABLE:
            x = self.ssm(x)
        else:
            x, _ = self.ssm(x)
        return x + residual


class MambaAudioClassifier(nn.Module):
    """
    Full model: raw waveform → emotion label probabilities.

    Input:  (batch, samples)          — raw float32 PCM, normalized [-1, 1]
    Output: (batch, n_classes)        — softmax probabilities
    """

    def __init__(
        self,
        n_classes: int = 8,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 6,
        sample_rate: int = 16000,
        input_patch_size: int = 160,   # 10ms patches at 16kHz (hop_length of mel)
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = input_patch_size

        # 1. Patch embedding: split waveform into 10ms patches, project to d_model
        #    This is like "tokenizing" audio into fixed-size frames
        self.patch_embed = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=input_patch_size,
                stride=input_patch_size,
                padding=0,
                bias=False,
            ),
            nn.GELU(),
        )

        # 2. Positional encoding (learned, up to 10 minutes of audio)
        max_seq_len = 6000  # match Kaggle training
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # 3. Mamba backbone
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 4. Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) — raw PCM float32
            return_embeddings: if True, return (batch, d_model) features instead of logits

        Returns:
            logits: (batch, n_classes)  — pass through softmax for probabilities
        """
        B, T = waveform.shape

        # Patch embed: (B, samples) → (B, d_model, n_patches) → (B, n_patches, d_model)
        x = waveform.unsqueeze(1)                    # (B, 1, T)
        x = self.patch_embed(x)                      # (B, d_model, n_patches)
        x = x.transpose(1, 2)                        # (B, n_patches, d_model)

        # Add positional embeddings
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embed(positions)

        # Mamba blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pool over time
        embeddings = x.mean(dim=1)                   # (B, d_model)

        if return_embeddings:
            return embeddings

        return self.head(embeddings)                 # (B, n_classes)

    def predict_proba(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convenience method — returns softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(waveform)
            return torch.softmax(logits, dim=-1)

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "MambaAudioClassifier":
        model = cls(**kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        return model

    def save_checkpoint(self, path: str, epoch: int = 0, loss: float = 0.0):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": self.state_dict(),
            "d_model": self.d_model,
        }, path)
        print(f"[MambaAudio] Checkpoint saved → {path}")


# ─── Training scaffold ────────────────────────────────────────────────────────

def train_one_epoch(
    model: MambaAudioClassifier,
    dataloader,           # torch DataLoader yielding (waveform, label)
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
) -> float:
    """One training epoch. Returns average loss."""
    model.train()
    # Get n_classes from model's output layer
    n_classes = model.head[-1].out_features
    # Force model to pay equal attention to all emotions
    weights = torch.ones(n_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    total_loss = 0.0

    for waveform, labels in dataloader:
        waveform = waveform.float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(waveform)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model = MambaAudioClassifier(n_classes=8, d_model=256, n_layers=4).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Simulate 2 seconds of audio at 16kHz
    dummy = torch.randn(2, 32000).to(device)
    probs = model.predict_proba(dummy)
    print(f"Input shape: {dummy.shape}")
    print(f"Output probs shape: {probs.shape}")
    print(f"Sum of probs (should be ~1): {probs.sum(dim=-1)}")
