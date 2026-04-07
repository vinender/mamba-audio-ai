"""
Audio utility functions.
Handles raw PCM bytes → normalized float32 tensors → mel spectrograms.
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional


def pcm_bytes_to_tensor(
    raw_bytes: bytes,
    sample_rate: int = 16000,
    bit_depth: int = 16,
) -> torch.Tensor:
    """
    Convert raw PCM bytes (from WebSocket) to a float32 tensor.
    Assumes mono audio, 16-bit signed PCM by default.

    Returns: shape (samples,)
    """
    dtype = np.int16 if bit_depth == 16 else np.float32
    audio_np = np.frombuffer(raw_bytes, dtype=dtype)

    if bit_depth == 16:
        audio_np = audio_np.astype(np.float32) / 32768.0  # normalize to [-1, 1]

    return torch.from_numpy(audio_np)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform to target sample rate."""
    if orig_sr == target_sr:
        return waveform
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Peak-normalize to [-1, 1] range."""
    peak = waveform.abs().max()
    return waveform / (peak + eps)


def waveform_to_mel(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    f_min: float = 0.0,
    f_max: Optional[float] = 8000.0,
) -> torch.Tensor:
    """
    Convert raw waveform to log-mel spectrogram.

    Input:  waveform (samples,) or (1, samples)
    Output: mel (n_mels, time_frames)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
    )
    mel = mel_transform(waveform)       # (1, n_mels, time)
    log_mel = torch.log(mel + 1e-9)    # log scale
    return log_mel.squeeze(0)           # (n_mels, time)


def chunk_waveform(
    waveform: torch.Tensor,
    chunk_size: int,
    overlap: int = 0,
) -> list[torch.Tensor]:
    """
    Split waveform into fixed-size chunks with optional overlap.
    Pads the last chunk with zeros if needed.

    Returns: list of tensors, each shape (chunk_size,)
    """
    chunks = []
    step = chunk_size - overlap
    total = waveform.shape[-1]

    for start in range(0, total, step):
        end = start + chunk_size
        chunk = waveform[start:end]
        if chunk.shape[-1] < chunk_size:
            # Pad last chunk
            pad_size = chunk_size - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
        chunks.append(chunk)

    return chunks


class AudioBuffer:
    """
    Rolling buffer for streaming audio.
    Accumulates incoming bytes, yields complete chunks when ready.
    """

    def __init__(self, chunk_size: int, max_buffer_samples: int):
        self.chunk_size = chunk_size
        self.max_buffer_samples = max_buffer_samples
        self._buffer: list[torch.Tensor] = []
        self._total_samples = 0

    def push(self, audio: torch.Tensor) -> list[torch.Tensor]:
        """
        Push new audio samples into buffer.
        Returns list of complete chunks ready for processing.
        """
        self._buffer.append(audio)
        self._total_samples += audio.shape[-1]

        # Truncate if buffer exceeds max
        if self._total_samples > self.max_buffer_samples:
            self._buffer = self._buffer[-10:]  # keep recent
            self._total_samples = sum(t.shape[-1] for t in self._buffer)

        ready_chunks = []
        while self._total_samples >= self.chunk_size:
            combined = torch.cat(self._buffer, dim=-1)
            chunk = combined[:self.chunk_size]
            remainder = combined[self.chunk_size:]

            ready_chunks.append(chunk)
            self._buffer = [remainder] if remainder.shape[-1] > 0 else []
            self._total_samples = remainder.shape[-1]

        return ready_chunks

    def flush(self) -> Optional[torch.Tensor]:
        """Return whatever is in the buffer (padded), then clear."""
        if not self._buffer or self._total_samples == 0:
            return None
        combined = torch.cat(self._buffer, dim=-1)
        pad_size = self.chunk_size - (combined.shape[-1] % self.chunk_size)
        if pad_size < self.chunk_size:
            combined = torch.nn.functional.pad(combined, (0, pad_size))
        self._buffer = []
        self._total_samples = 0
        return combined

    def reset(self):
        self._buffer = []
        self._total_samples = 0
