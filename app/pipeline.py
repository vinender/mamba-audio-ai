"""
Audio processing pipeline.

For each incoming chunk:
  1. Normalize + resample
  2. Run Mamba → emotion probabilities (real-time, every chunk)
  3. Buffer audio → run Whisper → transcription (every N chunks or on silence)
  4. Return combined result as dict
"""

import asyncio
import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings
from app.models.mamba_audio import MambaAudioClassifier
from app.models.transcriber import Transcriber
from app.utils.audio import (
    pcm_bytes_to_tensor,
    normalize_waveform,
    resample,
    AudioBuffer,
)


@dataclass
class AudioResult:
    """Single result returned to the WebSocket client."""
    timestamp: float
    # Transcription
    transcript: str = ""
    transcript_confidence: float = 0.0
    language: str = ""
    # Emotion
    emotion: str = ""
    emotion_confidence: float = 0.0
    emotion_scores: dict = field(default_factory=dict)
    # Meta
    chunk_duration_sec: float = 0.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "transcript": self.transcript,
            "transcript_confidence": self.transcript_confidence,
            "language": self.language,
            "emotion": self.emotion,
            "emotion_confidence": self.emotion_confidence,
            "emotion_scores": self.emotion_scores,
            "chunk_duration_sec": self.chunk_duration_sec,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class AudioPipeline:
    """
    Stateful pipeline for one WebSocket connection.
    Each client gets their own pipeline instance (own buffer, own state).
    """

    # How many chunks to accumulate before running Whisper
    # 5 chunks × 1sec = transcribe every 5 seconds (good latency/accuracy tradeoff)
    TRANSCRIBE_EVERY_N_CHUNKS = 5

    def __init__(
        self,
        mamba_model: MambaAudioClassifier,
        transcriber: Transcriber,
        device: str = "cuda",
    ):
        self.mamba = mamba_model
        self.transcriber = transcriber
        self.device = device

        self.buffer = AudioBuffer(
            chunk_size=settings.CHUNK_SIZE,
            max_buffer_samples=int(settings.SAMPLE_RATE * settings.MAX_BUFFER_SEC),
        )

        # Accumulate audio for Whisper (needs more context than 1 sec)
        self._whisper_buffer: list[torch.Tensor] = []
        self._chunk_count = 0
        self._session_start = time.time()

    def reset(self):
        """Reset state for a new session."""
        self.buffer.reset()
        self._whisper_buffer = []
        self._chunk_count = 0
        self._session_start = time.time()

    async def process_bytes(self, raw_bytes: bytes) -> list[AudioResult]:
        """
        Main entry point — receives raw PCM bytes from WebSocket.
        Returns list of results (one per complete chunk ready).
        """
        # Decode bytes → tensor
        waveform = pcm_bytes_to_tensor(raw_bytes, sample_rate=settings.SAMPLE_RATE)
        waveform = normalize_waveform(waveform)

        # Push to buffer, get complete chunks
        chunks = self.buffer.push(waveform)
        results = []

        for chunk in chunks:
            result = await self._process_chunk(chunk)
            results.append(result)

        return results

    async def _process_chunk(self, chunk: torch.Tensor) -> AudioResult:
        """Process a single fixed-size chunk through both models."""
        t_start = time.perf_counter()
        self._chunk_count += 1

        chunk = chunk.to(self.device)
        timestamp = time.time() - self._session_start

        # ── Mamba: emotion classification (runs on every chunk, fast) ─────────
        emotion, emotion_conf, emotion_scores = await asyncio.get_event_loop().run_in_executor(
            None, self._run_emotion, chunk
        )

        # ── Whisper: transcription (runs every N chunks) ──────────────────────
        self._whisper_buffer.append(chunk.cpu())
        transcript = ""
        transcript_conf = 0.0
        language = ""

        if self._chunk_count % self.TRANSCRIBE_EVERY_N_CHUNKS == 0:
            transcript, transcript_conf, language = await asyncio.get_event_loop().run_in_executor(
                None, self._run_transcription
            )
            self._whisper_buffer = []  # clear after transcribing

        t_end = time.perf_counter()
        processing_ms = (t_end - t_start) * 1000

        return AudioResult(
            timestamp=round(timestamp, 3),
            transcript=transcript,
            transcript_confidence=transcript_conf,
            language=language,
            emotion=emotion,
            emotion_confidence=emotion_conf,
            emotion_scores=emotion_scores,
            chunk_duration_sec=settings.CHUNK_DURATION_SEC,
            processing_time_ms=processing_ms,
        )

    def _run_emotion(self, chunk: torch.Tensor):
        """Run Mamba emotion model — synchronous, called via executor."""
        self.mamba.eval()
        with torch.no_grad():
            batch = chunk.unsqueeze(0)    # (1, samples)
            probs = self.mamba.predict_proba(batch)   # (1, n_classes)
            probs_np = probs[0].cpu().numpy()

        top_idx = int(probs_np.argmax())
        emotion = settings.EMOTION_LABELS[top_idx]
        confidence = float(probs_np[top_idx])
        scores = {
            label: round(float(p), 4)
            for label, p in zip(settings.EMOTION_LABELS, probs_np)
        }
        return emotion, round(confidence, 4), scores

    def _run_transcription(self):
        """Run Whisper on buffered audio — synchronous, called via executor."""
        if not self._whisper_buffer:
            return "", 0.0, ""

        combined = torch.cat(self._whisper_buffer, dim=-1).cpu().numpy()
        segments = self.transcriber.transcribe(combined, sample_rate=settings.SAMPLE_RATE)

        if not segments:
            return "", 0.0, ""

        full_text = " ".join(s.text for s in segments if s.text)
        avg_conf = sum(s.confidence for s in segments) / len(segments)
        language = segments[0].language if segments else ""

        return full_text, round(avg_conf, 4), language

    async def flush(self) -> Optional[AudioResult]:
        """Process any remaining audio in the buffer at session end."""
        remainder = self.buffer.flush()
        if remainder is None:
            return None
        return await self._process_chunk(remainder)


class PipelineManager:
    """
    Singleton that manages shared model instances.
    Models are loaded once at startup and shared across all WebSocket connections.
    Each connection gets its own AudioPipeline (stateful buffer).
    """

    def __init__(self):
        self._mamba: Optional[MambaAudioClassifier] = None
        self._transcriber: Optional[Transcriber] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def startup(self):
        """Load models at app startup (called from FastAPI lifespan)."""
        print(f"[Pipeline] Loading models on {self._device}...")

        import os
        if settings.MAMBA_CHECKPOINT and os.path.exists(settings.MAMBA_CHECKPOINT):
            state = torch.load(
                settings.MAMBA_CHECKPOINT,
                map_location=self._device,
                weights_only=True
            )
            model_type = state.get("model_type", "mamba")

            if model_type == "mel_gru":
                from app.models.mel_gru import AudioClassifier
                n_classes = state.get("n_classes", 5)
                self._mamba = AudioClassifier(
                    n_classes=n_classes,
                    d_model=256,
                    n_layers=6,
                ).float().to(self._device)
                if "emotion_labels" in state:
                    settings.EMOTION_LABELS = state["emotion_labels"]
                print(f"[Pipeline] Loading AudioClassifier (mel+GRU)")
            else:
                self._mamba = MambaAudioClassifier(
                    n_classes=len(settings.EMOTION_LABELS),
                    d_model=settings.MAMBA_D_MODEL,
                    d_state=settings.MAMBA_D_STATE,
                    n_layers=settings.MAMBA_N_LAYERS,
                ).float().to(self._device)
                print(f"[Pipeline] Loading MambaAudioClassifier")

            self._mamba.load_state_dict(state["model_state_dict"])
            print(f"[Pipeline] Loaded checkpoint: {settings.MAMBA_CHECKPOINT}")
        else:
            self._mamba = MambaAudioClassifier(
                n_classes=len(settings.EMOTION_LABELS),
                d_model=settings.MAMBA_D_MODEL,
                d_state=settings.MAMBA_D_STATE,
                n_layers=settings.MAMBA_N_LAYERS,
            ).float().to(self._device)
            print("[Pipeline] No checkpoint — random weights (dev mode)")

        self._transcriber = Transcriber(
            model_size=settings.WHISPER_MODEL_SIZE,
            device=self._device,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )

        print("[Pipeline] All models ready.")

    def new_pipeline(self) -> AudioPipeline:
        """Create a new pipeline for a WebSocket client."""
        assert self._mamba is not None and self._transcriber is not None, \
            "Call startup() first"
        return AudioPipeline(
            mamba_model=self._mamba,
            transcriber=self._transcriber,
            device=self._device,
        )

    def shutdown(self):
        del self._mamba
        del self._transcriber
        torch.cuda.empty_cache()


# Global singleton
pipeline_manager = PipelineManager()
