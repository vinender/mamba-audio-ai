import torch
import numpy as np
import whisper
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    confidence: float
    language: str


class Transcriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",   # ignored, kept for API compatibility
        language: Optional[str] = None,
        vad_filter: bool = True,
    ):
        self.language = language
        print(f"[Transcriber] Loading Whisper {model_size} on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        print("[Transcriber] Ready.")

    def transcribe(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[TranscriptSegment]:
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        result = self.model.transcribe(
            waveform,
            language=self.language,
            fp16=False,   # Mac CPU needs fp16=False
        )

        detected_lang = result.get("language", "en")
        segments = result.get("segments", [])

        return [
            TranscriptSegment(
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                confidence=round(float(seg.get("avg_logprob", 0)), 3),
                language=detected_lang,
            )
            for seg in segments
            if seg["text"].strip()
        ]

    def transcribe_tensor(self, waveform: torch.Tensor) -> list[TranscriptSegment]:
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)
        return self.transcribe(waveform.cpu().numpy())