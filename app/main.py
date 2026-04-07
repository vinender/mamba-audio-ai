"""
Mamba Audio AI — FastAPI Server
================================
Endpoints:
  WS  /ws/stream         — Real-time audio stream (WebSocket)
  POST /api/transcribe   — Upload an audio file, get full transcript + emotions
  GET  /api/health       — Health check
  GET  /api/session/{id} — Get session summary
"""

import json
import time
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
import numpy as np
import soundfile as sf
import io

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.pipeline import pipeline_manager, AudioResult
from app.utils.audio import pcm_bytes_to_tensor, normalize_waveform

logger = logging.getLogger("mamba-audio")
logging.basicConfig(level=logging.INFO)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    pipeline_manager.startup()
    yield
    # Shutdown
    pipeline_manager.shutdown()


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mamba Audio AI",
    description="Real-time speech transcription + emotion detection powered by Mamba SSM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Session store (in-memory, swap with Redis in production) ─────────────────

sessions: dict[str, dict] = {}


# ─── Response models ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    device: str
    whisper_model: str
    mamba_layers: int


class FileTranscribeResponse(BaseModel):
    session_id: str
    duration_sec: float
    transcript: str
    language: str
    emotions: dict
    segments: list[dict]
    processing_time_ms: float


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device="cuda" if torch.cuda.is_available() else "cpu",
        whisper_model=settings.WHISPER_MODEL_SIZE,
        mamba_layers=settings.MAMBA_N_LAYERS,
    )


# ─── Session summary ──────────────────────────────────────────────────────────

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


# ─── File upload transcription ────────────────────────────────────────────────

@app.post("/api/transcribe", response_model=FileTranscribeResponse)
async def transcribe_file(file: UploadFile = File(...)):
    """
    Upload an audio file (WAV, MP3, FLAC, OGG) and get:
    - Full transcript with timestamps
    - Per-segment emotion predictions
    - Overall emotion distribution
    """
    t_start = time.perf_counter()

    # Read + decode audio
    content = await file.read()
    try:
        audio_np, sample_rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)  # stereo → mono
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio file: {e}")

    pipeline = pipeline_manager.new_pipeline()
    duration_sec = len(audio_np) / sample_rate
    session_id = str(uuid.uuid4())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Whisper: transcribe the full audio at once (much better accuracy) ─────
    transcriber = pipeline.transcriber
    segments = await asyncio.get_event_loop().run_in_executor(
        None, transcriber.transcribe, audio_np, sample_rate
    )
    full_transcript = " ".join(s.text for s in segments if s.text)
    detected_lang = segments[0].language if segments else "unknown"

    # ── Mamba: run emotion on fixed-size chunks ──────────────────────────────
    chunk_size = settings.CHUNK_SIZE
    all_results: list[AudioResult] = []

    for start in range(0, len(audio_np), chunk_size):
        chunk_np = audio_np[start:start + chunk_size]
        if len(chunk_np) < chunk_size:
            chunk_np = np.pad(chunk_np, (0, chunk_size - len(chunk_np)))
        chunk_t = torch.from_numpy(chunk_np).to(device)

        emotion, emotion_conf, emotion_scores = pipeline._run_emotion(chunk_t)
        all_results.append(AudioResult(
            timestamp=round(start / sample_rate, 3),
            emotion=emotion,
            emotion_confidence=emotion_conf,
            emotion_scores=emotion_scores,
            chunk_duration_sec=settings.CHUNK_DURATION_SEC,
        ))

    # Aggregate emotion scores
    emotion_totals: dict[str, float] = {e: 0.0 for e in settings.EMOTION_LABELS}
    for r in all_results:
        for emotion, score in r.emotion_scores.items():
            emotion_totals[emotion] = emotion_totals.get(emotion, 0.0) + score
    n = len(all_results) or 1
    emotion_avg = {k: round(v / n, 4) for k, v in emotion_totals.items()}
    dominant_emotion = max(emotion_avg, key=emotion_avg.get)

    t_end = time.perf_counter()
    processing_ms = (t_end - t_start) * 1000

    response = FileTranscribeResponse(
        session_id=session_id,
        duration_sec=round(duration_sec, 2),
        transcript=full_transcript,
        language=detected_lang,
        emotions={
            "dominant": dominant_emotion,
            "scores": emotion_avg,
        },
        segments=[r.to_dict() for r in all_results],
        processing_time_ms=round(processing_ms, 2),
    )

    # Store session
    sessions[session_id] = response.model_dump()
    return response


# ─── WebSocket — real-time streaming ─────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time audio streaming endpoint.

    Protocol:
      Client sends:
        - Binary frames: raw PCM audio bytes (16-bit signed, 16kHz, mono)
        - Text frame "flush": request to process remaining buffer
        - Text frame "reset": reset session state
        - Text frame "end": close session

      Server sends:
        - JSON objects on each processed chunk (see AudioResult.to_dict())
        - JSON {"event": "ready"} on connection
        - JSON {"event": "flushed", "result": ...} on flush
        - JSON {"event": "error", "message": "..."} on errors

    Example client message rate: 16000 samples/sec ÷ 4096 bytes/frame = ~7.8 frames/sec
    """
    session_id = str(uuid.uuid4())
    await websocket.accept()

    pipeline = pipeline_manager.new_pipeline()
    logger.info(f"[WS] New session: {session_id}")

    # Send ready signal
    await websocket.send_json({
        "event": "ready",
        "session_id": session_id,
        "chunk_duration_sec": settings.CHUNK_DURATION_SEC,
        "sample_rate": settings.SAMPLE_RATE,
        "emotion_labels": settings.EMOTION_LABELS,
    })

    session_results: list[dict] = []

    try:
        while True:
            message = await websocket.receive()

            # ── Binary audio data ────────────────────────────────────────────
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                try:
                    results = await pipeline.process_bytes(raw_bytes)
                    for result in results:
                        payload = result.to_dict()
                        payload["event"] = "chunk"
                        session_results.append(payload)
                        await websocket.send_json(payload)
                except Exception as e:
                    logger.error(f"[WS] Processing error: {e}")
                    await websocket.send_json({"event": "error", "message": str(e)})

            # ── Text control messages ────────────────────────────────────────
            elif "text" in message and message["text"]:
                cmd = message["text"].strip().lower()

                if cmd == "flush":
                    result = await pipeline.flush()
                    if result:
                        payload = result.to_dict()
                        payload["event"] = "flushed"
                        await websocket.send_json(payload)
                    else:
                        await websocket.send_json({"event": "flushed", "result": None})

                elif cmd == "reset":
                    pipeline.reset()
                    session_results = []
                    await websocket.send_json({"event": "reset"})

                elif cmd == "end":
                    # Final flush + send session summary
                    result = await pipeline.flush()
                    if result:
                        session_results.append(result.to_dict())

                    # Aggregate session-level emotions
                    emotion_totals: dict[str, float] = {}
                    for r in session_results:
                        for e, s in r.get("emotion_scores", {}).items():
                            emotion_totals[e] = emotion_totals.get(e, 0.0) + s
                    n = len(session_results) or 1
                    emotion_avg = {k: round(v / n, 4) for k, v in emotion_totals.items()}

                    full_transcript = " ".join(
                        r.get("transcript", "") for r in session_results if r.get("transcript")
                    )

                    await websocket.send_json({
                        "event": "session_end",
                        "session_id": session_id,
                        "total_chunks": len(session_results),
                        "transcript": full_transcript,
                        "emotion_summary": emotion_avg,
                        "dominant_emotion": max(emotion_avg, key=emotion_avg.get) if emotion_avg else "neutral",
                    })
                    break

    except WebSocketDisconnect:
        logger.info(f"[WS] Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"[WS] Unexpected error in session {session_id}: {e}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        logger.info(f"[WS] Session {session_id} closed. Chunks processed: {len(session_results)}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL,
        reload=False,
    )
