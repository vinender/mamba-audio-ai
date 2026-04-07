# Mamba Audio AI

Real-time audio transcription + emotion detection powered by **Mamba SSM** (linear-time sequence model) + **Whisper**.

## Why Mamba for audio?

| Model | 30-sec audio | Memory | Speed |
|-------|-------------|--------|-------|
| Transformer | OOM at 480K tokens | O(n²) | Slow |
| Mamba | Handles 480K+ easily | **O(n) / O(1) per step** | **5× faster** |

Mamba runs as an RNN at inference — constant memory regardless of audio length.

## Architecture

```
Microphone / File
      │
      ▼ raw PCM bytes (WebSocket)
┌─────────────┐
│ AudioBuffer  │  accumulates chunks, yields fixed-size windows
└─────────────┘
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
┌──────────────────┐          ┌───────────────────┐
│ MambaAudioModel  │          │    Whisper ASR     │
│ (every 1s chunk) │          │ (every 5s buffer)  │
│                  │          │                    │
│ raw waveform     │          │ numpy float32      │
│ → patch embed    │          │ → transcript       │
│ → 6× Mamba2      │          │ → language detect  │
│ → global pool    │          └───────────────────┘
│ → 8 emotions     │
└──────────────────┘
      │                                  │
      └──────────────┬───────────────────┘
                     ▼
            AudioResult (JSON)
            ├── transcript
            ├── emotion + confidence
            ├── emotion_scores {}
            └── timestamp, processing_ms
```

## Setup

### Requirements
- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (for Mamba SSM kernel)
- 8GB+ VRAM recommended

### 1. Install base deps
```bash
pip install -r requirements.txt
```

### 2. Install Mamba SSM (requires CUDA + nvcc)
```bash
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

### 3. Configure (optional)
Copy `.env.example` to `.env` and edit:
```bash
WHISPER_MODEL_SIZE=base      # tiny|base|small|medium|large-v3
WHISPER_DEVICE=cuda
MAMBA_N_LAYERS=6
MAMBA_D_MODEL=256
MAMBA_CHECKPOINT=            # leave blank for dev (random weights)
```

### 4. Run the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Test with a file
```bash
python client/test_client.py --file path/to/audio.wav
```

### 6. Test with microphone
```bash
pip install pyaudio
python client/test_client.py --mic
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/ws/stream` | Real-time streaming |
| `POST` | `/api/transcribe` | Upload file, get full analysis |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/session/{id}` | Get session results |

## WebSocket protocol

**Client → Server:**
- Binary frame: raw PCM bytes (int16, 16kHz, mono)
- `"flush"` — process remaining buffer
- `"reset"` — reset session state
- `"end"` — close and get summary

**Server → Client:**
```json
{
  "event": "chunk",
  "timestamp": 1.0,
  "transcript": "hello world",
  "emotion": "happy",
  "emotion_confidence": 0.87,
  "emotion_scores": { "happy": 0.87, "neutral": 0.08, ... },
  "processing_time_ms": 23.4
}
```

## Train on RAVDESS

```bash
# Download dataset from https://zenodo.org/record/1188976
# Extract to ./data/ravdess/

python train.py \
  --data_dir ./data/ravdess \
  --epochs 30 \
  --batch_size 16 \
  --output ./checkpoints

# Update MAMBA_CHECKPOINT in .env with the best checkpoint path
```

## Project structure

```
mamba-audio-ai/
├── app/
│   ├── main.py              # FastAPI app + WebSocket
│   ├── config.py            # Settings (pydantic)
│   ├── pipeline.py          # Audio pipeline orchestrator
│   ├── models/
│   │   ├── mamba_audio.py   # Mamba SSM emotion classifier
│   │   └── transcriber.py   # faster-whisper wrapper
│   └── utils/
│       └── audio.py         # PCM decoding, chunking, buffer
├── client/
│   └── test_client.py       # WebSocket test client (file + mic)
├── train.py                 # RAVDESS fine-tuning script
├── requirements.txt
└── README.md
```

## Production notes

- **Mamba weights**: Use random weights during dev (emotion predictions are random until trained).
  Set `MAMBA_CHECKPOINT` after training on RAVDESS.
- **Whisper model**: `base` is fast and accurate for English. Use `large-v3` for multilingual.
- **Scaling**: Models are loaded once at startup and shared across WebSocket sessions.
  Each session gets its own `AudioPipeline` (stateful buffer). Safe for multiple concurrent users.
- **Buffer**: Replace the in-memory session store with Redis for multi-process deployments.
