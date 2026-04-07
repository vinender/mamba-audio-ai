from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Audio
    SAMPLE_RATE: int = 16000          # Hz — Whisper and most audio models expect 16kHz
    CHUNK_DURATION_SEC: float = 1.0   # Process audio in 1-second windows
    CHUNK_SIZE: int = 16000           # SAMPLE_RATE * CHUNK_DURATION_SEC
    MAX_BUFFER_SEC: float = 30.0      # Max audio buffer before forced flush

    # Mamba model
    MAMBA_D_MODEL: int = 256          # Embedding dim
    MAMBA_D_STATE: int = 64           # SSM state size
    MAMBA_N_LAYERS: int = 6           # Number of Mamba blocks
    MAMBA_CHECKPOINT: str = "./checkpoints/mamba_audio_best.pt"        # Path to fine-tuned weights

    # Whisper
    WHISPER_MODEL_SIZE: str = "base"  # tiny | base | small | medium | large-v3
    WHISPER_DEVICE: str = "cuda"      # cuda | cpu
    WHISPER_COMPUTE_TYPE: str = "int8"  # float16 (GPU) | int8 (CPU)

    # Emotion labels (RAVDESS-style)
    EMOTION_LABELS: list[str] = [
        "neutral", "happy", "sad", "angry", "fearful"
    ]

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    class Config:
        env_file = ".env"


settings = Settings()
