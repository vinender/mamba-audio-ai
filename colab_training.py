"""
Google Colab training setup for Mamba Audio AI
Run this in a Colab cell to train on long.wav with GPU

Steps:
1. Create new Colab notebook
2. Run cells 1-6 in order
3. Mount Google Drive (optional, for saving checkpoints)
"""

# ============================================================================
# CELL 1: Install dependencies and setup
# ============================================================================
# !pip install torch torchaudio -q
# !pip install fastapi uvicorn websockets pydantic pydantic-settings -q
# !pip install librosa soundfile numpy scipy scikit-learn -q
# !pip install openai-whisper -q

# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ============================================================================
# CELL 2: Clone repo or upload files
# ============================================================================
# Option A: Clone from GitHub (if repo is public)
# !git clone https://github.com/yourusername/mamba-audio-ai.git
# %cd mamba-audio-ai

# Option B: Upload zip file manually, then:
# !unzip mamba-audio-ai.zip
# %cd mamba-audio-ai

# Option C: Just create minimal structure for this notebook (no git needed)
import os
os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./data/long_wav", exist_ok=True)

# ============================================================================
# CELL 3: Upload long.wav (if not already in Drive)
# ============================================================================
from google.colab import files
print("Upload long.wav file:")
uploaded = files.upload()

# Move to data directory
import shutil
for filename, content in uploaded.items():
    shutil.move(filename, f"./data/long_wav/{filename}")
    print(f"Moved {filename} to ./data/long_wav/")

# ============================================================================
# CELL 4: Prepare training data from long.wav
# ============================================================================
import soundfile as sf
import numpy as np
import torch
import torchaudio

audio_file = "./data/long_wav/long.wav"
output_dir = "./data/long_wav/chunks"
os.makedirs(output_dir, exist_ok=True)

# Load audio
audio, sr = sf.read(audio_file, dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)

print(f"Loaded: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.1f} seconds)")

# Resample to 16kHz
if sr != 16000:
    print(f"Resampling from {sr}Hz to 16000Hz...")
    t = torch.from_numpy(audio).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, 16000)
    audio = t.squeeze(0).numpy()

# Split into 2-second chunks for training (32000 samples)
chunk_size = 32000
chunk_duration = chunk_size / 16000  # 2 seconds
chunk_idx = 0

for start in range(0, len(audio) - chunk_size, chunk_size // 2):  # 50% overlap
    chunk = audio[start:start + chunk_size]

    # Normalize
    peak = np.abs(chunk).max()
    chunk = chunk / (peak + 1e-9)

    # Save
    output_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}.wav")
    sf.write(output_path, chunk, 16000)
    chunk_idx += 1

print(f"Created {chunk_idx} training chunks from long.wav")

# ============================================================================
# CELL 5: Copy config and models (minimal version for Colab)
# ============================================================================
# Create minimal config
config_py = '''
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    SAMPLE_RATE: int = 16000
    CHUNK_DURATION_SEC: float = 2.0
    CHUNK_SIZE: int = 32000
    MAX_BUFFER_SEC: float = 30.0

    MAMBA_D_MODEL: int = 256
    MAMBA_D_STATE: int = 64
    MAMBA_N_LAYERS: int = 6
    MAMBA_CHECKPOINT: str = ""

    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cuda"
    WHISPER_COMPUTE_TYPE: str = "float16"

    EMOTION_LABELS: list = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

settings = Settings()
'''

os.makedirs("./app", exist_ok=True)
with open("./app/config.py", "w") as f:
    f.write(config_py)

# Create __init__.py
open("./app/__init__.py", "w").close()

print("Config files created")

# ============================================================================
# CELL 6: Run training on GPU
# ============================================================================
import os
import glob
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import time

# Minimal MambaAudioClassifier (without mamba-ssm, use GRU fallback)
class MambaAudioClassifier(nn.Module):
    def __init__(self, n_classes=8, d_model=256, d_state=64, n_layers=6, sample_rate=16000):
        super().__init__()
        self.d_model = d_model
        self.patch_size = 160

        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=160, stride=160, padding=0, bias=False),
            nn.GELU(),
        )

        max_seq_len = (sample_rate * 600) // 160
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.GRU(d_model, d_model, batch_first=True),
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, waveform):
        B, T = waveform.shape

        x = waveform.unsqueeze(1)
        x = self.patch_embed(x)
        x = x.transpose(1, 2)

        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embed(positions)

        for block in self.blocks:
            norm_layer, gru = block
            x = norm_layer(x)
            x_gru, _ = gru(x)
            x = x + x_gru

        x = self.norm(x)
        embeddings = x.mean(dim=1)
        return self.head(embeddings)

    def predict_proba(self, waveform):
        logits = self.forward(waveform)
        return torch.softmax(logits, dim=-1)

# Dataset
class AudioDataset(Dataset):
    def __init__(self, file_paths, chunk_size=32000):
        self.files = file_paths
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sr = sf.read(path, dtype="float32")

        if sr != 16000:
            import torchaudio
            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, 16000)
            audio = t.squeeze(0).numpy()

        if len(audio) < self.chunk_size:
            audio = np.pad(audio, (0, self.chunk_size - len(audio)))
        else:
            audio = audio[:self.chunk_size]

        peak = np.abs(audio).max()
        audio = audio / (peak + 1e-9)

        # Dummy label (no labels from long.wav, so assign random for demo)
        label = np.random.randint(0, 8)

        return torch.from_numpy(audio).float(), label

# Training
print("\n" + "="*60)
print("STARTING TRAINING ON GPU")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create dataset
audio_files = glob.glob("./data/long_wav/chunks/*.wav")
print(f"Found {len(audio_files)} audio chunks")

dataset = AudioDataset(audio_files, chunk_size=32000)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

# Model
model = MambaAudioClassifier(n_classes=8, d_model=256, d_state=64, n_layers=6).float().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
criterion = nn.CrossEntropyLoss()

# Train
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    t_start = time.time()

    for batch_idx, (waveform, labels) in enumerate(train_loader):
        waveform = waveform.float().to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(waveform)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    elapsed = time.time() - t_start
    scheduler.step()

    print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

# Save checkpoint
os.makedirs("./checkpoints", exist_ok=True)
checkpoint_path = "./checkpoints/mamba_audio_long_wav.pt"
torch.save({
    "epoch": num_epochs,
    "loss": avg_loss,
    "model_state_dict": model.state_dict(),
    "d_model": 256,
}, checkpoint_path)
print(f"\nCheckpoint saved: {checkpoint_path}")

# ============================================================================
# CELL 7: Download checkpoint to local machine (optional)
# ============================================================================
# from google.colab import files
# files.download("./checkpoints/mamba_audio_long_wav.pt")
# print("Checkpoint downloaded to local machine!")

print("\n✅ Training complete on GPU!")
print("Checkpoint available at: ./checkpoints/mamba_audio_long_wav.pt")
