import torch
import torch.nn as nn
import torchaudio.transforms as T

SAMPLE_RATE = 16000


class GRUBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gru  = nn.GRU(d_model, d_model, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        r = x
        x = self.norm(x)
        x, _ = self.gru(x)
        return self.proj(x) + r


class AudioClassifier(nn.Module):
    def __init__(self, n_classes=5, d_model=256, n_layers=6):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            hop_length=160,
            n_mels=80,
            f_min=0,
            f_max=8000,
        )
        self.mel_norm   = nn.InstanceNorm1d(80)
        self.input_proj = nn.Sequential(
            nn.Linear(80, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.pos_embed = nn.Embedding(6000, d_model)
        self.blocks    = nn.ModuleList([
            GRUBlock(d_model) for _ in range(n_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)
        self.attn_pool = nn.Linear(d_model, 1)
        self.head      = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, waveform):
        mel  = self.mel(waveform)
        mel  = torch.log(mel + 1e-9)
        mel  = self.mel_norm(mel)
        x    = mel.transpose(1, 2)
        x    = self.input_proj(x)
        pos  = torch.arange(x.shape[1], device=x.device)
        x    = x + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x    = self.norm(x)
        attn = torch.softmax(self.attn_pool(x), dim=1)
        x    = (x * attn).sum(dim=1)
        return self.head(x)

    def predict_proba(self, waveform):
        with torch.no_grad():
            return torch.softmax(self.forward(waveform), dim=-1)