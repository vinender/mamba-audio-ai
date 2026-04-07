"""
Test client — simulates real-time microphone streaming.
Reads a WAV file and sends it to the WebSocket server chunk by chunk,
simulating real-time capture at the correct speed.

Usage:
    python client/test_client.py --file path/to/audio.wav
    python client/test_client.py --file path/to/audio.wav --realtime  # throttle to real-time speed
    python client/test_client.py --mic  # stream from microphone (requires pyaudio)
"""

import asyncio
import argparse
import json
import time
import sys
import numpy as np
import soundfile as sf
import websockets


SERVER_URL = "ws://localhost:8000/ws/stream"
CHUNK_SIZE_BYTES = 8192     # 4096 int16 samples = 256ms at 16kHz


def load_wav_as_pcm16(path: str, target_sr: int = 16000) -> bytes:
    """Load a WAV file and return raw int16 PCM bytes at 16kHz mono."""
    audio, sr = sf.read(path, dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)

    if sr != target_sr:
        import torchaudio
        import torch
        t = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, target_sr)
        audio = (t.squeeze(0).numpy() * 32768.0).astype(np.int16)

    return audio.tobytes()


async def stream_file(filepath: str, realtime: bool = False):
    print(f"Connecting to {SERVER_URL}...")

    async with websockets.connect(
        SERVER_URL,
        max_size=10 * 1024 * 1024,
        ping_interval=60,
        ping_timeout=120,
        close_timeout=30,
    ) as ws:
        # Wait for ready signal
        ready = json.loads(await ws.recv())
        print(f"\n✓ Connected | session: {ready['session_id']}")
        print(f"  Sample rate: {ready['sample_rate']}Hz")
        print(f"  Emotions: {', '.join(ready['emotion_labels'])}")
        print(f"\nStreaming: {filepath}\n{'─'*60}")

        # Load audio
        pcm_bytes = load_wav_as_pcm16(filepath)
        total_bytes = len(pcm_bytes)
        sent = 0
        chunk_num = 0

        # Send audio in chunks (simulating mic capture)
        receive_task = asyncio.create_task(receive_results(ws))

        while sent < total_bytes:
            chunk = pcm_bytes[sent:sent + CHUNK_SIZE_BYTES]
            await ws.send(chunk)
            sent += len(chunk)
            chunk_num += 1

            if realtime:
                # 8192 bytes / 2 bytes per int16 = 4096 samples at 16kHz = 256ms
                await asyncio.sleep(0.256)

        # Flush remaining buffer
        print("\n⏎  Sending flush...")
        await ws.send("flush")
        await asyncio.sleep(0.5)

        # End session
        await ws.send("end")

        # Wait for session summary
        try:
            await asyncio.wait_for(receive_task, timeout=10.0)
        except asyncio.TimeoutError:
            receive_task.cancel()

        print("\n✓ Done")


async def receive_results(ws):
    """Receive and pretty-print results from the server."""
    async for raw in ws:
        msg = json.loads(raw)
        event = msg.get("event", "")

        if event == "chunk":
            transcript = msg.get("transcript", "")
            emotion = msg.get("emotion", "")
            conf = msg.get("emotion_confidence", 0)
            ts = msg.get("timestamp", 0)
            proc_ms = msg.get("processing_time_ms", 0)

            # Emotion bar
            bar_len = int(conf * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            line = f"[{ts:6.1f}s] "
            if transcript:
                line += f'"{transcript}" | '
            line += f"{emotion} [{bar}] {conf:.0%} ({proc_ms:.0f}ms)"
            print(line)

        elif event == "flushed":
            result = msg.get("result")
            if result and result.get("transcript"):
                print(f"\n[flush] \"{result['transcript']}\"")

        elif event == "session_end":
            print(f"\n{'═'*60}")
            print(f"SESSION SUMMARY")
            print(f"{'─'*60}")
            print(f"Transcript: {msg.get('transcript', '(none)')}")
            print(f"Dominant emotion: {msg.get('dominant_emotion', '').upper()}")
            print(f"Total chunks: {msg.get('total_chunks', 0)}")
            emotions = msg.get("emotion_summary", {})
            if emotions:
                print("\nEmotion distribution:")
                sorted_em = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                for em, score in sorted_em:
                    bar = "█" * int(score * 30)
                    print(f"  {em:12s} {bar} {score:.1%}")
            print(f"{'═'*60}")
            return

        elif event == "error":
            print(f"\n[ERROR] {msg.get('message')}")


async def stream_microphone():
    """Stream from system microphone — requires: pip install pyaudio"""
    try:
        import pyaudio
    except ImportError:
        print("Install pyaudio: pip install pyaudio")
        sys.exit(1)

    RATE = 16000
    FRAMES_PER_BUFFER = 4096  # 256ms chunks

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )

    print(f"Connecting to {SERVER_URL}...")
    async with websockets.connect(SERVER_URL) as ws:
        ready = json.loads(await ws.recv())
        print(f"✓ Connected | session: {ready['session_id']}")
        print("🎤 Listening... (Ctrl+C to stop)\n")

        receive_task = asyncio.create_task(receive_results(ws))

        try:
            loop = asyncio.get_event_loop()
            while True:
                data = await loop.run_in_executor(
                    None, lambda: stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                )
                await ws.send(data)
        except KeyboardInterrupt:
            print("\nStopping...")
            await ws.send("end")
            await asyncio.wait_for(receive_task, timeout=5.0)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mamba Audio AI — test client")
    parser.add_argument("--file", type=str, help="Path to audio file (WAV/MP3/FLAC)")
    parser.add_argument("--mic", action="store_true", help="Stream from microphone")
    parser.add_argument("--realtime", action="store_true", help="Throttle to real-time speed")
    parser.add_argument("--server", type=str, default=SERVER_URL, help="WebSocket URL")
    args = parser.parse_args()

    SERVER_URL = args.server

    if args.mic:
        asyncio.run(stream_microphone())
    elif args.file:
        asyncio.run(stream_file(args.file, realtime=args.realtime))
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python client/test_client.py --file audio/sample.wav")
        print("  python client/test_client.py --mic")
        print("  python client/test_client.py --file audio.wav --realtime")
