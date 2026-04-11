#!/usr/bin/env python3
"""Create voice file (using config paths) and start handler."""
import os
import sys
import wave

sys.path.insert(0, '/workspace/vibevoice')
import config

voices_dir = config.AUDIO_PROMPTS_DIR
os.makedirs(voices_dir, exist_ok=True)

voice_path = os.path.join(voices_dir, "Alice.wav")
if not os.path.exists(voice_path):
    with wave.open(voice_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(bytes(144000))
    print(f"Created default voice: {voice_path}")
else:
    print(f"Voice already exists: {voice_path}")

os.execvp("python3", ["python3", "/workspace/vibevoice/handler.py"])
