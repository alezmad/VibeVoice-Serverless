"""Minimal RunPod serverless handler for VibeVoice TTS."""
import runpod
import torch
import numpy as np
import base64
import io
import wave
import logging
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vibevoice-handler")

# Global model
pipe = None

def load_model():
    global pipe
    if pipe is not None:
        return pipe
    
    log.info("Loading VibeVoice model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from vibevoice.pipeline import VibeVoicePipeline
        pipe = VibeVoicePipeline.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
        pipe = pipe.to(device)
        log.info(f"Model loaded on {device}")
    except ImportError:
        log.info("Pipeline not available, trying direct model load...")
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        pipe = "direct"
    
    return pipe

def handler(job):
    """RunPod handler - generates TTS audio."""
    inp = job.get("input", {})
    text = inp.get("text", "")
    if not text:
        return {"error": "Missing 'text' parameter"}
    
    log.info(f"Generating TTS for: {text[:50]}...")
    
    model = load_model()
    
    try:
        output = model(text)
        
        if hasattr(output, 'cpu'):
            audio = output.cpu().numpy()
        elif hasattr(output, 'audio'):
            audio = output.audio.cpu().numpy() if hasattr(output.audio, 'cpu') else np.array(output.audio)
        else:
            audio = np.array(output)
        
        if audio.ndim > 1:
            audio = audio.squeeze()
        audio = audio.astype(np.float32)
        
        # Encode as WAV
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm16.tobytes())
        
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return {
            "status": "success",
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000,
            "duration_sec": round(len(audio) / 24000, 2),
        }
    except Exception as e:
        log.error(f"TTS failed: {e}")
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
