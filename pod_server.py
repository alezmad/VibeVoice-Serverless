"""FastAPI + WebSocket server for VibeVoice streaming TTS.

Runs as a long-lived RunPod Pod. Clients open a WebSocket, stream text tokens
as they arrive from an upstream LLM, and the server buffers until sentence
boundaries, synthesizes, and streams WAV bytes back in order.

Messages (client → server, JSON):
    {"type": "text", "data": "partial text chunk"}
    {"type": "flush"}               # force synth of current buffer
    {"type": "end"}                 # no more text, finalize

Messages (server → client):
    {"type": "ready"}
    {"type": "chunk", "seq": int, "audio_b64": str, "sample_rate": int, "text": str}
    {"type": "done"}
    {"type": "error", "message": str}
"""
import asyncio
import base64
import io
import logging
import os
import re
import sys
import wave

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, "/workspace/vibevoice")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pod_server")

import config  # noqa: E402
from inference import VibeVoiceInference  # noqa: E402

app = FastAPI()

_INFERENCE: VibeVoiceInference | None = None
_MODEL_READY = asyncio.Event()

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
MIN_CHUNK_CHARS = 40


@app.on_event("startup")
async def startup_event():
    """Load the model once on startup so WS connections are instant."""
    global _INFERENCE
    log.info("Loading VibeVoice model...")
    _INFERENCE = VibeVoiceInference()
    _INFERENCE.load_model()
    _MODEL_READY.set()
    log.info("Model loaded — ready.")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok" if _MODEL_READY.is_set() else "loading"})


def split_on_sentences(buffer: str) -> tuple[list[str], str]:
    """Split buffer into (complete_sentences, remainder)."""
    parts = SENTENCE_BOUNDARY.split(buffer)
    if len(parts) == 1:
        return [], buffer
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    complete = [c for c in complete if len(c) >= MIN_CHUNK_CHARS or c.endswith((".", "!", "?"))]
    return complete, remainder


def wav_bytes(audio, sample_rate: int = 24000) -> bytes:
    if hasattr(audio, "detach"):
        audio = audio.detach().to("cpu").float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def synthesize_chunk(text: str, speaker: str | None = None) -> tuple[np.ndarray, int]:
    """Run blocking inference in a thread so the event loop stays responsive."""
    loop = asyncio.get_event_loop()
    def _run():
        audio = _INFERENCE.generate(text, speaker_name=speaker)
        return audio, 24000
    return await loop.run_in_executor(None, _run)


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    await _MODEL_READY.wait()
    await ws.send_json({"type": "ready"})

    buffer = ""
    seq = 0
    speaker = None

    try:
        while True:
            msg = await ws.receive_json()
            mtype = msg.get("type")

            if mtype == "text":
                buffer += msg.get("data", "")
                if msg.get("speaker"):
                    speaker = msg["speaker"]
                complete, buffer = split_on_sentences(buffer)
                for sentence in complete:
                    seq += 1
                    try:
                        audio, sr = await synthesize_chunk(sentence, speaker)
                        await ws.send_json({
                            "type": "chunk",
                            "seq": seq,
                            "audio_b64": base64.b64encode(wav_bytes(audio, sr)).decode(),
                            "sample_rate": sr,
                            "text": sentence,
                        })
                    except Exception as e:
                        log.exception("synthesis failed")
                        await ws.send_json({"type": "error", "message": str(e)})

            elif mtype == "flush":
                if buffer.strip():
                    seq += 1
                    audio, sr = await synthesize_chunk(buffer.strip(), speaker)
                    await ws.send_json({
                        "type": "chunk",
                        "seq": seq,
                        "audio_b64": base64.b64encode(wav_bytes(audio, sr)).decode(),
                        "sample_rate": sr,
                        "text": buffer.strip(),
                    })
                    buffer = ""

            elif mtype == "end":
                if buffer.strip():
                    seq += 1
                    audio, sr = await synthesize_chunk(buffer.strip(), speaker)
                    await ws.send_json({
                        "type": "chunk",
                        "seq": seq,
                        "audio_b64": base64.b64encode(wav_bytes(audio, sr)).decode(),
                        "sample_rate": sr,
                        "text": buffer.strip(),
                    })
                    buffer = ""
                await ws.send_json({"type": "done"})
                break

    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as e:
        log.exception("ws error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
