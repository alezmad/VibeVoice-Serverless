"""FastAPI + WebSocket server for VibeVoice streaming TTS.

Runs as a long-lived RunPod Pod with raw TCP port exposure (no HTTPS proxy).

Protocol:
    Client → server (JSON text frames):
        {"type": "text", "data": "partial text"}
        {"type": "text", "data": "...", "speaker": "Alice"}
        {"type": "flush"}     # force synth of current buffer
        {"type": "end"}       # no more text, finalize

    Server → client:
        Text frames (JSON):
            {"type": "ready"}
            {"type": "chunk_header", "seq": N, "text": "...", "sample_rate": 24000}
            {"type": "done"}
            {"type": "error", "message": "..."}

        Binary frames (raw audio):
            One binary frame per chunk_header, containing raw int16 PCM samples
            (little-endian, mono). The chunk_header is always sent FIRST so the
            client knows which seq/text the next binary frame belongs to.
"""
import asyncio
import io
import json
import logging
import os
import random
import re
import sys

# MUST be set before first CUDA op for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Pin cuDNN/cuBLAS determinism. warn_only so ops without a deterministic
# kernel fall back instead of raising.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

sys.path.insert(0, "/workspace/vibevoice")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pod_server")

import config  # noqa: E402
from inference import VibeVoiceInference  # noqa: E402

app = FastAPI()

_INFERENCE: VibeVoiceInference | None = None
_MODEL_READY = asyncio.Event()
_GPU_LOCK = asyncio.Lock()

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
MIN_CHUNK_CHARS = 40


@app.on_event("startup")
async def startup_event():
    global _INFERENCE
    log.info("Loading VibeVoice model...")
    _INFERENCE = VibeVoiceInference()
    _INFERENCE.load_model()
    try:
        log.info("Warming up kernels with a dummy inference...")
        _INFERENCE.generate("Warming up.")
        log.info("Warmup done.")
    except Exception as e:
        log.warning(f"Warmup failed (non-fatal): {e}")
    _MODEL_READY.set()
    log.info("Model loaded — ready.")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok" if _MODEL_READY.is_set() else "loading"})


def split_on_sentences(buffer: str) -> tuple[list[str], str]:
    parts = SENTENCE_BOUNDARY.split(buffer)
    if len(parts) == 1:
        return [], buffer
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    remainder = parts[-1]
    complete = [c for c in complete if len(c) >= MIN_CHUNK_CHARS or c.endswith((".", "!", "?"))]
    return complete, remainder


def to_pcm16_bytes(audio) -> bytes:
    """Convert model output tensor/array to little-endian int16 PCM bytes."""
    if hasattr(audio, "detach"):
        audio = audio.detach().to("cpu").float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")
    return pcm.tobytes()


class _ForceSeed:
    """Force a fixed seed across ALL sources of randomness inference.py touches.

    Layered defence:
      1. Patch ``os.urandom`` so the internal ``session_seed`` derivation picks
         the same bytes every call.
      2. Patch the inference instance's ``_set_seed`` so it ignores the caller's
         argument and always seeds Python/numpy/torch/cuda with ``session_seed``.
      3. Immediately re-apply the deterministic seed so any intervening random
         calls land on a known state.
    """

    def __init__(self, inference, session_seed: int):
        self._inference = inference
        self._seed = session_seed
        self._rng = random.Random(session_seed)
        self._orig_urandom = None
        self._orig_set_seed = None

    def __enter__(self):
        self._orig_urandom = os.urandom
        rng = self._rng
        def _urandom(n):
            return bytes(rng.randrange(256) for _ in range(n))
        os.urandom = _urandom

        self._orig_set_seed = self._inference._set_seed
        forced = self._seed
        def _forced_set_seed(_ignored):
            random.seed(forced)
            np.random.seed(forced)
            torch.manual_seed(forced)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(forced)
        self._inference._set_seed = _forced_set_seed

        # Pre-seed so even top-level ops that ran before _set_seed is invoked
        # land on the same state.
        _forced_set_seed(forced)
        return self

    def __exit__(self, *args):
        os.urandom = self._orig_urandom
        self._inference._set_seed = self._orig_set_seed


async def synthesize(text: str, speaker: str | None, session_seed: int) -> bytes:
    """Run blocking inference in a thread pool with a fixed seed, return PCM16 bytes."""
    loop = asyncio.get_event_loop()
    async with _GPU_LOCK:
        def _run():
            with _ForceSeed(_INFERENCE, session_seed):
                return _INFERENCE.generate(text, speaker_name=speaker)
        audio = await loop.run_in_executor(None, _run)
    return to_pcm16_bytes(audio)


async def emit_chunk(ws: WebSocket, seq: int, text: str, pcm: bytes, sample_rate: int = 24000):
    """Send a header JSON text frame followed by the raw PCM binary frame."""
    await ws.send_text(json.dumps({
        "type": "chunk_header",
        "seq": seq,
        "text": text,
        "sample_rate": sample_rate,
        "num_bytes": len(pcm),
    }))
    await ws.send_bytes(pcm)


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    await _MODEL_READY.wait()
    # One seed per WS session → same voice for every chunk in this conversation.
    session_seed = int.from_bytes(os.urandom(4), "little")
    await ws.send_text(json.dumps({"type": "ready", "session_seed": session_seed}))

    buffer = ""
    seq = 0
    speaker: str | None = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "text":
                buffer += msg.get("data", "")
                if msg.get("speaker"):
                    speaker = msg["speaker"]
                complete, buffer = split_on_sentences(buffer)
                for sentence in complete:
                    seq += 1
                    try:
                        pcm = await synthesize(sentence, speaker, session_seed)
                        await emit_chunk(ws, seq, sentence, pcm)
                    except Exception as e:
                        log.exception("synthesis failed")
                        await ws.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif mtype == "flush":
                if buffer.strip():
                    seq += 1
                    pcm = await synthesize(buffer.strip(), speaker, session_seed)
                    await emit_chunk(ws, seq, buffer.strip(), pcm)
                    buffer = ""

            elif mtype == "end":
                if buffer.strip():
                    seq += 1
                    pcm = await synthesize(buffer.strip(), speaker, session_seed)
                    await emit_chunk(ws, seq, buffer.strip(), pcm)
                    buffer = ""
                await ws.send_text(json.dumps({"type": "done"}))
                break

    except WebSocketDisconnect:
        log.info("client disconnected")
    except Exception as e:
        log.exception("ws error")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
