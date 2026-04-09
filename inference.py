import sys
import os
import gc
import torch
import numpy as np
import logging
import soundfile as sf
import tempfile
import random
import time
import base64
from pathlib import Path
from typing import Generator, Dict, Any, Tuple

# Add VibeVoice to path (it's cloned at runtime in bootstrap.sh)
# Try both paths
for p in ['/runpod-volume/vibevoice/vibevoice', '/workspace/vibevoice']:
    if os.path.isdir(p):
        sys.path.insert(0, p)
        break

import config

# Import VibeVoice classes
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

log = logging.getLogger(__name__)

# =============================================================================
# LINACODEC IMPORTS (for streaming support)
# =============================================================================
try:
    from linacodec.codec import LinaCodec
    LINACODEC_AVAILABLE = True
    log.info("LinaCodec is available for streaming")
except ImportError:
    LINACODEC_AVAILABLE = False
    log.warning("LinaCodec not available. Streaming will fall back to raw audio.")

# =============================================================================
# LINACODEC GLOBAL CACHE
# =============================================================================
_LINA_CODEC_MODEL = None


def load_linacodec():
    """
    Load LinaCodec encoder/decoder (cached globally)

    Model is auto-downloaded from HuggingFace on first use.
    Uses RunPod's default HF cache paths to benefit from smart caching.
    """
    global _LINA_CODEC_MODEL

    if _LINA_CODEC_MODEL is not None:
        log.info("[VibeVoice] Using cached LinaCodec model")
        return _LINA_CODEC_MODEL

    if not LINACODEC_AVAILABLE:
        raise RuntimeError("LinaCodec is not installed")

    log.info("[VibeVoice] Loading LinaCodec model...")

    # LinaCodec auto-downloads from HuggingFace to default cache
    # RunPod's smart caching will mount the model if already cached
    _LINA_CODEC_MODEL = LinaCodec()

    log.info("[VibeVoice] LinaCodec loaded successfully!")
    return _LINA_CODEC_MODEL


def encode_to_linacodec(audio: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode audio to LinaCodec tokens.

    Args:
        audio: Audio tensor (float32 or bfloat16, 24kHz from VibeVoice)

    Returns:
        Tuple of (tokens, global_embedding)

    Note: LinaCodec automatically upsamples to 48kHz during encoding.
    """
    if not LINACODEC_AVAILABLE:
        raise RuntimeError("LinaCodec is not available")

    lina = load_linacodec()

    # Create temporary file for LinaCodec (it requires a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name

    try:
        # Prepare audio tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        # Squeeze all singleton dimensions (e.g. (1, 1, samples) -> (samples,))
        audio = audio.detach().cpu().squeeze()

        # Convert BFloat16 to Float32 (soundfile doesn't support bfloat16)
        if audio.dtype == torch.bfloat16:
            audio = audio.to(torch.float32)

        # Ensure 2D (channels, time) for torchaudio/soundfile
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 0:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() > 2:
            # If still > 2D, take the first slice of extra dimensions
            while audio.dim() > 2:
                audio = audio[0]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

        # VibeVoice outputs at 24kHz
        sf.write(tmp_wav_path, audio.T, 24000, format='WAV')

        # Encode from file path
        # LinaCodec handles loading and resampling to 48kHz
        tokens, embedding = lina.encode(tmp_wav_path)

        return tokens, embedding

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)


class VoiceMapper:
    """Maps speaker names to voice file paths"""
    
    def __init__(self):
        self.setup_voice_presets()

    def setup_voice_presets(self):
        """Setup voice presets by scanning voices directory."""
        voices_dir = Path(config.AUDIO_PROMPTS_DIR)
        
        if not voices_dir.exists():
            log.warning(f"Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            return
        
        # Scan for all WAV files in voices directory
        self.voice_presets = {}
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        log.info(f"Found {len(self.voice_presets)} voice files in {voices_dir}")
        if self.voice_presets:
            log.info(f"Available voices: {', '.join(self.voice_presets.keys())}")
        else:
            log.warning(f"No .wav files found in {voices_dir}")

    def get_voice_path(self, speaker_name: str) -> str:
        """Get voice file path for a given speaker name"""
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        
        # Try partial matching (case insensitive)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path

        # Default to first voice if no match found
        if not self.voice_presets:
            raise ValueError(
                f"No voice files found in {config.AUDIO_PROMPTS_DIR}. "
                f"Please upload at least one .wav file to the voices directory."
            )

        default_voice = list(self.voice_presets.values())[0]
        log.warning(f"No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice

class VibeVoiceInference:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_chunk_chars = config.MAX_CHUNK_CHARS
        self.voice_mapper = VoiceMapper()
        self.temp_dir = tempfile.mkdtemp()
        log.info(f"Created temporary directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup temporary directory on instance deletion"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                log.debug(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                log.warning(f"Failed to cleanup temp directory {self.temp_dir}: {e}")

    def load_model(self):
        """Load VibeVoice 7B model"""
        if self.model is not None:
            return self.model

        log.info(f"Loading VibeVoice 7B model on {self.device}...")
        
        # Clear any existing GPU memory before loading
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(config.MODEL_PATH)

            # Determine dtype and attention implementation
            if self.device == "cuda":
                load_dtype = torch.bfloat16
                
                # Try flash_attention_2 first (best performance), fall back to sdpa
                try:
                    log.info("Loading model with flash_attention_2...")
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        config.MODEL_PATH,
                        torch_dtype=load_dtype,
                        device_map="cuda",
                        attn_implementation="flash_attention_2",
                        low_cpu_mem_usage=True,
                    )
                    log.info("Model loaded successfully with flash_attention_2")
                except Exception as e:
                    log.warning(f"flash_attention_2 failed ({e}), falling back to sdpa...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        config.MODEL_PATH,
                        torch_dtype=load_dtype,
                        device_map="cuda",
                        attn_implementation="sdpa",
                        low_cpu_mem_usage=True,
                    )
                    log.info("Model loaded successfully with sdpa")
            else:
                load_dtype = torch.float32
                attn_impl = "sdpa"
                log.info(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl}")

                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    config.MODEL_PATH,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                )

            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)

            log.info(f"Model loaded successfully (sample rate: {config.DEFAULT_SAMPLE_RATE})")
            return self.model
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            # Clean up on failure
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            raise

    def _smart_chunk_text(self, text: str, max_chars: int = None) -> list[str]:
        """Split text into chunks only at sentence boundaries to preserve prosody."""
        if max_chars is None:
            max_chars = self.max_chunk_chars

        if len(text) <= max_chars:
            return [text]

        chunks = []
        # ONLY split at sentence endings. Splitting at commas (clauses) destroys VibeVoice prosody.
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']

        def split_keep_boundaries(text_segment: str, boundaries: list[str]) -> list[str]:
            parts = []
            current = ""
            i = 0
            while i < len(text_segment):
                match_found = False
                for boundary in boundaries:
                    if text_segment[i:i+len(boundary)] == boundary:
                        current += boundary
                        parts.append(current)
                        current = ""
                        i += len(boundary)
                        match_found = True
                        break
                
                if not match_found:
                    current += text_segment[i]
                    i += 1
            
            if current:
                parts.append(current)
            return parts

        # Split into sentences
        segments = split_keep_boundaries(text, sentence_endings)
        current_chunk = ""

        for segment in segments:
            if len(segment) > max_chars:
                # Flush existing buffer
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Sentence itself is too long. This is rare (>800 chars).
                # Fall back to word-level split as a last resort.
                words = segment.split()
                for word in words:
                    prefix = " " if current_chunk else ""
                    if len(current_chunk) + len(word) + 1 <= max_chars:
                        current_chunk += prefix + word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
            else:
                # Segment fits. Try to group sentences up to max_chars.
                prefix = "" if not current_chunk or current_chunk.endswith(" ") else " "
                if len(current_chunk) + len(segment) + len(prefix) <= max_chars:
                    current_chunk += prefix + segment
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = segment

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Cleanup
        chunks = [c.strip() for c in chunks if c.strip()]

        # Merge tiny final chunks
        min_last_chunk_chars = min(
            getattr(config, "MIN_LAST_CHUNK_CHARS", 0),
            max_chars,
        )
        if len(chunks) > 1 and min_last_chunk_chars > 0:
            last_len = len(chunks[-1])
            if last_len < min_last_chunk_chars:
                # Merge last chunk into previous
                merged = f"{chunks[-2]} {chunks[-1]}".strip()
                chunks[-2] = merged
                chunks.pop()
                log.info(
                    "Last chunk too short (%s chars); merged into previous (min %s).",
                    last_len,
                    min_last_chunk_chars,
                )

        if not chunks:
            return [text]

        log.info(f"Split text into {len(chunks)} chunks (max {max_chars} chars each, sentence-level only)")
        return chunks

    def _set_seed(self, seed: int) -> None:
        """Helper to set seed across all libraries for consistency"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate(
        self,
        text,
        speaker_name=None,
        cfg_scale=1.3,
        disable_prefill=False,
        max_chunk_chars=None,
    ):
        """Generate audio from text using VibeVoice 7B with smart chunking"""
        if self.model is None:
            self.load_model()

        log.info(f"Generating audio for text ({len(text)} chars): {text[:50]}...")

        session_seed = int.from_bytes(os.urandom(4), "little")
        log.info(f"Using session seed: {session_seed}")

        # Get voice path
        speaker_name = speaker_name or config.DEFAULT_SPEAKER
        voice_path = self.voice_mapper.get_voice_path(speaker_name)

        # Smart chunk text if it's too long
        chunk_limit = max_chunk_chars if max_chunk_chars is not None else self.max_chunk_chars
        chunks = self._smart_chunk_text(text, chunk_limit)

        if len(chunks) == 1:
            # Single chunk, generate directly
            formatted_text = f"Speaker 1: {chunks[0]}"
            temp_txt_path = None
            try:
                temp_txt_path = os.path.join(self.temp_dir, f"temp_{os.getpid()}.txt")
                with open(temp_txt_path, 'w') as f:
                    f.write(formatted_text)

                with torch.no_grad():
                    self._set_seed(session_seed)
                    # voice_samples must be a list of lists: [[path1, path2, ...]]
                    inputs = self.processor(
                        text=[formatted_text],
                        voice_samples=[[voice_path]],  # List of lists!
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )

                    # Move tensors to target device
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            inputs[k] = v.to(self.device)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        verbose=False,
                        is_prefill=not disable_prefill,
                    )
                
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)

                if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                    return outputs.speech_outputs[0]
                return None

            except Exception:
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                raise

        # Multiple chunks, generate and concatenate
        log.info(f"Generating {len(chunks)} chunks (max {chunk_limit} chars each)...")
        combined_wav = []
        
        for i, chunk_text in enumerate(chunks, 1):
            log.info(f"Generating chunk {i}/{len(chunks)} ({len(chunk_text)} chars)...")
            
            formatted_chunk = f"Speaker 1: {chunk_text}"
            temp_txt_path = None
            
            try:
                temp_txt_path = os.path.join(self.temp_dir, f"temp_{os.getpid()}_{i}.txt")
                with open(temp_txt_path, 'w') as f:
                    f.write(formatted_chunk)
                    
                with torch.no_grad():
                    self._set_seed(session_seed)
                    inputs = self.processor(
                        text=[formatted_chunk],
                        voice_samples=[[voice_path]],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            inputs[k] = v.to(self.device)
                            
                    chunk_wav = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        verbose=False,
                        is_prefill=not disable_prefill,
                    )
                
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                
                if chunk_wav.speech_outputs and chunk_wav.speech_outputs[0] is not None:
                    wav_data = chunk_wav.speech_outputs[0]
                    if torch.is_tensor(wav_data):
                        # Convert BFloat16 to Float32 before numpy (BFloat16 not supported by numpy)
                        if wav_data.dtype == torch.bfloat16:
                            wav_data = wav_data.float()
                        wav_data = wav_data.cpu().numpy().squeeze()
                    combined_wav.append(wav_data)
                    
                    # Add silence between chunks
                    if i < len(chunks):
                        silence = np.zeros(int(config.DEFAULT_SAMPLE_RATE * (config.CHUNK_SILENCE_MS / 1000)))
                        combined_wav.append(silence)
                        
            except Exception as e:
                log.error(f"Error generating chunk {i}: {e}")
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                # Continue with other chunks or fail? For now fail
                raise e

        if not combined_wav:
            return None
            
        return np.concatenate(combined_wav)

    def generate_stream(
        self,
        text,
        speaker_name=None,
        cfg_scale=1.3,
        disable_prefill=False,
        max_chunk_chars=None,
    ):
        """Generate audio in chunks and yield each chunk as it completes."""
        if self.model is None:
            self.load_model()

        session_seed = int.from_bytes(os.urandom(4), "little")
        log.info(f"Using session seed: {session_seed}")

        # Get voice path
        speaker_name = speaker_name or config.DEFAULT_SPEAKER
        voice_path = self.voice_mapper.get_voice_path(speaker_name)

        # Smart chunk text if it's too long
        chunk_limit = max_chunk_chars if max_chunk_chars is not None else self.max_chunk_chars
        chunks = self._smart_chunk_text(text, chunk_limit)

        log.info(f"Streaming {len(chunks)} chunks (max_chunk_chars={chunk_limit})...")

        for i, chunk_text in enumerate(chunks, 1):
            log.info(f"Generating chunk {i}/{len(chunks)} ({len(chunk_text)} chars)...")

            formatted_chunk = f"Speaker 1: {chunk_text}"
            temp_txt_path = None

            try:
                # VibeVoice might rely on file existence in some internal path checks?
                # Keeping file writing just in case, matching original behavior.
                temp_txt_path = os.path.join(self.temp_dir, f"temp_{os.getpid()}_{i}.txt")
                with open(temp_txt_path, 'w') as f:
                    f.write(formatted_chunk)

                with torch.no_grad():
                    self._set_seed(session_seed)
                    inputs = self.processor(
                        text=[formatted_chunk],
                        voice_samples=[[voice_path]],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )

                    # Move tensors to target device
                    for k, v in inputs.items():
                        if torch.is_tensor(v):
                            inputs[k] = v.to(self.device)

                    chunk_wav = self.model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        verbose=False,
                        is_prefill=not disable_prefill,
                    )

                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)

                if chunk_wav.speech_outputs and chunk_wav.speech_outputs[0] is not None:
                    yield chunk_wav.speech_outputs[0]

            except Exception:
                if temp_txt_path and os.path.exists(temp_txt_path):
                    os.remove(temp_txt_path)
                raise

    def encode_mp3(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Encode PCM numpy array to MP3 bytes using ffmpeg"""
        import subprocess

        # Ensure int16
        if audio_array.dtype != np.int16:
            audio_int16 = (audio_array * 32767).astype(np.int16)
        else:
            audio_int16 = audio_array

        raw_bytes = audio_int16.tobytes()

        try:
            process = subprocess.Popen(
                ['ffmpeg', '-y', '-f', 's16le', '-ar', str(sample_rate), '-ac', '1',
                 '-i', 'pipe:0', '-f', 'mp3', '-b:a', '192k', 'pipe:1'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            mp3_bytes, _ = process.communicate(input=raw_bytes)
            return mp3_bytes
        except Exception as e:
            log.error(f"FFmpeg encoding failed: {e}")
            return b""

    def generate_audio_stream_decoded(
        self,
        text,
        speaker_name=None,
        cfg_scale=1.3,
        disable_prefill=False,
        max_chunk_chars=None,
        output_format="pcm_16",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate streaming audio with LinaCodec compression then decode.

        This is the compatibility mode for Cloudflare Workers:
        - Generate audio chunk (24kHz from VibeVoice)
        - Encode to LinaCodec tokens (compression)
        - Decode back to audio (quality upgrade to 48kHz)
        - Yield decoded audio chunk as base64 PCM-16 or MP3

        This gives us the quality benefits of LinaCodec (48kHz output) while
        outputting standard PCM that doesn't require browser decoding.

        Args:
            text: Text to synthesize
            speaker_name: Speaker name for voice cloning
            cfg_scale: Classifier-free guidance scale
            disable_prefill: Disable prefill optimization
            max_chunk_chars: Maximum characters per chunk
            output_format: 'pcm_16' or 'mp3'

        Yields:
            Dictionaries with streaming chunk data
        """
        # Determine target format
        is_mp3 = output_format == "mp3"

        if not LINACODEC_AVAILABLE:
            # Fallback: stream raw audio without LinaCodec
            log.warning("[Streaming] LinaCodec not available, streaming raw audio at 24kHz")

            for chunk_num, chunk_wav in enumerate(self.generate_stream(
                text=text,
                speaker_name=speaker_name,
                cfg_scale=cfg_scale,
                disable_prefill=disable_prefill,
                max_chunk_chars=max_chunk_chars,
            ), 1):
                # Convert to numpy (handle BFloat16)
                chunk_tensor = chunk_wav.squeeze(0)
                if chunk_tensor.dtype == torch.bfloat16:
                    chunk_tensor = chunk_tensor.float()
                audio_array = chunk_tensor.cpu().numpy()

                # Encode if MP3 requested
                if is_mp3:
                    audio_bytes = self.encode_mp3(audio_array, 24000)
                    fmt = "mp3"
                else:
                    audio_array = np.clip(audio_array, -1.0, 1.0)
                    audio_int16 = (audio_array * 32767.0).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()
                    fmt = "pcm_16"

                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                yield {
                    'status': 'streaming',
                    'chunk': chunk_num,
                    'format': fmt,
                    'audio_chunk': audio_b64,
                    'sample_rate': 24000
                }

            yield {
                'status': 'complete',
                'format': fmt,
                'message': 'All chunks streamed (no LinaCodec)'
            }
            return

        # LinaCodec is available - use encode/decode pattern for 48kHz output
        # IMPORTANT: Load VibeVoice FIRST to claim GPU memory before LinaCodec
        if self.model is None:
            self.load_model()

        lina = load_linacodec()
        start_time = time.time()

        for chunk_num, chunk_wav in enumerate(self.generate_stream(
            text=text,
            speaker_name=speaker_name,
            cfg_scale=cfg_scale,
            disable_prefill=disable_prefill,
            max_chunk_chars=max_chunk_chars,
        ), 1):
            chunk_start = time.time()

            # Encode to LinaCodec tokens
            tokens, embedding = encode_to_linacodec(chunk_wav)

            # Decode back to audio (now at 48kHz!)
            decoded_audio = lina.decode(tokens, embedding)

            process_time = time.time() - chunk_start

            # Convert to base64 for transmission
            audio_array = decoded_audio.cpu().numpy() if hasattr(decoded_audio, 'cpu') else decoded_audio

            log.info(f"[Streaming] Chunk {chunk_num} decoded: shape={audio_array.shape}, dtype={audio_array.dtype}")

            # Ensure 1D array (squeeze if needed)
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()

            # Convert float32 to int16 PCM
            audio_int16 = (audio_array * 32767).astype(np.int16)

            if is_mp3:
                # Encode to MP3 @ 48kHz
                audio_bytes = self.encode_mp3(audio_int16, 48000)
                fmt = "mp3"
            else:
                audio_bytes = audio_int16.tobytes()
                fmt = "pcm_16"

            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            log.debug(f"[Streaming] Chunk {chunk_num}: {len(audio_array)} samples (process: {process_time:.3f}s)")

            yield {
                'status': 'streaming',
                'chunk': chunk_num,
                'format': fmt,
                'audio_chunk': audio_b64,
                'sample_rate': 48000,
                'process_time_ms': process_time * 1000
            }

        elapsed = time.time() - start_time
        log.info(f"[Streaming] Complete: {chunk_num} chunks, {elapsed:.2f}s")

        yield {
            'status': 'complete',
            'format': fmt,
            'message': 'All chunks streamed',
            'total_chunks': chunk_num,
            'elapsed_time_seconds': elapsed
        }
