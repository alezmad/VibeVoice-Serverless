import os

# Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# HuggingFace cache configuration
HF_HOME = os.environ.get("HF_HOME")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME
if HF_HUB_CACHE:
    os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE

# Base directory - use runpod volume if available, otherwise workspace
RUNPOD_VOLUME = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/workspace"
VIBEVOICE_DIR = os.environ.get("VIBEVOICE_DIR", f"{RUNPOD_VOLUME}/vibevoice")

# Torch cache
TORCH_HOME = os.environ.get("TORCH_HOME", f"{VIBEVOICE_DIR}/torch_cache")
os.environ["TORCH_HOME"] = TORCH_HOME

# S3 Configuration
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")

# Directory structure
MODEL_CACHE_DIR = f"{VIBEVOICE_DIR}/models"
OUTPUT_DIR = f"{VIBEVOICE_DIR}/output"
AUDIO_PROMPTS_DIR = f"{VIBEVOICE_DIR}/demo/voices"

# Create dirs if they don't exist
for d in [MODEL_CACHE_DIR, OUTPUT_DIR, AUDIO_PROMPTS_DIR, TORCH_HOME]:
    os.makedirs(d, exist_ok=True)

# Application Configuration
MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", "2000"))
DEFAULT_SAMPLE_RATE = int(os.environ.get("DEFAULT_SAMPLE_RATE", "24000"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "400"))
MIN_LAST_CHUNK_CHARS = int(os.environ.get("MIN_LAST_CHUNK_CHARS", "150"))
CHUNK_SILENCE_MS = int(os.environ.get("CHUNK_SILENCE_MS", "40"))

# Audio configuration
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}
MIN_AUDIO_DURATION = 3.0
MAX_AUDIO_DURATION = 30.0

# VibeVoice model configuration
MODEL_PATH = "vibevoice/VibeVoice-7B"
DEFAULT_SPEAKER = os.environ.get("DEFAULT_SPEAKER", "Alice")
DEFAULT_CFG_SCALE = float(os.environ.get("DEFAULT_CFG_SCALE", "1.3"))
