#!/usr/bin/env python3
"""Pod entrypoint: seed voice files into config.AUDIO_PROMPTS_DIR, then exec pod_server."""
import os
import shutil
import sys

sys.path.insert(0, "/workspace/vibevoice")
import config  # noqa: E402

src = "/workspace/vibevoice/demo/voices"
dst = config.AUDIO_PROMPTS_DIR
os.makedirs(dst, exist_ok=True)

if os.path.isdir(src):
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if not os.path.exists(d) and os.path.isfile(s):
            shutil.copy2(s, d)
            print(f"seeded voice: {d}", flush=True)

os.execvp("python3", ["python3", "/workspace/vibevoice/pod_server.py"])
