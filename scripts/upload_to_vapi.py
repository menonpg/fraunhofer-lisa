#!/usr/bin/env python3
"""
Upload a new project .md file to VAPI's file knowledge base.

Usage:
  python scripts/upload_to_vapi.py docs/projects/56-aim_hairpin_busbar_laser_welding.md

Env vars (or .env):
  VAPI_PRIVATE_KEY  — VAPI private API key
"""

import os
import sys
import requests
from pathlib import Path

# Load .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

VAPI_KEY = os.environ.get("VAPI_PRIVATE_KEY", "bacd5a8c-8974-47d8-af28-cec2746cffda")

if not VAPI_KEY:
    print("❌ VAPI_PRIVATE_KEY not set")
    sys.exit(1)

files = sys.argv[1:] or list(Path("docs/projects").glob("*.md"))

for file_path in files:
    fp = Path(file_path)
    if not fp.exists():
        print(f"❌ Not found: {fp}")
        continue

    print(f"📤 Uploading {fp.name}...")
    with open(fp, "rb") as f:
        resp = requests.post(
            "https://api.vapi.ai/file",
            headers={"Authorization": f"Bearer {VAPI_KEY}"},
            files={"file": (fp.name, f, "text/markdown")},
            timeout=30,
        )

    if resp.status_code in (200, 201):
        d = resp.json()
        print(f"   ✅ id={d.get('id')} status={d.get('status')}")
    else:
        print(f"   ❌ {resp.status_code}: {resp.text[:200]}")
