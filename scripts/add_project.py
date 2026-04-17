#!/usr/bin/env python3
"""
Add a new project to ALL three knowledge bases in one shot:
  1. Qdrant (via /api/index on Railway)
  2. VAPI file KB (upload .md file)
  3. VAPI files list stored in this repo's scripts/vapi_files.json (for tracking)

Usage:
  python scripts/add_project.py docs/projects/56-aim_hairpin_busbar_laser_welding.md

Env vars (or .env):
  VAPI_PRIVATE_KEY
  RAILWAY_WEBHOOK_URL   (default: https://lisa-webhook-production.up.railway.app)
"""

import os
import sys
import json
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
WEBHOOK_URL = os.environ.get("RAILWAY_WEBHOOK_URL", "https://lisa-webhook-production.up.railway.app")

if len(sys.argv) < 2:
    print("Usage: python scripts/add_project.py path/to/project.md")
    sys.exit(1)

fp = Path(sys.argv[1])
if not fp.exists():
    print(f"❌ File not found: {fp}")
    sys.exit(1)

text = fp.read_text()
print(f"\n📄 Adding: {fp.name} ({len(text)} chars)")
print("=" * 60)

# 1. Index into Qdrant via Railway /api/index
print("\n1️⃣  Indexing into Qdrant (Railway)...")
resp = requests.post(
    f"{WEBHOOK_URL}/api/index",
    json={"texts": [text]},
    timeout=30,
)
if resp.ok:
    d = resp.json()
    print(f"   ✅ indexed={d.get('indexed')} total={d.get('total')}")
else:
    print(f"   ❌ {resp.status_code}: {resp.text[:200]}")

# 2. Upload to VAPI file KB
print("\n2️⃣  Uploading to VAPI file KB...")
with open(fp, "rb") as f:
    resp = requests.post(
        "https://api.vapi.ai/file",
        headers={"Authorization": f"Bearer {VAPI_KEY}"},
        files={"file": (fp.name, f, "text/markdown")},
        timeout=30,
    )

vapi_file_id = None
if resp.status_code in (200, 201):
    d = resp.json()
    vapi_file_id = d.get("id")
    print(f"   ✅ id={vapi_file_id} status={d.get('status')}")
else:
    print(f"   ❌ {resp.status_code}: {resp.text[:200]}")

# 3. Update tracking file
tracking = Path(__file__).parent / "vapi_files.json"
tracked = json.loads(tracking.read_text()) if tracking.exists() else []
tracked.append({
    "file": fp.name,
    "vapi_id": vapi_file_id,
})
tracking.write_text(json.dumps(tracked, indent=2))
print(f"\n3️⃣  Tracking file updated: {tracking}")

print("\n✅ Done!")
