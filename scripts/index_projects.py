#!/usr/bin/env python3
"""
Index all Fraunhofer CMA project .md files into Qdrant.

Usage:
  python scripts/index_projects.py

Env vars (or .env file):
  QDRANT_URL, QDRANT_API_KEY
  AZURE_EMBEDDING_ENDPOINT, AZURE_EMBEDDING_KEY
  GITHUB_TOKEN
"""

import os
import re
import sys
import json
import hashlib
import requests
from pathlib import Path

# Load .env file if present
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "fraunhofer_cma_projects")
AZURE_EMBEDDING_ENDPOINT = os.environ.get("AZURE_EMBEDDING_ENDPOINT", "")
AZURE_EMBEDDING_KEY = os.environ.get("AZURE_EMBEDDING_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

GITHUB_REPO = "menonpg/fraunhofer-cma-projects"
PROJECTS_PATH = "docs/projects"
EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
VECTOR_SIZE = 3072


def github_list_files(repo, path):
    """List files in a GitHub repo directory."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return [f for f in resp.json() if f["name"].endswith(".md")]


def github_get_file(repo, file_path):
    """Get raw file content from GitHub."""
    url = f"https://raw.githubusercontent.com/{repo}/main/{file_path}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_project_md(content, filename):
    """Parse a project .md file into structured chunks."""
    chunks = []

    # Extract title
    title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else filename.replace(".md", "").replace("-", " ").title()

    # Split by ## headers
    sections = re.split(r'\n##\s+', content)

    # Full document chunk
    full_text = f"Project: {title}\n\n{content[:3000]}"
    chunks.append({
        "text": full_text,
        "metadata": {
            "project": title,
            "section": "full",
            "filename": filename,
        }
    })

    # Individual section chunks
    for section in sections[1:]:
        lines = section.split("\n", 1)
        section_title = lines[0].strip()
        section_body = lines[1].strip() if len(lines) > 1 else ""

        if len(section_body) < 20:
            continue

        chunk_text = f"Project: {title}\nSection: {section_title}\n\n{section_body}"
        chunks.append({
            "text": chunk_text[:2000],
            "metadata": {
                "project": title,
                "section": section_title.lower(),
                "filename": filename,
            }
        })

    return chunks


def embed_texts(texts, batch_size=16):
    """Create embeddings using Azure text-embedding-3-large."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        url = f"{AZURE_EMBEDDING_ENDPOINT}/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings?api-version=2023-05-15"
        resp = requests.post(
            url,
            headers={"api-key": AZURE_EMBEDDING_KEY, "Content-Type": "application/json"},
            json={"input": batch},
            timeout=60,
        )
        resp.raise_for_status()
        embeddings = [d["embedding"] for d in resp.json()["data"]]
        all_embeddings.extend(embeddings)
        print(f"  Embedded batch {i//batch_size + 1} ({len(batch)} texts)")
    return all_embeddings


def qdrant_ensure_collection(collection_name):
    """Create Qdrant collection if it doesn't exist."""
    url = f"{QDRANT_URL}/collections/{collection_name}"
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

    # Check if exists
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        print(f"✅ Collection '{collection_name}' already exists")
        return

    # Create
    resp = requests.put(
        url,
        headers=headers,
        json={"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"✅ Created collection '{collection_name}'")


def qdrant_upsert(collection_name, points):
    """Upsert points into Qdrant."""
    url = f"{QDRANT_URL}/collections/{collection_name}/points"
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

    # Batch in groups of 50
    for i in range(0, len(points), 50):
        batch = points[i:i + 50]
        resp = requests.put(
            url,
            headers=headers,
            json={"points": batch},
            timeout=60,
        )
        resp.raise_for_status()
        print(f"  Upserted batch {i//50 + 1} ({len(batch)} points)")


def qdrant_count(collection_name):
    """Count points in collection."""
    url = f"{QDRANT_URL}/collections/{collection_name}/points/count"
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json={"exact": True}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("result", {}).get("count", 0)


def main():
    print("=" * 60)
    print("Fraunhofer CMA Projects — Qdrant Indexer")
    print("=" * 60)

    # Validate config
    missing = []
    if not QDRANT_URL: missing.append("QDRANT_URL")
    if not QDRANT_API_KEY: missing.append("QDRANT_API_KEY")
    if not AZURE_EMBEDDING_ENDPOINT: missing.append("AZURE_EMBEDDING_ENDPOINT")
    if not AZURE_EMBEDDING_KEY: missing.append("AZURE_EMBEDDING_KEY")
    if missing:
        print(f"❌ Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    # 1. List project files from GitHub
    print(f"\n📂 Listing projects from {GITHUB_REPO}/{PROJECTS_PATH}...")
    files = github_list_files(GITHUB_REPO, PROJECTS_PATH)
    print(f"   Found {len(files)} project files")

    # 2. Download and parse each project
    all_chunks = []
    for f in files:
        filename = f["name"]
        print(f"  📄 {filename}...")
        try:
            content = github_get_file(GITHUB_REPO, f"{PROJECTS_PATH}/{filename}")
            chunks = parse_project_md(content, filename)
            all_chunks.extend(chunks)
            print(f"     → {len(chunks)} chunks")
        except Exception as e:
            print(f"     ❌ Error: {e}")

    print(f"\n📊 Total chunks: {len(all_chunks)}")

    # 3. Create embeddings
    print("\n🧮 Creating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    # 4. Prepare Qdrant points
    points = []
    for chunk, embedding in zip(all_chunks, embeddings):
        # Generate deterministic ID from text
        text_hash = int(hashlib.md5(chunk["text"].encode()).hexdigest()[:16], 16)
        points.append({
            "id": text_hash,
            "vector": embedding,
            "payload": {
                "text": chunk["text"],
                "project": chunk["metadata"]["project"],
                "section": chunk["metadata"]["section"],
                "filename": chunk["metadata"]["filename"],
            }
        })

    # 5. Ensure collection exists and upsert
    print(f"\n📤 Upserting into Qdrant collection '{QDRANT_COLLECTION}'...")
    qdrant_ensure_collection(QDRANT_COLLECTION)
    qdrant_upsert(QDRANT_COLLECTION, points)

    # 6. Verify
    count = qdrant_count(QDRANT_COLLECTION)
    print(f"\n✅ Done! Collection '{QDRANT_COLLECTION}' now has {count} points")
    print(f"   Projects indexed: {len(files)}")
    print(f"   Chunks indexed: {len(points)}")


if __name__ == "__main__":
    main()
