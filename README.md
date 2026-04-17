# Fraunhofer Lisa — Railway Backend

Voice AI and chat backend for the Fraunhofer CMA technology portfolio site.  
Live: [menonpg.github.io/fraunhofer-cma-projects](https://menonpg.github.io/fraunhofer-cma-projects)  
Dashboard: [menonpg.github.io/fraunhofer-lisa/docs/dashboard.html](https://menonpg.github.io/fraunhofer-lisa/docs/dashboard.html)

---

## Architecture

```
Browser (GitHub Pages)
  ├── v1: docs/index.html         — Text chat + VAPI voice
  └── v2: docs/v2/index.html      — Text chat + LiveKit WebRTC voice

Railway (lisa-webhook-production.up.railway.app)
  ├── /api/chat                   — Text chat (RAG + RLM via soul.py)
  ├── /api/query-fast             — Fast Qdrant retrieval (used by LiveKit agent)
  ├── /api/index                  — Simple text indexing (soul_remember → Qdrant)
  ├── /api/index-project          — Full project indexing (chunked embedding)
  ├── /livekit/start-session      — Create LiveKit room + dispatch agent
  ├── /livekit/transcript         — Receive transcript from LiveKit agent
  ├── /livekit/webhook            — LiveKit Cloud webhook (room events)
  ├── /vapi/webhook               — VAPI webhook (call events + tool calls)
  ├── /api/calls                  — List all calls (VAPI + LiveKit)
  └── /api/health                 — System health + Qdrant collection sizes

LiveKit Cloud (monica-shg8b37e.livekit.cloud)
  └── Agent: CA_tk98nnaezPQR      — Python agent (Deepgram STT + Azure GPT-4o + Cartesia TTS)
                                     Dispatch name: fraunhofer-lisa, Region: eu-central
```

---

## Knowledge Bases

| Store | Collection | Purpose |
|-------|-----------|---------|
| Qdrant | `fraunhofer_cma_projects` | Project KB (vector search for RAG) |
| Qdrant | `fraunhofer_cma_calls` | Call history (visitor conversations) |
| VAPI Files | — | 56 .md files uploaded (file-based KB) |
| soul/MEMORY.md | — | Flat-file context loaded on startup |
| soul/SOUL.md | — | Lisa's identity and persona |
| GitHub | `vapi-calls/*.json` | VAPI call records |
| GitHub | `livekit-calls/*.json` | LiveKit call records |

---

## Adding a New Project

**One command does everything:**
```bash
python scripts/add_project.py /path/to/project.md
```

This script:
1. **Qdrant** — Embeds project file into `fraunhofer_cma_projects` with section chunking (via `/api/index-project`)
2. **VAPI** — Uploads .md file to VAPI file knowledge base
3. **soul/MEMORY.md** — Appends compact project summary for startup context
4. **scripts/vapi_files.json** — Tracks the VAPI file ID for reference

**Then manually:**
1. Create `.md` in `fraunhofer-cma-projects/docs/projects/` (use existing files as template)
2. Add project JS object to `docs/index.html` (v1) and `docs/v2/index.html` (v2) PROJECTS array
3. Add slug to `PROJECT_SLUGS` dict in both HTML files
4. Commit + push `fraunhofer-cma-projects`
5. Commit + push `fraunhofer-lisa` (Railway auto-redeploys)

---

## Environment Variables (Railway)

| Variable | Purpose |
|----------|---------|
| `QDRANT_URL` | Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | Qdrant auth key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI for chat + embeddings |
| `AZURE_OPENAI_KEY` | Azure OpenAI key |
| `AZURE_EMBEDDING_ENDPOINT` | Azure embedding endpoint |
| `AZURE_EMBEDDING_KEY` | Azure embedding key |
| `ANTHROPIC_API_KEY` | Claude fallback |
| `GITHUB_TOKEN` | Push call records to GitHub |
| `GITHUB_REPO` | `menonpg/fraunhofer-lisa` |
| `VAPI_PRIVATE_KEY` | VAPI private key |
| `VAPI_PUBLIC_KEY` | VAPI public key |
| `VAPI_ASSISTANT_ID` | Lisa VAPI assistant ID |
| `LIVEAVATAR_API_KEY` | LiveAvatar API |

---

## LiveKit Agent

Agent code: `fraunhofer-cma-projects/lisa-agent/agent.py`

```bash
# Deploy updated agent
cd fraunhofer-cma-projects/lisa-agent
lk agent deploy \
  --project fraunhofer-lisa \
  --secrets DEEPGRAM_API_KEY=... \
  --secrets AZURE_OPENAI_ENDPOINT=... \
  --secrets AZURE_OPENAI_KEY=... \
  --secrets AZURE_DEPLOYMENT=gpt-4o \
  --secrets AZURE_API_VERSION=2025-01-01-preview \
  --secrets WEBHOOK_BASE=https://lisa-webhook-production.up.railway.app \
  --region eu-central \
  --yes .
```

**Key design decisions:**
- `preemptive_generation=False` — required so RAG injection runs before LLM generates
- `on_user_turn_completed` — overridden in `LisaAgent` to inject Qdrant RAG per turn
- Transcript posted synchronously on room `disconnected` event (async fails on SIGTERM)
- `user_away_timeout=120.0` — prevent early session close on silence

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/add_project.py` | **Use this** to add a new project — updates Qdrant, VAPI, MEMORY.md |
| `scripts/index_projects.py` | Full re-index of all projects from GitHub into Qdrant |
| `scripts/upload_to_vapi.py` | Upload one or more .md files to VAPI file KB |
| `scripts/vapi_files.json` | Tracks all VAPI file uploads (id, name, status) |

---

## Repos

| Repo | Purpose |
|------|---------|
| `menonpg/fraunhofer-cma-projects` | GitHub Pages frontend (v1 + v2) + LiveKit agent |
| `menonpg/fraunhofer-lisa` | This repo — Railway backend, dashboard, soul |

---

## Railway Service IDs

```
Project:     f5449cce-9380-41a8-bd5c-41e0d84fe366
Service:     a5f8c59e-33b2-4b3b-b8df-181e2252ad65
Environment: 6f9bb002-fbcd-41f8-a521-21d2878be7ef
```
