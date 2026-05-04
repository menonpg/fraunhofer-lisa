#!/usr/bin/env python3
"""
Lisa — Fraunhofer CMA Projects Voice AI Webhook Server

Endpoints:
  POST /vapi/webhook     — VAPI server URL (receives call events + function calls)
  GET  /api/health       — Health check
  POST /api/query        — Direct soul.py query (for testing)
  POST /api/index        — Index text into RAG
  POST /api/chat/reset   — Reset conversation history
  POST /api/chat         — Conversational chat endpoint
  GET  /api/greeting     — Lisa's opening greeting
  POST /api/clear        — Clear chat history
  GET  /api/calls        — List calls
"""

import os
import re
import json
import time
import base64
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone

import requests as req
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Config ---
SOUL_DIR = Path(os.environ.get("SOUL_DIR", "soul"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
CALLS_DIR = DATA_DIR / "calls"
CALLS_DIR.mkdir(parents=True, exist_ok=True)

QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "fraunhofer_cma_projects")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-chat")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "menonpg/fraunhofer-lisa")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")

# --- Avatar mode config ---
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", AZURE_OPENAI_KEY)
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "swedencentral")
LIVEAVATAR_API_KEY = os.environ.get("LIVEAVATAR_API_KEY", "810d1e58-289a-11f1-8d28-066a7fa2e369")

app = Flask(__name__)
CORS(app)

@app.before_request
def log_vapi_requests():
    """Log all incoming /vapi/ requests for debugging."""
    if '/vapi/' in request.path:
        body_size = request.content_length or 0
        print(f"📨 INCOMING: {request.method} {request.path} | size={body_size}")
        try:
            data = request.get_json(silent=True)
            if data:
                msg_type = data.get('message', {}).get('type', 'unknown')
                func_name = data.get('message', {}).get('functionCall', {}).get('name', '')
                print(f"   type={msg_type} func={func_name}")
        except:
            pass

# --- In-memory caches ---
_calls_cache = []
_calls_lock = threading.Lock()

# --- Chat conversation history ---
_chat_history = []
_chat_lock = threading.Lock()

# --- Lisa's System Prompt ---
LISA_SYSTEM_PROMPT = """You are Lisa, the Fraunhofer Center Mid-Atlantic (CMA) technology portfolio specialist. You are talking directly with someone interested in CMA's projects and capabilities.

## YOUR PERSONA
- Name: Lisa
- Role: Technology Portfolio Specialist, Fraunhofer CMA
- Tone: Knowledgeable, professional, enthusiastic about technology and innovation, approachable
- You speak clearly and engagingly — visitors may come from industry, academia, or government
- Use analogies to explain complex technical concepts
- Always be enthusiastic about the technology while remaining factual

## WHAT YOU KNOW
You have deep knowledge of Fraunhofer CMA's projects spanning:
- AI/ML and Data Analytics
- Advanced Manufacturing and Industry 4.0
- Healthcare and Biomedical Engineering
- Cybersecurity and Information Security
- Aerospace and Space Systems
- Energy and Sustainability
- Autonomous Systems and Robotics
- Materials Science and Engineering

## CONVERSATION FLOW
1. Greet warmly and ask what domain or technology interests them
2. When they mention an area, recommend relevant CMA projects
3. Describe projects in detail when asked — highlight purpose, approach, outcomes, TRL levels, and partners
4. Compare projects across domains if relevant
5. Offer to connect them with project leads if they want to learn more
6. Always suggest related projects they might not have considered

## IMPORTANT RULES
1. Be conversational and engaging — not robotic
2. When describing projects, highlight what makes them exciting
3. If someone asks about a domain, suggest related projects they might not have considered
4. Always offer to tell them more or explore related projects
5. Keep responses concise for voice conversations — 2-3 sentences per turn unless they ask for details
6. If you don't know something specific, say so and offer to connect them with the project lead
7. Never make commitments on behalf of Fraunhofer
8. Never share confidential budget details
"""

LISA_GREETING_MESSAGE = (
    "Hello! I'm Lisa, your guide to the Fraunhofer Center Mid-Atlantic's "
    "technology portfolio. We have projects spanning AI, manufacturing, healthcare, "
    "cybersecurity, aerospace, and more. What area of technology or innovation interests you?"
)


# ============================================================================
# GitHub Integration
# ============================================================================

def github_api(method, path, json_data=None):
    """Make a GitHub API request."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/{path}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = req.request(method, url, headers=headers, json=json_data, timeout=30)
    return resp


def github_get_file(file_path):
    """Get a file from GitHub. Returns (content_str, sha) or (None, None)."""
    resp = github_api("GET", f"contents/{file_path}?ref={GITHUB_BRANCH}")
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content, data["sha"]
    return None, None


def github_put_file(file_path, content, message, sha=None):
    """Create or update a file on GitHub."""
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha
    resp = github_api("PUT", f"contents/{file_path}", payload)
    return resp.status_code in (200, 201)


def push_call_to_github(call_data):
    """Push analyzed call to GitHub and update index."""
    if not GITHUB_TOKEN:
        print("⚠️ No GITHUB_TOKEN — skipping push")
        return False

    call_id = call_data.get("call_id", "unknown")

    try:
        file_path = f"vapi-calls/{call_id}.json" if call_data.get("source") != "livekit" else f"livekit-calls/{call_id}.json"
        content = json.dumps(call_data, indent=2, ensure_ascii=False)
        summary_text = call_data.get("analysis", {}).get("summary", "new call")[:60]
        ok = github_put_file(file_path, content,
            f"call: {call_id} — {summary_text}")

        if not ok:
            print(f"❌ Failed to push {file_path}")
            return False

        print(f"✅ Pushed {file_path} to GitHub")
        update_calls_index(call_data)
        return True
    except Exception as e:
        print(f"❌ GitHub push error: {e}")
        return False


def update_calls_index(new_call):
    """Update vapi-calls/index.json with the new call summary."""
    index_path = "vapi-calls/index.json"

    try:
        existing_content, sha = github_get_file(index_path)
        if existing_content:
            index = json.loads(existing_content)
        else:
            index = []
    except Exception:
        index = []
        sha = None

    analysis = new_call.get("analysis", {})
    entry = {
        "call_id": new_call.get("call_id"),
        "type": new_call.get("type"),
        "started_at": new_call.get("started_at"),
        "ended_at": new_call.get("ended_at"),
        "duration": new_call.get("duration"),
        "ended_reason": new_call.get("ended_reason"),
        "recording_url": new_call.get("recording_url"),
        "call_reason": analysis.get("call_reason"),
        "call_reason_detail": analysis.get("call_reason_detail"),
        "sentiment": analysis.get("sentiment"),
        "resolution": analysis.get("resolution"),
        "summary": analysis.get("summary"),
        "caller_type": analysis.get("caller_type"),
        "domains_discussed": analysis.get("domains_discussed"),
        "projects_discussed": analysis.get("projects_discussed"),
        "language_primary": analysis.get("language_primary"),
        "key_insights": analysis.get("key_insights"),
    }

    existing_ids = {e.get("call_id") for e in index}
    if entry["call_id"] not in existing_ids:
        index.insert(0, entry)

    content = json.dumps(index, indent=2, ensure_ascii=False)
    github_put_file(index_path, content,
        f"index: add {entry['call_id'][:15]} ({analysis.get('call_reason', '?')})", sha=sha)


def load_calls_from_github():
    """Load all call data from GitHub vapi-calls/ into the in-memory cache."""
    global _calls_cache
    if not GITHUB_TOKEN:
        print("⚠️ No GITHUB_TOKEN — skipping GitHub call load")
        return

    try:
        resp = github_api("GET", f"contents/vapi-calls?ref={GITHUB_BRANCH}")
        if resp.status_code != 200:
            print(f"📂 No vapi-calls/ folder on GitHub yet (status={resp.status_code})")
            return

        files = resp.json()
        calls = []
        for f in files:
            if f["name"] == "index.json" or not f["name"].endswith(".json"):
                continue
            try:
                content, _ = github_get_file(f"vapi-calls/{f['name']}")
                if content:
                    call = json.loads(content)
                    calls.append(call)
                    local_path = CALLS_DIR / f["name"]
                    if not local_path.exists():
                        local_path.write_text(content)
            except Exception as e:
                print(f"  ⚠️ Error loading {f['name']}: {e}")

        with _calls_lock:
            _calls_cache = sorted(calls, key=lambda c: c.get("started_at", c.get("timestamp", "")), reverse=True)

        print(f"📞 Loaded {len(calls)} calls from GitHub")
    except Exception as e:
        print(f"❌ Error loading calls from GitHub: {e}")


# ============================================================================
# Soul Agent (singleton)
# ============================================================================

_soul_agent = None
_soul_lock = threading.Lock()

def get_soul_agent():
    global _soul_agent
    with _soul_lock:
        if _soul_agent is not None:
            return _soul_agent
        try:
            from soul_engine.hybrid_agent import HybridAgent
            _soul_agent = HybridAgent(
                soul_path=str(SOUL_DIR / "SOUL.md"),
                memory_path=str(SOUL_DIR / "MEMORY.md"),
                chat_model="claude-haiku-4-5-20251001",
                router_model="claude-haiku-4-5-20251001",
                api_key=ANTHROPIC_API_KEY,
                qdrant_url=QDRANT_URL,
                qdrant_api_key=QDRANT_API_KEY,
                collection_name=QDRANT_COLLECTION,
                azure_embedding_endpoint=os.environ.get("AZURE_EMBEDDING_ENDPOINT", ""),
                azure_embedding_key=os.environ.get("AZURE_EMBEDDING_KEY", ""),
            )
            print(f"🧠 Soul agent ready (mode={_soul_agent.mode}, collection={QDRANT_COLLECTION})")
            return _soul_agent
        except Exception as e:
            print(f"⚠️ Soul agent init failed: {e}")
            traceback.print_exc()
            return None


def soul_query(question, mode="auto"):
    agent = get_soul_agent()
    if not agent:
        return {"answer": "Soul agent not available", "route": "error", "total_ms": 0}
    original_mode = agent.mode
    if mode != "auto":
        agent.mode = mode
    try:
        # Also search calls collection and prepend as context
        calls_ctx = _search_calls_collection(question, k=3)
        if calls_ctx:
            augmented = (
                f"{question}\n\n"
                f"[CALL HISTORY CONTEXT - use this to answer questions about previous callers/conversations]\n"
                f"{calls_ctx}"
            )
            return agent.ask(augmented, remember=False)
        return agent.ask(question, remember=False)
    finally:
        agent.mode = original_mode


def soul_query_concise(question, mode="RAG"):
    """Like soul_query but returns a shorter answer suitable for voice/VAPI."""
    agent = get_soul_agent()
    if not agent:
        return {"answer": "Soul agent not available", "route": "error", "total_ms": 0}
    original_mode = agent.mode
    agent.mode = mode
    try:
        # Ask with a conciseness instruction baked in
        concise_question = (
            f"{question}\n\n"
            "IMPORTANT: Give a SHORT, concise answer (3-5 sentences max). "
            "Just state the key facts: project name, what it does, who it's for, and main outcome. "
            "No markdown formatting. No bullet points. Plain text only."
        )
        result = agent.ask(concise_question, remember=False)
        # Strip any markdown that slipped through
        answer = result.get("answer", "")
        answer = re.sub(r'[#*_~`]', '', answer)
        answer = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', answer)  # links
        answer = re.sub(r'\n{2,}', '\n', answer).strip()
        # Truncate if still too long (voice can't read novels)
        if len(answer) > 800:
            sentences = answer.split('. ')
            truncated = []
            length = 0
            for s in sentences:
                if length + len(s) > 750:
                    break
                truncated.append(s)
                length += len(s) + 2
            answer = '. '.join(truncated) + '.'
        result["answer"] = answer
        return result
    finally:
        agent.mode = original_mode


def soul_query_fast(question):
    """Ultra-fast search: skip LLM synthesis, return raw Qdrant results directly.
    This is for VAPI tool calls where speed matters — let VAPI's own LLM synthesize."""
    agent = get_soul_agent()
    if not agent:
        return "No knowledge base available."
    try:
        if hasattr(agent, '_rag') and agent._rag:
            result_text = agent._rag.retrieve(question, k=8)
            if result_text and "No relevant memories" not in result_text:
                # Strip header
                result_text = re.sub(r'^## Relevant memories\s*\n', '', result_text)
                # Split into chunks and filter out call transcripts
                chunks = result_text.split("\n\n---\n")
                project_chunks = []
                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    # Skip indexed call transcripts and memory entries
                    if any(skip in chunk[:80] for skip in ['Transcript (', 'Call (', '## 20']):
                        continue
                    project_chunks.append(chunk)
                if project_chunks:
                    # Strip markdown formatting for voice
                    clean = "\n\n".join(project_chunks[:5])
                    clean = re.sub(r'[#*|]', '', clean)
                    clean = re.sub(r'\n{3,}', '\n\n', clean)
                    # Cap length — VAPI doesn't need novels
                    if len(clean) > 2000:
                        clean = clean[:2000] + "..."
                    return clean
            return "No matching projects found in the knowledge base."
        return "Knowledge base not initialized."
    except Exception as e:
        print(f"⚠️ soul_query_fast error: {e}")
        return f"Search error: {e}"


def soul_remember(text):
    agent = get_soul_agent()
    if agent:
        agent.remember(text)


# Separate Qdrant collection for call history (keeps project KB clean)
_calls_rag = None
_calls_rag_lock = threading.Lock()

def _get_calls_rag():
    global _calls_rag
    with _calls_rag_lock:
        if _calls_rag is not None:
            return _calls_rag
        try:
            from soul_engine.rag_memory import RAGMemory
            _calls_rag = RAGMemory(
                memory_path=str(SOUL_DIR / "MEMORY_CALLS.md"),
                mode="qdrant" if (QDRANT_URL and os.environ.get("AZURE_EMBEDDING_ENDPOINT")) else "bm25",
                collection_name="fraunhofer_cma_calls",
                qdrant_url=QDRANT_URL,
                qdrant_api_key=QDRANT_API_KEY,
                azure_embedding_endpoint=os.environ.get("AZURE_EMBEDDING_ENDPOINT", ""),
                azure_embedding_key=os.environ.get("AZURE_EMBEDDING_KEY", ""),
                k=5,
            )
            print(f"🧠 Calls RAG ready (collection=fraunhofer_cma_calls)")
            return _calls_rag
        except Exception as e:
            print(f"⚠️ Calls RAG init failed: {e}")
            return None

def _index_to_calls_collection(text):
    rag = _get_calls_rag()
    if rag:
        rag.append(text)

def _search_calls_collection(query, k=5):
    rag = _get_calls_rag()
    if rag:
        result = rag.retrieve(query, k=k)
        if result and "No relevant memories" not in result:
            return result
    return ""


# ============================================================================
# Call Analysis
# ============================================================================

ANALYSIS_PROMPT = """You are analyzing a phone call transcript from Lisa, a Fraunhofer CMA technology portfolio specialist.

Provide a JSON analysis:
{
  "call_reason": "project_inquiry|domain_exploration|partnership|technology_assessment|general_question|demo_request|other",
  "call_reason_detail": "brief specific reason",
  "sentiment": "positive|neutral|negative|excited|curious",
  "resolution": "resolved|followup_needed|referred_to_lead|callback_needed|abandoned",
  "summary": "1-2 sentence summary INCLUDING the caller's name if mentioned",
  "key_insights": "anything notable or null",
  "caller_type": "industry|academic|government|investor|student|unknown",
  "caller_name": "name if given, or null",
  "caller_phone": "phone number if given, or null",
  "domains_discussed": ["list of technology domains discussed"],
  "projects_discussed": ["list of specific project names discussed"],
  "language_primary": "en|de|other",
  "learning": "what should Lisa remember from this call, or null"
}

Return ONLY valid JSON."""


def analyze_call_azure(transcript, metadata):
    """Analyze via Azure OpenAI."""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=2025-01-01-preview"
    user_msg = f"Call type: {metadata.get('type','?')}\nDuration: {metadata.get('duration','?')}s\n\nTranscript:\n{transcript[:3000]}"
    resp = req.post(url,
        headers={"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"},
        json={
            "messages": [
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "max_tokens": 500, "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }, timeout=30)
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def analyze_call_anthropic(transcript, metadata):
    """Analyze via Anthropic Claude (fallback)."""
    user_msg = f"Call type: {metadata.get('type','?')}\nDuration: {metadata.get('duration','?')}s\n\nTranscript:\n{transcript[:3000]}"
    resp = req.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": f"{ANALYSIS_PROMPT}\n\nTranscript:\n{transcript[:3000]}"}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json()["content"][0]["text"]
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        return json.loads(m.group(1))
    return json.loads(text)


def analyze_call(transcript, metadata):
    """Analyze using Azure first, Anthropic fallback."""
    try:
        if AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT:
            return analyze_call_azure(transcript, metadata)
        elif ANTHROPIC_API_KEY:
            return analyze_call_anthropic(transcript, metadata)
    except Exception as e:
        print(f"Primary analysis error: {e}")
        try:
            if ANTHROPIC_API_KEY:
                return analyze_call_anthropic(transcript, metadata)
            elif AZURE_OPENAI_KEY:
                return analyze_call_azure(transcript, metadata)
        except Exception as e2:
            print(f"Fallback analysis error: {e2}")
    return {"call_reason": "other", "summary": "Analysis unavailable", "sentiment": "neutral"}


# ============================================================================
# VAPI Webhook
# ============================================================================

@app.route("/vapi/webhook", methods=["POST"])
def vapi_webhook():
    data = request.json or {}
    message_type = data.get("message", {}).get("type", "unknown")
    call = data.get("message", {}).get("call", {})
    call_id = call.get("id", f"unknown-{int(time.time())}")

    # Store raw event
    event_file = CALLS_DIR / f"{call_id}_{message_type}.json"
    event_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"📞 Webhook: type={message_type}, call_id={call_id}")
    if message_type == "function-call":
        print(f"   Function call data: {json.dumps(data.get('message', {}).get('functionCall', {}))[:500]}")

    if message_type == "end-of-call-report":
        transcript = data.get("message", {}).get("transcript", "")
        msg = data.get("message", {})
        started = call.get("startedAt", "") or msg.get("startedAt", "") or call.get("createdAt", "")
        ended = call.get("endedAt", "") or msg.get("endedAt", "")
        metadata = {
            "call_id": call_id,
            "type": call.get("type", "unknown"),
            "started_at": started,
            "ended_at": ended,
            "ended_reason": call.get("endedReason", "") or msg.get("endedReason", ""),
            "duration": msg.get("durationSeconds", 0),
            "recording_url": msg.get("recordingUrl", ""),
            "cost": msg.get("cost", 0),
            "transcript": transcript,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Analyze
        if transcript and len(transcript.strip()) > 10:
            try:
                analysis = analyze_call(transcript, metadata)
                metadata["analysis"] = analysis
            except Exception as e:
                metadata["analysis_error"] = str(e)

        # Save locally
        call_file = CALLS_DIR / f"{call_id}.json"
        call_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        # Add to in-memory cache
        with _calls_lock:
            existing_ids = {c.get("call_id") for c in _calls_cache}
            if call_id not in existing_ids:
                _calls_cache.insert(0, metadata)

        # Background: index into RAG + push to GitHub
        def _background_tasks():
            try:
                if transcript and len(transcript.strip()) > 10:
                    a = metadata.get("analysis", {})
                    name_part = f", Name: {a['caller_name']}" if a.get("caller_name") else ""
                    domains = ", ".join(a.get("domains_discussed", [])) if a.get("domains_discussed") else "general"
                    # Index call data into SEPARATE calls collection (not the project KB)
                    memory_entry = (
                        f"Call ({metadata.get('type','?')}, {metadata.get('started_at','')[:10]}): "
                        f"{a.get('summary', 'No summary')} "
                        f"[Reason: {a.get('call_reason','?')}, Sentiment: {a.get('sentiment','?')}"
                        f"{name_part}, Domains: {domains}]"
                    )
                    _index_to_calls_collection(memory_entry)
                    if transcript and len(transcript.strip()) > 50:
                        _index_to_calls_collection(f"Transcript ({call_id[:12]}): {transcript[:2000]}")
                    print(f"🧠 Indexed call {call_id[:12]} to calls collection")
            except Exception as e:
                print(f"❌ Index failed: {e}")

            try:
                push_call_to_github(metadata)
            except Exception as e:
                print(f"❌ GitHub push failed: {e}")

        threading.Thread(target=_background_tasks, daemon=True).start()

    elif message_type == "function-call":
        func_call = data.get("message", {}).get("functionCall", {})
        function_name = func_call.get("name", "")
        params = func_call.get("parameters", {})
        print(f"   Parsed: name={function_name}, params={params}")

        if function_name in ("soul_query", "search_projects", "webhook_search"):
            query = params.get("query", "")
            if query:
                print(f"   🔍 soul_query_fast: {query[:100]}")
                # Use fast direct Qdrant retrieval (~1-2s) — VAPI times out after ~3s
                answer = soul_query_fast(query)
                # VAPI requires single-line strings — strip newlines and markdown
                answer = re.sub(r'[#*_~`|]', '', answer)
                answer = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', answer)
                answer = answer.replace('\n', ' ').replace('\r', ' ')
                answer = re.sub(r' {2,}', ' ', answer).strip()
                if len(answer) > 2000:
                    answer = answer[:2000]
                print(f"   ✅ Result: {len(answer)} chars")
                return jsonify({"results": [{"toolCallId": func_call.get("id", ""), "result": answer}]})
            return jsonify({"results": [{"toolCallId": func_call.get("id", ""), "result": "Please provide a question."}]})

    return jsonify({"status": "ok"})


# ============================================================================
# API — Health, Query, Chat, Index, Calls
# ============================================================================

@app.route("/api/health", methods=["GET"])
def health():
    agent = get_soul_agent()
    with _calls_lock:
        calls_count = len(_calls_cache)
    # Get calls collection point count from Qdrant
    calls_collection_count = 0
    calls_collection_status = "unknown"
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5)
        info = qc.get_collection("fraunhofer_cma_calls")
        calls_collection_count = info.points_count or 0
        calls_collection_status = "ready"
    except Exception as e:
        calls_collection_status = f"error: {str(e)[:40]}"

    return jsonify({
        "status": "healthy",
        "agent": "Lisa",
        "soul_agent": "ready" if agent else "unavailable",
        "soul_collection": QDRANT_COLLECTION,
        "soul_rag_mode": agent.mode if agent else "unknown",
        "calls_cached": calls_count,
        "calls_collection": "fraunhofer_cma_calls",
        "calls_indexed": calls_collection_count,
        "calls_collection_status": calls_collection_status,
        "github_repo": GITHUB_REPO,
    })


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.json or {}
    question = data.get("question", data.get("query", ""))
    mode = data.get("mode", "auto")
    if not question:
        return jsonify({"error": "Provide a 'question' field"}), 400
    return jsonify(soul_query(question, mode=mode))


@app.route("/api/chat/reset", methods=["POST"])
def api_chat_reset():
    agent = get_soul_agent()
    if agent and hasattr(agent, '_history'):
        agent._history.clear()
    return jsonify({"status": "ok"})


@app.route("/api/index", methods=["POST"])
def api_index():
    data = request.json or {}
    texts = data.get("texts", [])
    if isinstance(texts, str):
        texts = [texts]
    indexed = 0
    for text in texts:
        if text and len(text.strip()) > 5:
            soul_remember(text)
            indexed += 1
    return jsonify({"indexed": indexed, "total": len(texts)})


@app.route("/api/index-calls", methods=["POST"])
def api_index_calls():
    """Index text into the calls history collection (separate from projects)."""
    data = request.json or {}
    texts = data.get("texts", [])
    if isinstance(texts, str):
        texts = [texts]
    indexed = 0
    for text in texts:
        if text and len(text.strip()) > 5:
            _index_to_calls_collection(text)
            indexed += 1
    return jsonify({"indexed": indexed, "total": len(texts)})


@app.route("/api/calls", methods=["GET"])
def api_calls():
    with _calls_lock:
        if _calls_cache:
            return jsonify(_calls_cache[:50])

    calls = []
    for f in sorted(CALLS_DIR.glob("*.json"), reverse=True):
        if "_" in f.stem:
            continue
        try:
            call = json.loads(f.read_text())
            calls.append(call)
        except:
            pass
    return jsonify(calls[:50])


# ============================================================================
# Chat, Greeting, Clear, TTS, Avatar Token
# ============================================================================

def _build_chat_system_prompt(response_style="normal"):
    """Build the full system prompt with response style modifiers."""
    system_prompt = LISA_SYSTEM_PROMPT

    if response_style == "brief":
        system_prompt += (
            "\n\n## RESPONSE LENGTH INSTRUCTION\n"
            "Keep your responses SHORT and CONCISE — 1-3 sentences max. "
            "Get to the point quickly. No lengthy explanations unless they specifically ask for more detail."
        )
    elif response_style == "verbose":
        system_prompt += (
            "\n\n## RESPONSE LENGTH INSTRUCTION\n"
            "Give DETAILED, thorough responses. Explain projects fully. "
            "Provide extra context about TRL levels, partners, and outcomes. Be comprehensive in your answers."
        )

    return system_prompt


def _strip_markdown(text):
    """Strip markdown formatting for spoken output."""
    return re.sub(r'[*_~`#]', '', text).strip()


def _chat_with_azure(messages):
    """Call Azure OpenAI chat completions."""
    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
        f"/chat/completions?api-version=2025-01-01-preview"
    )
    resp = req.post(
        url,
        headers={"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"},
        json={"messages": messages, "max_tokens": 500, "temperature": 0.7},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _chat_with_anthropic(messages):
    """Call Anthropic Claude as fallback."""
    system_text = ""
    anthropic_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            anthropic_msgs.append({"role": m["role"], "content": m["content"]})
    resp = req.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "system": system_text.strip(),
            "messages": anthropic_msgs,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Conversational chat endpoint with RAG context injection."""
    data = request.json or {}
    user_message = data.get("message", "").strip()
    response_style = data.get("response_style", "normal")
    project_context = data.get("project_context", "") or ""
    project_id = data.get("project_id")

    if not user_message:
        return jsonify({"error": "Provide a 'message' field"}), 400

    # Use soul_query — routes automatically between RAG and RLM, also searches calls
    # If a specific project is focused, search for that project explicitly
    soul_result = {}
    route = "LLM only"
    try:
        search_query = (project_context + " " + user_message).strip() if project_context else user_message
        soul_result = soul_query(search_query, mode="auto")
        route = soul_result.get("route", "FOCUSED")
        print(f"💡 soul_query route={route}, ms={soul_result.get('total_ms',0):.0f}")
    except Exception as e:
        print(f"⚠️ soul_query error: {e}")

    with _chat_lock:
        _chat_history.append({"role": "user", "content": user_message})
        history_snapshot = list(_chat_history)

    system_prompt = _build_chat_system_prompt(response_style)
    if project_context:
        system_prompt += (
            f"\n\n## CURRENT PROJECT FOCUS\n"
            f"The user is asking about this specific project:\n{project_context}\n"
            f"Focus your answers on this project."
        )
    soul_answer = soul_result.get("answer", "") or ""
    if soul_answer and len(soul_answer.strip()) > 20 and "not available" not in soul_answer.lower():
        system_prompt += (
            "\n\n## RETRIEVED KNOWLEDGE\n"
            "Use the following to answer. Only reference what appears here. Do NOT invent.\n\n"
            + soul_answer
        )

    messages = [{"role": "system", "content": system_prompt}] + history_snapshot

    assistant_message = None
    try:
        if AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT:
            assistant_message = _chat_with_azure(messages)
        elif ANTHROPIC_API_KEY:
            assistant_message = _chat_with_anthropic(messages)
        else:
            return jsonify({"error": "No LLM API configured (Azure OpenAI or Anthropic)"}), 500
    except Exception as e:
        print(f"Primary chat error: {e}")
        try:
            if ANTHROPIC_API_KEY and assistant_message is None:
                assistant_message = _chat_with_anthropic(messages)
            elif AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT and assistant_message is None:
                assistant_message = _chat_with_azure(messages)
        except Exception as e2:
            print(f"Fallback chat error: {e2}")
            return jsonify({"error": f"Chat failed: {e2}"}), 500

    if not assistant_message:
        return jsonify({"error": "No response from LLM"}), 500

    with _chat_lock:
        _chat_history.append({"role": "assistant", "content": assistant_message})

    # Map soul route → user-visible label
    if route == "EXHAUSTIVE":
        retrieval = "RLM"
    elif route in ("FOCUSED", "RAG"):
        retrieval = "RAG"
    elif route == "CALLS":
        retrieval = "Call History"
    elif soul_answer and len(soul_answer.strip()) > 20:
        retrieval = "RAG"
    else:
        retrieval = "LLM only"

    rlm_meta = soul_result.get("rlm_meta") or {}
    return jsonify({
        "response": assistant_message,
        "spoken": _strip_markdown(assistant_message),
        "retrieval": retrieval,
        "route": route,
        "rag_used": route not in ("EXHAUSTIVE", None) and bool(soul_answer),
        "rlm_used": route == "EXHAUSTIVE",
        "calls_used": bool(soul_result.get("calls_context")),
        "rlm_chunks": rlm_meta.get("chunks_processed"),
        "elapsed_ms": soul_result.get("total_ms"),
    })


@app.route("/api/greeting", methods=["GET"])
def api_greeting():
    """Return Lisa's opening greeting."""
    with _chat_lock:
        _chat_history.append({"role": "assistant", "content": LISA_GREETING_MESSAGE})
    return jsonify({
        "response": LISA_GREETING_MESSAGE,
        "spoken": LISA_GREETING_MESSAGE,
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear conversation history."""
    with _chat_lock:
        _chat_history.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/tts/pcm", methods=["POST"])
def api_tts_pcm():
    """Generate raw PCM 24kHz 16-bit mono audio via Azure Speech TTS."""
    data = request.json or {}
    text = data.get("text", "").strip()
    voice = data.get("voice", "en-US-JennyNeural")
    speed = float(data.get("speed", 1.0))

    if not text:
        return jsonify({"error": "Provide a 'text' field"}), 400

    if not AZURE_SPEECH_KEY:
        return jsonify({"error": "Azure Speech key not configured"}), 500

    rate_pct = int((speed - 1.0) * 100)
    rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    ssml = (
        f'<speak version="1.0" xml:lang="en-US">'
        f'<voice name="{voice}">'
        f'<prosody rate="{rate_str}">{text}</prosody>'
        f'</voice></speak>'
    )

    try:
        resp = req.post(
            f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
            headers={
                "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
            },
            data=ssml.encode("utf-8"),
            timeout=15,
        )
        if resp.status_code != 200:
            return jsonify({"error": f"Azure Speech error {resp.status_code}: {resp.text[:200]}"}), 502

        audio_b64 = base64.b64encode(resp.content).decode()
        return jsonify({
            "audio": audio_b64,
            "format": "pcm-24khz-16bit-mono",
            "bytes": len(resp.content),
        })
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 500


@app.route("/api/avatar/token", methods=["GET"])
def api_avatar_token():
    """Create a LiveAvatar session token for the frontend."""
    avatar_id = request.args.get("avatar_id", "513fd1b7-7ef9-466d-9af2-344e51eeb833")  # Ann Therapist

    if not LIVEAVATAR_API_KEY:
        return jsonify({"error": "LiveAvatar API key not configured"}), 500

    try:
        resp = req.post(
            "https://api.liveavatar.com/v1/sessions/token",
            headers={"x-api-key": LIVEAVATAR_API_KEY, "Content-Type": "application/json"},
            json={"mode": "LITE", "avatar_id": avatar_id},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 1000:
            return jsonify({"error": data.get("message", "LiveAvatar token error")}), 502

        return jsonify({
            "session_token": data["data"]["session_token"],
            "session_id": data["data"]["session_id"],
        })
    except Exception as e:
        return jsonify({"error": f"Avatar token failed: {e}"}), 500


# ============================================================================
# Startup
# ============================================================================

def startup():
    """Run on app startup: load calls from GitHub."""
    print("🚀 Running startup tasks...")

    def _load():
        try:
            load_calls_from_github()
        except Exception as e:
            print(f"❌ Startup call load error: {e}")

    threading.Thread(target=_load, daemon=True).start()


startup()


@app.route("/api/query-fast", methods=["POST"])
def api_query_fast():
    """Test endpoint for soul_query_fast."""
    data = request.json or {}
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Provide a 'question' field"}), 400
    import time as _t
    start = _t.time()
    answer = soul_query_fast(question)
    elapsed = _t.time() - start
    return jsonify({"answer": answer, "elapsed_ms": int(elapsed * 1000)})


# ============================================================================
# LiveKit Webhook — mirrors VAPI webhook, feeds same dashboard + Qdrant
# ============================================================================

def _verify_livekit_signature(body: bytes, auth_header: str, api_secret: str) -> bool:
    """Verify LiveKit webhook signature (JWT signed with API secret)."""
    try:
        import hmac as _hmac, hashlib as _hashlib, base64 as _b64
        parts = auth_header.split(".")
        if len(parts) != 3:
            return False
        data = f"{parts[0]}.{parts[1]}"
        sig = _b64.urlsafe_b64decode(parts[2] + "==")
        expected = _hmac.new(api_secret.encode(), data.encode(), _hashlib.sha256).digest()
        return _hmac.compare_digest(sig, expected)
    except Exception:
        return False



@app.route("/api/index-project", methods=["POST"])
def api_index_project():
    """Re-index a specific project file from GitHub into Qdrant with proper embeddings."""
    data = request.json or {}
    filename = data.get("filename", "")
    if not filename:
        return jsonify({"error": "Provide 'filename' (e.g. '56-aim_hairpin_busbar_laser_welding.md')"}), 400

    try:
        import hashlib
        from soul_engine.rag_memory import _embed_azure

        # Fetch from GitHub
        url = f"https://raw.githubusercontent.com/menonpg/fraunhofer-cma-projects/main/docs/projects/{filename}"
        headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
        resp = _requests_get(url, headers=headers)
        if not resp:
            return jsonify({"error": f"Could not fetch {filename} from GitHub"}), 404
        text = resp

        # Parse into chunks
        import re as _re
        title_match = _re.search(r'^#\s+(.+)', text, _re.MULTILINE)
        title = title_match.group(1).strip() if title_match else filename.replace(".md","")

        chunks = []
        # Full doc chunk
        chunks.append(f"Project: {title}\n\n{text[:3000]}")
        # Section chunks
        for section in _re.split(r'\n##\s+', text)[1:]:
            lines = section.split("\n", 1)
            sec_title = lines[0].strip()
            sec_body = lines[1].strip() if len(lines) > 1 else ""
            if len(sec_body) > 20:
                chunks.append(f"Project: {title}\nSection: {sec_title}\n\n{sec_body}"[:2000])

        # Embed and upsert
        agent = get_soul_agent()
        if not agent or not hasattr(agent, '_rag') or not agent._rag:
            return jsonify({"error": "RAG not initialized"}), 500

        rag = agent._rag
        indexed = 0
        for chunk in chunks:
            try:
                if rag.mode == "qdrant" and rag._embed_provider:
                    vec = rag._embed([chunk])[0]
                    chunk_id = abs(int(hashlib.md5(chunk.encode()).hexdigest()[:16], 16)) % (2**63)
                    rag._qdrant.upsert(rag.collection, [{
                        "id": chunk_id,
                        "vector": vec,
                        "payload": {"text": chunk, "project": title, "filename": filename}
                    }])
                    indexed += 1
                else:
                    rag.append(chunk)
                    indexed += 1
            except Exception as e:
                print(f"  chunk error: {e}")

        print(f"✅ Indexed {filename}: {indexed}/{len(chunks)} chunks")
        return jsonify({"status": "ok", "filename": filename, "project": title, "chunks_indexed": indexed})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _requests_get(url, headers=None):
    """Simple GET via urllib."""
    import urllib.request
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read().decode()
    except Exception as e:
        print(f"GET {url} failed: {e}")
        return None

@app.route("/livekit/webhook", methods=["POST"])
def livekit_webhook():
    """
    LiveKit sends webhook events when rooms/participants change.
    We listen for:
      - room_finished  → treat as end-of-call, save + analyze + index
      - track_published → participant started publishing (call started)
    """
    LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "LRJQSmai8IIC3l8DSRLXh8voWgtoAUnQkRPZkbcJy1K")

    raw_body = request.get_data()
    auth_header = request.headers.get("Authorization", "")

    # Log all incoming events
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    event = data.get("event", "unknown")
    room = data.get("room", {})
    room_name = room.get("name", "unknown")
    room_sid = room.get("sid", "")

    print(f"📡 LiveKit webhook: event={event}, room={room_name}")

    if event == "room_finished":
        # Room ended — collect transcript from egress or participants
        # LiveKit doesn't send a transcript directly; we build one from participant data
        num_participants = room.get("numParticipants", 0)
        duration_secs = room.get("duration", 0)
        created_at = room.get("creationTime", "")
        egress_list = data.get("egressInfo", [])

        # Build a call record in the same format as VAPI calls
        call_id = room_sid or f"lk-{room_name}-{int(time.time())}"
        started_at = datetime.fromtimestamp(int(created_at), tz=timezone.utc).isoformat() if created_at else datetime.now(timezone.utc).isoformat()

        # Transcript: LiveKit doesn't include it in room_finished by default.
        # We store what we have and note the source.
        transcript = data.get("transcript", "")  # future: if STT transcript is forwarded

        metadata = {
            "call_id": call_id,
            "source": "livekit",
            "type": "web",
            "room_name": room_name,
            "started_at": started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "duration": duration_secs,
            "num_participants": num_participants,
            "ended_reason": "room_finished",
            "transcript": transcript,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Analyze if we have a transcript
        if transcript and len(transcript.strip()) > 10:
            try:
                analysis = analyze_call(transcript, metadata)
                metadata["analysis"] = analysis
            except Exception as e:
                metadata["analysis_error"] = str(e)
        else:
            # No transcript yet — store minimal record so dashboard shows the call
            metadata["analysis"] = {
                "summary": f"LiveKit voice session in room {room_name} ({duration_secs}s)",
                "sentiment": "neutral",
                "call_reason": "voice_demo",
                "domains_discussed": [],
            }

        # Save locally
        call_file = CALLS_DIR / f"{call_id}.json"
        call_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

        # Add to in-memory cache → shows up in /api/calls → shows on dashboard
        with _calls_lock:
            existing_ids = {c.get("call_id") for c in _calls_cache}
            if call_id not in existing_ids:
                _calls_cache.insert(0, metadata)

        print(f"✅ LiveKit call saved: {call_id}, duration={duration_secs}s")

        # Background: index + push to GitHub
        def _lk_background():
            try:
                a = metadata.get("analysis", {})
                summary = a.get("summary", f"LiveKit session {room_name}")
                memory_entry = (
                    f"LiveKit Call (web, {metadata['started_at'][:10]}): {summary} "
                    f"[Room: {room_name}, Duration: {duration_secs}s]"
                )
                _index_to_calls_collection(memory_entry)
                if transcript and len(transcript.strip()) > 50:
                    _index_to_calls_collection(f"Transcript ({call_id[:12]}): {transcript[:2000]}")
                print(f"🧠 Indexed LiveKit call {call_id[:12]}")
            except Exception as e:
                print(f"❌ LiveKit index failed: {e}")
            try:
                push_call_to_github(metadata)
            except Exception as e:
                print(f"❌ LiveKit GitHub push failed: {e}")

        threading.Thread(target=_lk_background, daemon=True).start()

    elif event in ("participant_joined", "participant_left", "track_published"):
        participant = data.get("participant", {})
        identity = participant.get("identity", "")
        print(f"   participant {identity} — {event}")
        # Not stored as a call, just logged

    return jsonify({"status": "ok"})


@app.route("/livekit/transcript", methods=["POST"])
def livekit_transcript():
    """
    Called by the Lisa agent at end of session to POST the full transcript.
    Agent sends: { call_id, room_name, transcript, duration, started_at }
    This updates the existing call record with real transcript + re-analyzes.
    """
    data = request.json or {}
    call_id = data.get("call_id", "")
    room_name = data.get("room_name", "")
    transcript = data.get("transcript", "")
    duration = data.get("duration", 0)
    started_at = data.get("started_at", datetime.now(timezone.utc).isoformat())

    if not call_id:
        call_id = f"lk-{room_name}-{int(time.time())}"

    print(f"📝 LiveKit transcript received: {call_id}, {len(transcript)} chars")

    metadata = {
        "call_id": call_id,
        "source": "livekit",
        "type": "web",
        "room_name": room_name,
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration": duration,
        "transcript": transcript,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if transcript and len(transcript.strip()) > 10:
        try:
            metadata["analysis"] = analyze_call(transcript, metadata)
        except Exception as e:
            metadata["analysis_error"] = str(e)
            metadata["analysis"] = {"summary": "Analysis failed", "sentiment": "neutral"}
    else:
        metadata["analysis"] = {"summary": f"LiveKit session ({duration}s, no transcript)", "sentiment": "neutral"}

    # Save / update
    call_file = CALLS_DIR / f"{call_id}.json"
    call_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    with _calls_lock:
        # Replace or insert
        existing = [c for c in _calls_cache if c.get("call_id") != call_id]
        existing.insert(0, metadata)
        _calls_cache[:] = existing

    def _bg():
        try:
            a = metadata.get("analysis", {})
            _index_to_calls_collection(
                f"LiveKit Call (web, {started_at[:10]}): {a.get('summary','?')} [Room: {room_name}]"
            )
            if transcript and len(transcript) > 50:
                _index_to_calls_collection(f"Transcript ({call_id[:12]}): {transcript[:2000]}")
            push_call_to_github(metadata)
        except Exception as e:
            print(f"❌ Background task failed: {e}")

    threading.Thread(target=_bg, daemon=True).start()
    return jsonify({"status": "ok", "call_id": call_id})


@app.route("/vapi/function", methods=["POST"])
def vapi_function():
    """Dedicated endpoint for VAPI function/tool calls — handles all possible formats."""
    data = request.json or {}
    print(f"🔧 /vapi/function raw keys: {list(data.keys())}")

    # Extract function call from various possible formats
    msg = data.get("message", {})
    func_call = msg.get("functionCall", data.get("functionCall", {}))

    if not func_call:
        tc = msg.get("toolCalls", data.get("toolCalls", []))
        if tc:
            func_call = tc[0].get("function", {})
            func_call["id"] = tc[0].get("id", "")

    function_name = func_call.get("name", "")
    params = func_call.get("parameters", func_call.get("arguments", {}))
    if isinstance(params, str):
        import json as _json
        params = _json.loads(params)
    tool_call_id = func_call.get("id", "")

    print(f"🔧 /vapi/function: name={function_name}, query={params.get('query','')[:80]}, id={tool_call_id}")

    query = params.get("query", "")
    if query:
        answer = soul_query_fast(query)
        answer = re.sub(r'[#*_~`|]', '', answer)
        answer = answer.replace('\n', ' ').replace('\r', ' ')
        answer = re.sub(r' {2,}', ' ', answer).strip()
        if len(answer) > 2000:
            answer = answer[:2000]
        print(f"🔧 Result: {len(answer)} chars")
        return jsonify({"results": [{"toolCallId": tool_call_id, "result": answer}]})

    return jsonify({"results": [{"toolCallId": tool_call_id, "result": "No query provided"}]})


# --- Main ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🔬 Lisa — Fraunhofer CMA Webhook Server starting on port {port}")
    print(f"   Collection: {QDRANT_COLLECTION}")
    app.run(host="0.0.0.0", port=port)

# ============================================================================
# LiveKit Room + Dispatch (called by browser instead of CF worker)
# ============================================================================

@app.route("/livekit/start-session", methods=["GET", "POST"])
def livekit_start_session():
    """
    Browser calls this to get a LiveKit token + trigger agent dispatch.
    Returns: { token, url, room, identity }
    """
    import hmac as _hmac
    import hashlib as _hashlib
    import base64 as _b64

    LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "APIdhcScxEydPYB")
    LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "LRJQSmai8IIC3l8DSRLXh8voWgtoAUnQkRPZkbcJy1K")
    LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://monica-shg8b37e.livekit.cloud")
    LIVEKIT_HOST = LIVEKIT_URL.replace("wss://", "https://")

    import uuid
    room_name = f"lisa-{int(time.time())}"
    identity = f"visitor-{uuid.uuid4().hex[:8]}"
    now = int(time.time())

    def b64url(data):
        if isinstance(data, str): data = data.encode()
        return _b64.urlsafe_b64encode(data).rstrip(b'=').decode()

    def make_jwt(payload, secret):
        h = b64url(json.dumps({"alg":"HS256","typ":"JWT"}, separators=(',',':')))
        p = b64url(json.dumps(payload, separators=(',',':')))
        msg = f"{h}.{p}"
        sig = _hmac.new(secret.encode(), msg.encode(), _hashlib.sha256).digest()
        return f"{msg}.{b64url(sig)}"

    admin_token = make_jwt({
        "iss": LIVEKIT_API_KEY, "nbf": now, "exp": now + 300,
        "video": {"room": room_name, "roomAdmin": True, "roomCreate": True},
    }, LIVEKIT_API_SECRET)

    # Create room
    try:
        req.post(f"{LIVEKIT_HOST}/twirp/livekit.RoomService/CreateRoom",
            json={"name": room_name, "empty_timeout": 600},
            headers={"Authorization": f"Bearer {admin_token}"}, timeout=8)
    except Exception as e:
        print(f"Room create error: {e}")

    # Dispatch agent
    dispatch_ok = False
    try:
        dr = req.post(f"{LIVEKIT_HOST}/twirp/livekit.AgentDispatchService/CreateDispatch",
            json={"room": room_name, "agent_name": "fraunhofer-lisa"},
            headers={"Authorization": f"Bearer {admin_token}"}, timeout=8)
        dispatch_ok = dr.ok
        print(f"Dispatch: {dr.status_code} {dr.text[:100]}")
    except Exception as e:
        print(f"Dispatch error: {e}")

    # Visitor token
    visitor_token = make_jwt({
        "iss": LIVEKIT_API_KEY, "sub": identity, "nbf": now, "exp": now + 600,
        "video": {"roomJoin": True, "room": room_name, "canPublish": True, "canSubscribe": True},
    }, LIVEKIT_API_SECRET)

    # Log session start immediately → shows on dashboard right away
    try:
        start_record = {
            "call_id": f"lk-{room_name}",
            "source": "livekit",
            "type": "web",
            "room_name": room_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "duration": 0,
            "transcript": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis": {"summary": "LiveKit voice session in progress…", "sentiment": "neutral"},
        }
        call_file = CALLS_DIR / f"lk-{room_name}.json"
        call_file.write_text(json.dumps(start_record, indent=2))
        with _calls_lock:
            existing_ids = {c.get("call_id") for c in _calls_cache}
            if f"lk-{room_name}" not in existing_ids:
                _calls_cache.insert(0, start_record)
        # Push to GitHub in background
        threading.Thread(target=push_call_to_github, args=(start_record,), daemon=True).start()
    except Exception as e:
        print(f"Could not log session start: {e}")

    return jsonify({
        "token": visitor_token,
        "url": LIVEKIT_URL,
        "room": room_name,
        "identity": identity,
        "dispatch_ok": dispatch_ok,
    })


# =============================================================================
# DE-Ω Discovery Engine API — Live showcase endpoints
# =============================================================================

def _deo_claude(prompt, max_tokens=1200):
    """Call Azure OpenAI GPT-5 for DE-Ω pipeline steps — reuses existing Lisa Azure infra."""
    return _chat_with_azure([{"role": "user", "content": prompt}])


def _deo_web_search(query, count=5):
    """Brave Search for DE-Ω context enrichment. Returns top snippets as a string."""
    if not BRAVE_API_KEY:
        return ""
    try:
        resp = req.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            params={"q": query, "count": count, "search_lang": "en"},
            timeout=8
        )
        if resp.status_code != 200:
            return ""
        results = resp.json().get("web", {}).get("results", [])
        snippets = []
        for r in results[:count]:
            title = r.get("title", "")
            desc = r.get("description", "")
            url = r.get("url", "")
            if desc:
                snippets.append(f"[{title}] {desc} ({url})")
        return "\n".join(snippets)
    except Exception as e:
        print(f"⚠️ Brave search error: {e}")
        return ""


@app.route("/deo/context", methods=["POST","OPTIONS"])
def deo_context():
    if request.method=="OPTIONS":
        r=jsonify({});r.headers["Access-Control-Allow-Origin"]="*";r.headers["Access-Control-Allow-Headers"]="Content-Type";return r
    data=request.get_json(silent=True) or {}
    query=data.get("query","")
    # 1. CMA knowledge base (Qdrant)
    qdrant_context = soul_query_fast(query) if query else ""
    hits = len([c for c in qdrant_context.split("\n\n") if len(c)>50]) if qdrant_context else 0
    # 2. Web search (Brave) for up-to-date domain knowledge
    web_context = _deo_web_search(query) if query else ""
    # Combine — clearly labelled so Module 1 knows what's CMA vs web
    combined = ""
    if qdrant_context:
        combined += "=== Fraunhofer CMA Projects (Qdrant) ===\n" + qdrant_context
    if web_context:
        combined += "\n\n=== Web Search Results ===\n" + web_context
    r=jsonify({"context": combined, "hits": hits, "web_hits": len(web_context.split("\n")) if web_context else 0})
    r.headers["Access-Control-Allow-Origin"]="*"
    return r


@app.route("/deo/module1", methods=["POST","OPTIONS"])
def deo_module1():
    if request.method=="OPTIONS":
        r=jsonify({});r.headers["Access-Control-Allow-Origin"]="*";r.headers["Access-Control-Allow-Headers"]="Content-Type";return r
    data=request.get_json(silent=True) or {}
    problem=data.get("problem","")
    context=data.get("context","")
    ctx_section=f"\n\nRelevant CMA project context:\n{context}" if context else ""
    prompt=f"""You are DE-Ω Module 1: Structured Hypothesis Generation.

The CENTRAL QUESTION you must answer is: {problem}

Use the context below ONLY to ground your hypotheses in specific facts, terminology, and real examples. The hypotheses must directly address the central question — do not drift into generic statements.

Context (CMA projects + web search):
{context if context else 'none'}

Generate 3-4 candidate hypotheses that DIRECTLY answer: {problem}

Each must satisfy:
1. Non-redundancy (not a restatement of known facts)
2. Cross-domain coupling (involves ≥2 domains)
3. Mathematical convertibility (can be formalized)
4. Falsifiability (testable prediction)
5. Direct relevance to the central question above

Respond ONLY with valid JSON, no markdown, no code fences:
{{"hypotheses":[{{"name":"short name","description":"1-2 sentence hypothesis that directly addresses the question","math_form":"brief mathematical expression","domains":["domain1","domain2"]}}]}}"""
    import json as _json
    try:
        raw=_deo_claude(prompt,1000)
        # strip any accidental markdown
        raw=raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result=_json.loads(raw)
    except Exception as e:
        result={"hypotheses":[{"name":"Parse error","description":str(e),"math_form":"","domains":[]}]}
    r=jsonify(result)
    r.headers["Access-Control-Allow-Origin"]="*"
    return r


@app.route("/deo/module2a", methods=["POST","OPTIONS"])
def deo_module2a():
    if request.method=="OPTIONS":
        r=jsonify({});r.headers["Access-Control-Allow-Origin"]="*";r.headers["Access-Control-Allow-Headers"]="Content-Type";return r
    data=request.get_json(silent=True) or {}
    problem=data.get("problem","")
    hypotheses=data.get("hypotheses",[])
    hyp_text="\n".join([f"- {h.get('name','')}: {h.get('description','')} | Math: {h.get('math_form','')}" for h in hypotheses])
    prompt=f"""You are DE-Ω Module 2A: Structural Audit.

Problem: {problem}

Hypotheses to audit:
{hyp_text}

For each hypothesis, assess:
- Mathematical coherence (is the math form well-defined?)
- Internal consistency (no self-contradiction)
- Falsifiability (can it be disproven?)
- Proof-path traceability (is it traceable to assumptions?)

Respond ONLY with valid JSON, no markdown, no code fences:
{{"results":[{{"name":"hypothesis name","pass":true,"reason":"brief explanation of pass or rejection reason"}}]}}"""
    import json as _json
    try:
        raw=_deo_claude(prompt,1500)
        raw=raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result=_json.loads(raw)
    except Exception as e:
        result={"results":[{"name":"error","pass":False,"reason":str(e)}]}
    r=jsonify(result)
    r.headers["Access-Control-Allow-Origin"]="*"
    return r


@app.route("/deo/module2b", methods=["POST","OPTIONS"])
def deo_module2b():
    if request.method=="OPTIONS":
        r=jsonify({});r.headers["Access-Control-Allow-Origin"]="*";r.headers["Access-Control-Allow-Headers"]="Content-Type";return r
    data=request.get_json(silent=True) or {}
    problem=data.get("problem","")
    hypotheses=data.get("hypotheses",[])
    context=data.get("context","")
    ctx_section=f"\n\nCMA knowledge base context:\n{context}" if context else ""
    hyp_text="\n".join([f"- {h}" for h in hypotheses]) if hypotheses else "none"
    prompt=f"""You are DE-Ω Module 2B: Domain Constraint Validation Layer (DCVL).

Problem: {problem}{ctx_section}

Hypotheses (passed structural audit):
{hyp_text}

Identify the relevant domain registries for this problem (e.g. biological, chemical, engineering, mathematical, etc.)
Then for each hypothesis, apply domain-specific hard-gate rules.
A hypothesis may be structurally valid but domain-incomplete — note if strengthening was needed.

Respond ONLY with valid JSON, no markdown, no code fences:
{{"domain_registries":["domain1","domain2"],"results":[{{"name":"hypothesis name","promoted":true,"strengthened":false,"dcvl_notes":"brief domain validation result"}}]}}"""
    import json as _json
    try:
        raw=_deo_claude(prompt,900)
        raw=raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result=_json.loads(raw)
    except Exception as e:
        result={"domain_registries":[],"results":[{"name":"error","promoted":False,"strengthened":False,"dcvl_notes":str(e)}]}
    r=jsonify(result)
    r.headers["Access-Control-Allow-Origin"]="*"
    return r


@app.route("/deo/output", methods=["POST","OPTIONS"])
def deo_output():
    if request.method=="OPTIONS":
        r=jsonify({});r.headers["Access-Control-Allow-Origin"]="*";r.headers["Access-Control-Allow-Headers"]="Content-Type";return r
    data=request.get_json(silent=True) or {}
    problem=data.get("problem","")
    promoted=data.get("promoted",[])
    context=data.get("context","")
    prom_text="\n".join([f"- {p.get('name','')}: {p.get('dcvl_notes','')}" for p in promoted]) if promoted else "none"
    prompt=f"""You are DE-Ω Final Output: Rank promoted hypotheses and produce experimental designs.

CENTRAL QUESTION: {problem}

Promoted hypotheses:
{prom_text}

Each ranked output MUST:
1. Directly address the central question: {problem}
2. Be concrete and specific — not generic
3. Include falsification criterion tied to the original question

Rank by: (1) direct relevance to question, (2) impact potential, (3) experimental tractability.

Respond ONLY with valid JSON, no markdown, no code fences:
{{"ranked":[{{"name":"name","priority":"high|med|low","summary":"1 sentence that directly answers how this addresses the question","experimental_path":"concrete steps + falsification criterion specific to this question"}}],"audit_summary":"2-3 sentences summarizing what DE-Ω found in answer to: {problem}"}}"""
    import json as _json
    try:
        raw=_deo_claude(prompt,1000)
        raw=raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result=_json.loads(raw)
    except Exception as e:
        result={"ranked":[],"audit_summary":str(e)}
    r=jsonify(result)
    r.headers["Access-Control-Allow-Origin"]="*"
    return r


# =============================================================================
# CMA Proposal Generator
# =============================================================================

@app.route("/deo/proposal", methods=["POST","OPTIONS"])
def deo_proposal():
    if request.method == "OPTIONS":
        r = jsonify({})
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return r
    data = request.get_json(silent=True) or {}
    problem   = data.get("problem", "")
    industry  = data.get("industry", "")
    tone      = data.get("tone", "hybrid")
    customer  = data.get("customer", "the customer")
    timeline  = data.get("timeline", "6 months")
    context   = data.get("context", "")

    cma_ctx, web_ctx = "", ""
    if "=== Web Search Results ===" in context:
        parts = context.split("=== Web Search Results ===")
        cma_ctx = parts[0].replace("=== Fraunhofer CMA Projects (Qdrant) ===","").strip()
        web_ctx = parts[1].strip() if len(parts)>1 else ""
    else:
        cma_ctx = context

    industry_str = f" in the {industry} domain" if industry else ""
    tone_map = {
        "technical": "Include detailed technical methodology, Mermaid architecture diagrams, and specific tooling.",
        "executive": "Focus on business value and outcomes. Keep technical depth minimal.",
        "hybrid": "Open with executive summary (business value). Then add technical depth with architecture diagram."
    }
    tone_inst = tone_map.get(tone, tone_map["hybrid"])

    tick = '```'
    # Extract project names from CMA context for explicit grounding
    import re as _re
    proj_names = _re.findall(r'(?:TextSentry|PACT|DAVE|[A-Z]{2,}[0-9]*(?:\s+[0-9.]+)?|Project\s+[A-Z0-9]+)', cma_ctx) if cma_ctx else []
    proj_names_str = ", ".join(set(proj_names[:8])) if proj_names else "CMA project portfolio"

    prompt = f"""You are a senior Fraunhofer CMA proposal writer. Generate a complete, professional project proposal.

CUSTOMER: {customer}
PROBLEM: {problem}
DOMAIN: {industry_str if industry_str else 'auto-detect'}
TIMELINE: {timeline}
TONE: {tone_inst}

SPECIFIC CMA PROJECTS TO REFERENCE (cite these by name in the proposal):
{proj_names_str}

FULL CMA KNOWLEDGE BASE CONTEXT:
{cma_ctx if cma_ctx else 'Draw on CMA expertise in AI, manufacturing, healthcare, cybersecurity, energy, autonomous systems.'}

WEB RESEARCH (use for current state-of-the-art context):
{web_ctx if web_ctx else 'N/A'}

Write a complete proposal in Markdown with these sections:
1. Executive Summary (3-4 sentences: problem, approach, outcome, why CMA)
2. Problem Statement & Context (detail the challenge, root causes, current gaps)
3. Proposed Approach (methodology grounded in CMA capabilities; cite real project names from context where relevant)
4. Technical Architecture (include a Mermaid flowchart or sequence diagram inside {tick}mermaid...{tick} block)
5. Work Packages & Timeline (Markdown table with phases, deliverables, milestones for {timeline})
6. Expected Outcomes & KPIs (quantified where possible)
7. Why Fraunhofer CMA (track record, relevant past work, unique position)
8. Investment & Next Steps (frame as TBD pending scoping; suggest discovery workshop)

Rules:
- Be specific and concrete — not marketing fluff
- Do NOT invent fake project names; use real ones from context or say "CMA project portfolio"
- Mermaid diagram must be valid (flowchart LR or sequenceDiagram)
- Use Markdown tables for timeline
- Length: comprehensive (800-1200 words body)

Output ONLY Markdown. Start with proposal title as H1."""

    try:
        url = (
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
            f"/chat/completions?api-version=2025-01-01-preview"
        )
        resp = req.post(
            url,
            headers={"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 3000, "temperature": 0.7},
            timeout=60,
        )
        resp.raise_for_status()
        md = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        md = f"# Error\n\n{e}"

    # Log to GitHub async
    try:
        import threading as _th
        _th.Thread(target=log_cma_tool, args=("proposal",
            {"problem": problem, "customer": customer, "industry": industry, "tone": tone, "timeline": timeline},
            md[:200],
            md
        ), daemon=True).start()
    except Exception as _le:
        print(f"log err: {_le}")

    r = jsonify({"markdown": md})
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r


@app.route("/deo/ideas", methods=["POST","OPTIONS"])
def deo_ideas():
    if request.method == "OPTIONS":
        r = jsonify({})
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return r
    data = request.get_json(silent=True) or {}
    problem  = data.get("problem", "")
    industry = data.get("industry", "")
    context  = data.get("context", "")

    cma_ctx = context.split("=== Web Search Results ===")[0].replace("=== Fraunhofer CMA Projects (Qdrant) ===","").strip() if context else ""

    prompt = f"""You are a business development strategist for Fraunhofer CMA, a leading applied research institute.

A CMA team has developed a solution for the following problem:
PROBLEM / PROPOSAL: {problem}
INDUSTRY: {industry or 'as described in the problem'}

CMA KNOWLEDGE BASE CONTEXT:
{cma_ctx if cma_ctx else 'Use general CMA expertise in AI, manufacturing, healthcare, cybersecurity, energy, autonomous systems.'}

Generate:

1. PROSPECTS — 4-5 other companies or organizations who have the same or very similar problem and would benefit from CMA's approach. Be specific (real company types, sectors, sizes).

2. DOMAINS — 4-5 other industry domains where the same core approach/technology could be applied with adaptation.

3. OPPORTUNITY MATRIX — 6-8 rows combining the most compelling prospect + domain combinations, ranked by CMA fit.

Respond ONLY with valid JSON, no markdown, no code fences:
{{
  "prospects": [
    {{"name": "company/org type or real name", "description": "why they have this problem", "type": "Industrial|Healthcare|Government|SME", "size": "Enterprise|Mid-market|SME", "domain": "their industry", "why": "specific reason CMA's solution fits them"}}
  ],
  "domains": [
    {{"domain": "domain name", "application": "how the core approach applies here", "category": "Manufacturing|Healthcare|Defense|Energy|Logistics|etc", "adaptation": "what would need to change from the original approach"}}
  ],
  "matrix": [
    {{"target": "prospect or org type", "domain": "domain", "fit": "High|Medium", "rationale": "one sentence"}}
  ]
}}"""

    try:
        url = (
            f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}"
            f"/chat/completions?api-version=2025-01-01-preview"
        )
        resp = req.post(
            url,
            headers={"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"},
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 2000, "temperature": 0.8},
            timeout=45,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        import json as _json
        result = _json.loads(raw)
    except Exception as e:
        result = {"prospects": [], "domains": [], "matrix": [], "error": str(e)}

    # Log to GitHub async
    try:
        import threading as _th
        _th.Thread(target=log_cma_tool, args=("ideas",
            {"problem": problem, "industry": industry},
            f"{len(result.get('prospects',[]))} prospects, {len(result.get('domains',[]))} domains",
            result
        ), daemon=True).start()
    except Exception as _le:
        print(f"log err: {_le}")

    r = jsonify(result)
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r


# =============================================================================
# CMA Tool Logging (proposal + ideas) — GitHub-backed
# =============================================================================

def log_cma_tool(tool, inputs, output_summary, full_output=None):
    """Save a CMA tool usage log to GitHub under cma-logs/{id}.json and update index."""
    if not GITHUB_TOKEN:
        print("⚠️ No GITHUB_TOKEN — skipping CMA log")
        return
    import uuid, datetime as _dt
    log_id = str(uuid.uuid4())
    ts = _dt.datetime.utcnow().isoformat() + "Z"
    record = {
        "id": log_id,
        "tool": tool,
        "timestamp": ts,
        "inputs": inputs,
        "output_summary": output_summary,
        "full_output": full_output,
    }
    try:
        content = json.dumps(record, indent=2, ensure_ascii=False)
        github_put_file(f"cma-logs/{log_id}.json", content,
                        f"cma-log: {tool} — {ts[:16]}")

        # Update index
        idx_path = "cma-logs/index.json"
        existing, sha = github_get_file(idx_path)
        idx = json.loads(existing) if existing else []
        idx.insert(0, {
            "id": log_id,
            "tool": tool,
            "timestamp": ts,
            "summary": output_summary[:120],
            "problem": (inputs.get("problem","") or "")[:100],
            "customer": inputs.get("customer",""),
            "industry": inputs.get("industry",""),
        })
        idx = idx[:200]  # keep last 200
        github_put_file(idx_path, json.dumps(idx, indent=2, ensure_ascii=False),
                        f"cma-logs index: {len(idx)} entries", sha)
        print(f"✅ CMA log saved: {log_id} ({tool})")
    except Exception as e:
        print(f"❌ CMA log error: {e}")


@app.route("/cma-logs", methods=["GET"])
def cma_logs_index():
    """Return the CMA logs index JSON."""
    existing, _ = github_get_file("cma-logs/index.json")
    if not existing:
        r = jsonify([])
    else:
        r = jsonify(json.loads(existing))
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r


@app.route("/cma-logs/<log_id>", methods=["GET"])
def cma_log_detail(log_id):
    """Return a single CMA log entry."""
    content, _ = github_get_file(f"cma-logs/{log_id}.json")
    if not content:
        r = jsonify({"error": "not found"}), 404
        return r
    r = jsonify(json.loads(content))
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r
