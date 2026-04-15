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
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "swedencentral")
LIVEAVATAR_API_KEY = os.environ.get("LIVEAVATAR_API_KEY", "810d1e58-289a-11f1-8d28-066a7fa2e369")

app = Flask(__name__)
CORS(app)

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
You have deep knowledge of Fraunhofer CMA's 55+ projects spanning:
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
    "technology portfolio. We have over 55 projects spanning AI, manufacturing, healthcare, "
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
        file_path = f"vapi-calls/{call_id}.json"
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
        # Direct Qdrant vector search — bypass LLM synthesis entirely
        if hasattr(agent, '_rag') and agent._rag and hasattr(agent._rag, '_qdrant') and agent._rag._qdrant:
            rag = agent._rag
            vec = rag._embed([question])[0]
            raw_results = rag._qdrant.search(rag.collection, vec, 5)
            if raw_results:
                chunks = []
                for i, r in enumerate(raw_results[:5], 1):
                    text = r.get("payload", {}).get("text", "")
                    if len(text) > 500:
                        text = text[:500] + "..."
                    chunks.append(f"Result {i}: {text}")
                return "\n\n".join(chunks)
            else:
                return "No matching projects found in the knowledge base."
        else:
            # Fallback: use retrieve() which returns a formatted string — still no LLM
            result_text = agent._rag.retrieve(question, k=5) if hasattr(agent, '_rag') else ""
            if result_text and "No relevant memories" not in result_text:
                return result_text
            return "No matching projects found in the knowledge base."
    except Exception as e:
        print(f"⚠️ soul_query_fast error: {e}")
        traceback.print_exc()
        return f"Search error: {e}"


def soul_remember(text):
    agent = get_soul_agent()
    if agent:
        agent.remember(text)


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
                    memory_entry = (
                        f"Call ({metadata.get('type','?')}, {metadata.get('started_at','')[:10]}): "
                        f"{a.get('summary', 'No summary')} "
                        f"[Reason: {a.get('call_reason','?')}, Sentiment: {a.get('sentiment','?')}"
                        f"{name_part}, Domains: {domains}]"
                    )
                    soul_remember(memory_entry)
                    soul_remember(f"Transcript ({call_id[:12]}): {transcript[:2000]}")
                    print(f"🧠 Indexed call {call_id[:12]}")
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

        if function_name == "soul_query":
            query = params.get("query", "")
            if query:
                print(f"   🔍 soul_query_fast: {query[:100]}")
                # Use fast direct Qdrant search — skip LLM synthesis
                # VAPI's own LLM will synthesize from the raw results
                answer = soul_query_fast(query)
                print(f"   ✅ Result length: {len(answer)} chars")
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
    return jsonify({
        "status": "healthy",
        "agent": "Lisa",
        "soul_agent": "ready" if agent else "unavailable",
        "soul_collection": QDRANT_COLLECTION,
        "soul_rag_mode": agent.mode if agent else "unknown",
        "calls_cached": calls_count,
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
    """Conversational chat endpoint."""
    data = request.json or {}
    user_message = data.get("message", "").strip()
    response_style = data.get("response_style", "normal")

    if not user_message:
        return jsonify({"error": "Provide a 'message' field"}), 400

    with _chat_lock:
        _chat_history.append({"role": "user", "content": user_message})
        history_snapshot = list(_chat_history)

    system_prompt = _build_chat_system_prompt(response_style)
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

    return jsonify({
        "response": assistant_message,
        "spoken": _strip_markdown(assistant_message),
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


# --- Main ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🔬 Lisa — Fraunhofer CMA Webhook Server starting on port {port}")
    print(f"   Collection: {QDRANT_COLLECTION}")
    app.run(host="0.0.0.0", port=port)
