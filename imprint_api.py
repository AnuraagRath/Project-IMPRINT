"""
Project IMPRINT — Flask API Server
Exposes 4 endpoints matching the CLI menu:
  POST /api/create     → run full distillation pipeline
  POST /api/correct    → apply correction to existing skill
  POST /api/delete     → delete skill artifact
  POST /api/chat       → chat with a skill via RAG
  GET  /api/skills     → list all available skills
  GET  /api/skill/<slug> → get skill metadata + preview
"""

import os
import json
import re
import pickle
import numpy as np
import faiss
import boto3
import botocore
import time
import random
import shutil
import threading

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# ── ENV ───────────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

MODEL_ID  = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
EMBED_ID  = "amazon.titan-embed-text-v2:0"
DATA_DIR  = Path(os.getenv("IMPRINT_DATA_DIR", "./imprint_data"))
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="frontend")
CORS(app)

# In-memory job status tracker (replace with Redis for prod)
job_status: Dict[str, Any] = {}

# ── Bedrock helpers ───────────────────────────────────────────────────────────

def bedrock_call(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    body_dict: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "anthropic_version": "bedrock-2023-05-31",
    }
    if system:
        body_dict["system"] = system

    body = json.dumps(body_dict)
    retries, max_wait = 10, 60
    for attempt in range(retries):
        try:
            response = bedrock.invoke_model(modelId=MODEL_ID, body=body)
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                wait = min((2 ** attempt) * 3 + random.random(), max_wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Bedrock call failed after retries.")


def get_embedding(text: str) -> List[float]:
    body = json.dumps({"inputText": text[:8000]})
    response = bedrock.invoke_model(modelId=EMBED_ID, body=body)
    return json.loads(response["body"].read())["embedding"]


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ── Skill storage helpers ─────────────────────────────────────────────────────

def skill_dir(slug: str) -> Path:
    d = DATA_DIR / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_meta(slug: str) -> Dict | None:
    p = skill_dir(slug) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_skills() -> List[Dict]:
    skills = []
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    skills.append({
                        "slug": meta.get("slug", d.name),
                        "name": meta.get("colleague-name", d.name),
                        "version": meta.get("version", "1.0.0"),
                        "status": meta.get("lifecycle-status", "draft"),
                        "generated": meta.get("generated", ""),
                        "correction_count": meta.get("correction-count", 0),
                        "source_count": len(meta.get("source-scope", [])),
                    })
                except Exception:
                    pass
    return sorted(skills, key=lambda x: x["generated"], reverse=True)

# ── Pipeline (runs in background thread) ─────────────────────────────────────

def _log(slug: str, msg: str):
    job_status.setdefault(slug, {"logs": [], "status": "running", "result": None})
    job_status[slug]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    print(f"[{slug}] {msg}")


def run_pipeline(slug: str, colleague_name: str, trace_folder: str):
    """Full distillation pipeline — runs in background thread."""
    try:
        sdir = skill_dir(slug)
        _log(slug, f"Starting IMPRINT pipeline for {colleague_name}")

        # ── Stage 1: Intake ──────────────────────────────────────────────────
        _log(slug, "Stage 1/5 — Ingesting trace files...")
        folder = Path(trace_folder)
        corpus: Dict[str, str] = {}
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in {".txt", ".md", ".json", ".csv"}:
                try:
                    corpus[f.name] = f.read_text(encoding="utf-8")
                    _log(slug, f"  Loaded: {f.name} ({len(corpus[f.name])} chars)")
                except Exception as e:
                    _log(slug, f"  WARN: Could not read {f.name}: {e}")

        if not corpus:
            _log(slug, "ERROR: No trace files found.")
            job_status[slug]["status"] = "error"
            return

        # Save raw corpus
        (sdir / "raw_corpus.json").write_text(
            json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ── Stage 2: Embed + FAISS index ─────────────────────────────────────
        _log(slug, "Stage 2/5 — Building FAISS vector index...")
        all_chunks, all_embeddings = [], []
        chunk_metadata = []

        for filename, text in corpus.items():
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                emb = get_embedding(chunk)
                all_chunks.append(chunk)
                all_embeddings.append(emb)
                chunk_metadata.append({"file": filename, "chunk_index": i})

        _log(slug, f"  Embedded {len(all_chunks)} chunks across {len(corpus)} files")

        arr = np.array(all_embeddings).astype("float32")
        dim = arr.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(arr)

        # Persist index + chunks for chat use later
        faiss.write_index(index, str(sdir / "faiss.index"))
        with open(sdir / "chunks.pkl", "wb") as f:
            pickle.dump({"chunks": all_chunks, "metadata": chunk_metadata}, f)
        _log(slug, "  FAISS index saved.")

        # ── Stage 3: RAG distillation ─────────────────────────────────────────
        _log(slug, "Stage 3/5 — RAG distillation pass...")
        file_summaries = []
        for filename, text in corpus.items():
            chunks = chunk_text(text)
            if not chunks:
                continue
            embs = np.array([get_embedding(c) for c in chunks]).astype("float32")
            idx = faiss.IndexFlatL2(embs.shape[1])
            idx.add(embs)
            q_emb = np.array([get_embedding(
                f"Key decision heuristics, escalation patterns, expertise in {filename}"
            )]).astype("float32")
            D, I = idx.search(q_emb, k=min(5, len(chunks)))
            retrieved = [chunks[i] for i in I[0]]
            summary = bedrock_call(
                f"Distil expert knowledge from '{filename}'.\n"
                f"Extract: decision heuristics, escalation criteria, communication norms, "
                f"recurring judgment calls. Be specific and concrete.\n\n"
                + "\n\n---\n\n".join(retrieved)
            )
            file_summaries.append(f"### Source: {filename}\n{summary}")
            _log(slug, f"  Distilled: {filename}")

        consolidated = bedrock_call(
            "Synthesise these per-document distillations into one expert profile covering:\n"
            "- Core expertise and domain knowledge\n- Decision heuristics and triage\n"
            "- Escalation thresholds and patterns\n- Communication style\n"
            "- COE / lessons-learned patterns\n\n"
            + "\n\n".join(file_summaries),
            max_tokens=3000,
        )
        (sdir / "consolidated.md").write_text(consolidated, encoding="utf-8")
        _log(slug, "  Consolidated distillation saved.")

        # ── Stage 4: Dual-track distillation ─────────────────────────────────
        _log(slug, "Stage 4/5 — Dual-track distillation (capability + persona)...")
        capability_track = bedrock_call(
            f"Generate the CAPABILITY TRACK for {colleague_name}.\n"
            f"Source:\n{consolidated}\n\n"
            "Sections: Core Domain Expertise, Decision Heuristics, Triage Criteria, "
            "Escalation Patterns, COE Patterns, Task Workflows. Be specific and concrete.",
            max_tokens=2500,
        )
        (sdir / "work.md").write_text(
            f"# Capability Track — {colleague_name}\n\n{capability_track}", encoding="utf-8"
        )
        _log(slug, "  Capability track written.")

        persona_track = bedrock_call(
            f"Generate the PERSONA/BEHAVIOR TRACK for {colleague_name}.\n"
            f"Source:\n{consolidated}\n\n"
            "Sections: Communication Style, Response Structure, Uncertainty Signalling, "
            "Interaction Rules, Escalation Communication. Bounded style reference only — "
            "not impersonation.\n\nEnd with:\n## Correction History\n```\n[]\n```",
            max_tokens=2000,
        )
        (sdir / "persona.md").write_text(
            f"# Persona Track — {colleague_name}\n\n{persona_track}", encoding="utf-8"
        )
        _log(slug, "  Persona track written.")

        # ── Stage 5: Artifact writer ──────────────────────────────────────────
        _log(slug, "Stage 5/5 — Writing SKILL.md and metadata...")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        skill_md = (
            f"---\nname: {slug}-skill\ndescription: Person-grounded AI skill for "
            f"{colleague_name}.\nversion: 1.0.0\ngenerated: {timestamp}\n"
            f"preset: colleague\nuser-invocable: true\n---\n\n"
            f"# {colleague_name} — IMPRINT Skill\n\n"
            f"# PART A — Capability Track\n\n{capability_track}\n\n"
            f"---\n\n# PART B — Behavior Track\n\n{persona_track}\n"
        )
        (sdir / "SKILL.md").write_text(skill_md, encoding="utf-8")

        manifest = {
            "name": f"{slug}-skill", "version": "1.0.0", "generated": timestamp,
            "entrypoints": {
                "full": "SKILL.md", "capability-only": "work.md", "persona-only": "persona.md"
            },
            "source-files": list(corpus.keys()),
        }
        (sdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        meta = {
            "schema-version": 3, "slug": slug, "colleague-name": colleague_name,
            "generated": timestamp, "version": "1.0.0", "correction-count": 0,
            "rollback-history": [], "source-scope": list(corpus.keys()),
            "lifecycle-status": "active",
            "governance": {
                "consent-obtained": False, "associate-owned": True,
                "deletable": True, "local-first-storage": True,
                "not-for-performance-mgmt": True,
            },
        }
        (sdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        _log(slug, "Pipeline complete! ✓")
        job_status[slug]["status"] = "complete"
        job_status[slug]["result"] = {
            "slug": slug, "name": colleague_name,
            "files": ["SKILL.md", "work.md", "persona.md", "manifest.json", "meta.json",
                      "faiss.index", "chunks.pkl"],
        }

    except Exception as e:
        _log(slug, f"PIPELINE ERROR: {e}")
        job_status[slug]["status"] = "error"


# ── Chat with skill (RAG) ─────────────────────────────────────────────────────

def chat_with_skill(slug: str, user_message: str, mode: str = "full") -> Dict:
    """
    RAG-backed chat. Retrieves relevant chunks from the FAISS index,
    then prompts Claude to respond AS the skill (bounded — not impersonation).
    mode: full | capability | persona
    """
    sdir = skill_dir(slug)
    meta = load_meta(slug)
    if not meta:
        return {"error": f"Skill '{slug}' not found."}

    colleague_name = meta.get("colleague-name", slug)

    # Load FAISS index + chunks
    index_path = sdir / "faiss.index"
    chunks_path = sdir / "chunks.pkl"
    if not index_path.exists() or not chunks_path.exists():
        return {"error": "Vector index not found. Re-run the distillation pipeline."}

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        store = pickle.load(f)
    chunks = store["chunks"]
    chunk_meta = store.get("metadata", [{}] * len(chunks))

    # Retrieve top-k relevant chunks
    q_emb = np.array([get_embedding(user_message)]).astype("float32")
    k = min(6, len(chunks))
    D, I = index.search(q_emb, k=k)
    retrieved = [chunks[i] for i in I[0]]
    sources = [chunk_meta[i].get("file", "unknown") for i in I[0]]

    # Load the right track for context
    if mode == "capability":
        track_file = sdir / "work.md"
    elif mode == "persona":
        track_file = sdir / "persona.md"
    else:
        track_file = sdir / "SKILL.md"

    skill_context = track_file.read_text(encoding="utf-8") if track_file.exists() else ""

    system_prompt = (
        f"You are {colleague_name}, an AI assistant built from their documented work style, "
        f"expertise, and decision patterns. Respond naturally and conversationally — like a "
        f"knowledgeable colleague, not a formal system.\n\n"
        f"Embody their persona fully: use their communication style, tone, and judgment calls "
        f"as documented. Be direct and helpful. If knowledge is limited, say so briefly and "
        f"move on — don't over-hedge or repeat disclaimers.\n\n"
        f"OUTPUT FORMAT: If the user asks for CSV, JSON, a table, bullet points, or any "
        f"specific format, respond ONLY in that format without preamble. Match the format "
        f"of any input data provided (e.g. if given CSV input, output CSV).\n\n"
        f"SKILL ARTIFACT:\n{skill_context[:6000]}"
    )

    rag_context = "\n\n---\n\n".join(
        f"[From: {src}]\n{chunk}"
        for src, chunk in zip(sources, retrieved)
    )
    prompt = (
        f"CONTEXT FROM TRACES:\n{rag_context}\n\n"
        f"USER: {user_message}"
    )

    response = bedrock_call(prompt, system=system_prompt, max_tokens=2000)

    return {
        "response": response,
        "sources": list(set(sources)),
        "chunks_retrieved": len(retrieved),
        "mode": mode,
        "colleague": colleague_name,
    }

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/skills", methods=["GET"])
def api_list_skills():
    return jsonify({"skills": list_skills()})


@app.route("/api/skill/<slug>", methods=["GET"])
def api_get_skill(slug):
    meta = load_meta(slug)
    if not meta:
        return jsonify({"error": "Skill not found"}), 404
    sdir = skill_dir(slug)
    skill_preview = ""
    skill_path = sdir / "SKILL.md"
    if skill_path.exists():
        skill_preview = skill_path.read_text(encoding="utf-8")[:3000]
    return jsonify({"meta": meta, "skill_preview": skill_preview})


@app.route("/api/create", methods=["POST"])
def api_create():
    data = request.json or {}
    colleague_name = (data.get("colleague_name") or "").strip()
    trace_folder   = (data.get("trace_folder") or "traces").strip()

    if not colleague_name:
        return jsonify({"error": "colleague_name is required"}), 400
    if not Path(trace_folder).exists():
        return jsonify({"error": f"Trace folder '{trace_folder}' not found"}), 400

    slug = re.sub(r"[^a-z0-9]", "-", colleague_name.lower().strip())
    job_status[slug] = {"logs": [], "status": "running", "result": None}

    thread = threading.Thread(
        target=run_pipeline, args=(slug, colleague_name, trace_folder), daemon=True
    )
    thread.start()

    return jsonify({"slug": slug, "status": "started",
                    "message": f"Pipeline started for {colleague_name}"})


@app.route("/api/job/<slug>", methods=["GET"])
def api_job_status(slug):
    info = job_status.get(slug)
    if not info:
        # Check if skill already exists
        meta = load_meta(slug)
        if meta:
            return jsonify({"status": "complete", "logs": [], "result": {"slug": slug}})
        return jsonify({"error": "Job not found"}), 404
    return jsonify(info)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    slug    = (data.get("slug") or "").strip()
    message = (data.get("message") or "").strip()
    mode    = (data.get("mode") or "full").strip()

    if not slug or not message:
        return jsonify({"error": "slug and message are required"}), 400

    result = chat_with_skill(slug, message, mode)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route("/api/correct", methods=["POST"])
def api_correct():
    data = request.json or {}
    slug    = (data.get("slug") or "").strip()
    scene   = (data.get("scene") or "").strip()
    wrong   = (data.get("wrong") or "").strip()
    correct = (data.get("correct") or "").strip()

    if not all([slug, scene, wrong, correct]):
        return jsonify({"error": "slug, scene, wrong, correct are all required"}), 400

    sdir = skill_dir(slug)
    meta_path = sdir / "meta.json"
    persona_path = sdir / "persona.md"

    if not meta_path.exists():
        return jsonify({"error": f"Skill '{slug}' not found"}), 404

    meta = json.loads(meta_path.read_text())
    version = meta.get("version", "1.0.0")

    # Archive current persona
    if persona_path.exists():
        archive = sdir / f"persona_v{version}_archive.md"
        shutil.copy(persona_path, archive)

    # Append correction record
    record = {"scene": scene, "wrong": wrong, "correct": correct,
              "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    persona_text = persona_path.read_text(encoding="utf-8") if persona_path.exists() else ""

    if "## Correction History" in persona_text:
        parts = persona_text.split("## Correction History")
        try:
            existing = json.loads(
                re.search(r"```\n(.*?)\n```", parts[1], re.DOTALL).group(1)
            )
        except Exception:
            existing = []
        existing.append(record)
        persona_text = parts[0] + f"\n## Correction History\n```\n{json.dumps(existing, indent=2)}\n```\n"
    else:
        persona_text += f"\n## Correction History\n```\n{json.dumps([record], indent=2)}\n```\n"

    persona_path.write_text(persona_text, encoding="utf-8")

    # Bump version
    parts_v = version.split(".")
    parts_v[-1] = str(int(parts_v[-1]) + 1)
    new_version = ".".join(parts_v)
    meta["version"] = new_version
    meta["correction-count"] = meta.get("correction-count", 0) + 1
    meta["rollback-history"].append({"from-version": version, "archive": f"persona_v{version}_archive.md"})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return jsonify({"message": "Correction applied", "new_version": new_version,
                    "correction_count": meta["correction-count"]})


@app.route("/api/delete", methods=["POST"])
def api_delete():
    data = request.json or {}
    slug = (data.get("slug") or "").strip()
    if not slug:
        return jsonify({"error": "slug is required"}), 400

    sdir = skill_dir(slug)
    if not sdir.exists():
        return jsonify({"error": f"Skill '{slug}' not found"}), 404

    shutil.rmtree(sdir)
    if slug in job_status:
        del job_status[slug]

    return jsonify({"message": f"Skill '{slug}' deleted successfully."})


if __name__ == "__main__":
    print("#--------------------------------------------------------#")
    print("          Project IMPRINT — API Server v1.0               ")
    print("          http://localhost:5000                            ")
    print("#--------------------------------------------------------#")
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)
