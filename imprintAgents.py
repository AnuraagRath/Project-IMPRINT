import os
import re
import json
import time
import pickle
import shutil
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import boto3
import botocore
import faiss
from dotenv import load_dotenv

# -----------------------------
# ENV + Bedrock client
# -----------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
# modelId="anthropic.claude-3-sonnet-20240229-v1:0",
MODEL_ID   = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
EMBED_ID   = "amazon.titan-embed-text-v2:0"
MAX_TOKENS = 2000

# =====================================================================
# SECTION 1 — INTAKE: parse raw trace files from a folder
# =====================================================================

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv"}

def ingest_traces(trace_folder: str) -> Dict[str, str]:
    """
    Read all supported files from the trace folder and return
    a dict of {filename: raw_text}.
    """
    folder = Path(trace_folder)
    if not folder.exists():
        print(f"[WARN] Trace folder '{trace_folder}' not found.")
        return {}

    corpus: Dict[str, str] = {}
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                if f.suffix.lower() == ".json":
                    data = json.loads(f.read_text(encoding="utf-8"))
                    corpus[f.name] = json.dumps(data, indent=2)
                else:
                    corpus[f.name] = f.read_text(encoding="utf-8")
                print(f"[INTAKE] Loaded: {f.name} ({len(corpus[f.name])} chars)")
            except Exception as e:
                print(f"[WARN] Could not read {f.name}: {e}")

    print(f"[INTAKE] Total files loaded: {len(corpus)}")
    return corpus


def save_raw_corpus(corpus: Dict[str, str], slug: str):
    """Persist raw corpus as a single JSON for audit trail."""
    out = Path(f"imprint_{slug}_raw_corpus.json")
    out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INTAKE] Raw corpus saved → {out.name}")


# =====================================================================
# SECTION 2 — EMBED + RAG (mirrors anuRAG from vizSquadAgents)
# =====================================================================

def get_embedding(text: str) -> List[float]:
    body = json.dumps({"inputText": text[:8000]})   # Titan v2 limit
    response = bedrock.invoke_model(modelId=EMBED_ID, body=body)
    result = json.loads(response["body"].read())
    return result["embedding"]


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_with_cache(chunks: List[str], cache_file: Path):
    if cache_file.exists():
        print(f"[RAG] Cache hit: {cache_file.name}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return np.array(data["embeddings"]).astype("float32"), data["chunks"]

    print(f"[RAG] Embedding {len(chunks)} chunks...")
    embeddings = []
    for i, c in enumerate(chunks):
        emb = get_embedding(c)
        embeddings.append(emb)
        if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
            print(f"[RAG] Embedded {i+1}/{len(chunks)} chunks")

    arr = np.array(embeddings).astype("float32")
    with open(cache_file, "wb") as f:
        pickle.dump({"embeddings": arr.tolist(), "chunks": chunks}, f)
    return arr, chunks


def bedrock_call(prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """Single Bedrock Claude call with throttle-retry (mirrors claude_summarize)."""
    body = json.dumps({
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
        "anthropic_version": "bedrock-2023-05-31",
    })
    retries, max_wait = 10, 60
    for attempt in range(retries):
        try:
            response = bedrock.invoke_model(modelId=MODEL_ID, body=body)
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                wait = min((2 ** attempt) * 3 + random.random(), max_wait)
                print(f"[WARN] Throttled. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] Bedrock error: {e}")
                raise
    raise RuntimeError("Bedrock call failed after retries.")


def rag_distill_corpus(corpus: Dict[str, str], slug: str) -> str:
    """
    FAISS-backed RAG pass over all trace files.
    Returns a consolidated plain-text distillation (mirrors anuRAG).
    """
    print("\n#--------------------------------------------------------#")
    print("      [INFO] Initiating RAG Distillation Pass             ")
    print("#--------------------------------------------------------#\n")

    file_summaries: List[str] = []

    for filename, text in corpus.items():
        chunks = chunk_text(text)
        if not chunks:
            print(f"[WARN] {filename} produced 0 chunks, skipping.")
            continue

        print(f"[RAG] Processing {filename}: {len(chunks)} chunks")
        cache_file = Path(f"imprint_{slug}_{Path(filename).stem}_emb.pkl")
        embeddings, chunks = embed_with_cache(chunks, cache_file)

        if embeddings.shape[0] == 0:
            print(f"[WARN] No embeddings for {filename}, skipping.")
            continue

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        query_text = (
            f"What are the key decision heuristics, escalation patterns, "
            f"work habits, communication style, and expertise in {filename}?"
        )
        query_emb = np.array([get_embedding(query_text)]).astype("float32")
        D, I = index.search(query_emb, k=min(5, len(chunks)))
        retrieved = [chunks[i] for i in I[0]]

        summary_prompt = (
            f"You are distilling expert knowledge from a work trace document called '{filename}'.\n"
            f"Extract durable heuristics, decision patterns, escalation criteria, "
            f"communication norms, and any recurring judgment calls.\n"
            f"Be specific and concrete — avoid generic statements.\n\n"
            + "\n\n---\n\n".join(retrieved)
        )
        summary = bedrock_call(summary_prompt)
        file_summaries.append(f"### Source: {filename}\n{summary}\n")
        print(f"[RAG] Distilled: {filename}")

    if not file_summaries:
        print("[WARN] No valid summaries produced.")
        return ""

    print("[RAG] Generating consolidated distillation...")
    master_prompt = (
        "You are building a skill package for an expert professional.\n"
        "Below are per-document distillations from their work traces.\n"
        "Synthesise these into a single coherent profile covering:\n"
        "- Core technical expertise and domain knowledge\n"
        "- Decision heuristics and triage criteria\n"
        "- Escalation thresholds and patterns\n"
        "- Communication style and response norms\n"
        "- Notable COE / lessons-learned patterns\n\n"
        + "\n\n".join(file_summaries)
    )
    consolidated = bedrock_call(master_prompt, max_tokens=3000)

    # Clean up embedding caches (mirrors deleteFiles)
    for pkl in Path(".").glob(f"imprint_{slug}_*_emb.pkl"):
        pkl.unlink()
        print(f"[CLEAN] Removed cache: {pkl.name}")

    return consolidated


# =====================================================================
# SECTION 3 — DUAL-TRACK DISTILLATION
# =====================================================================

def distill_capability_track(consolidated: str, colleague_name: str, slug: str) -> str:
    """
    Capability track: practices, mental models, decision heuristics, escalation.
    Mirrors the 'capability track' from COLLEAGUE.SKILL.
    """
    print("[DISTILL] Building capability track...")
    prompt = f"""You are generating the CAPABILITY TRACK of an AI skill artifact for {colleague_name}.

The capability track captures WHAT they know and HOW they make decisions.
It must be concrete, inspectable, and directly usable by an AI agent.

Source distillation:
{consolidated}

Generate the capability track covering these sections with real specifics from the data above:

## Core Domain Expertise
(What domains, tools, systems they are expert in)

## Decision Heuristics
(Specific rules-of-thumb they apply — e.g. "Always check auth before rate limits")

## Triage Criteria
(How they prioritise — what gets escalated vs handled, what's P1 vs P3)

## Escalation Patterns
(When and how they escalate, to whom, with what framing)

## COE / Lessons Learned Patterns
(Recurring root-cause categories, failure modes they watch for)

## Task Workflows
(Typical step sequences for common task types)

Be specific. No generic filler. If data is thin, note [INFERRED] or [THIN-EVIDENCE]."""

    track = bedrock_call(prompt, max_tokens=2500)
    out = Path(f"imprint_{slug}_work.md")
    out.write_text(
        f"# Capability Track — {colleague_name}\n"
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
        + track,
        encoding="utf-8",
    )
    print(f"[DISTILL] Capability track → {out.name}")
    return track


def distill_persona_track(consolidated: str, colleague_name: str, slug: str) -> str:
    """
    Persona / behavior track: communication style, interaction rules, uncertainty signals.
    Mirrors the 'persona track' from COLLEAGUE.SKILL.
    """
    print("[DISTILL] Building persona track...")
    prompt = f"""You are generating the PERSONA / BEHAVIOR TRACK of an AI skill artifact for {colleague_name}.

This track captures HOW they communicate, NOT a simulation of them as a person.
It defines bounded behavior constraints — expression preferences and interaction rules only.

Source distillation:
{consolidated}

Generate the behavior track covering:

## Communication Style
(Tone, formality level, typical reply length, use of bullets vs prose)

## Response Structure Preferences
(How they structure explanations — do they lead with verdict, or context first?)

## Uncertainty Signalling
(How they express low confidence — hedges, explicit flags, escalation instead of guessing)

## Interaction Rules
(Things they always / never do in responses — e.g. "always include a next-action", "never guess on security issues")

## Escalation Communication
(How they frame handoffs and escalations — what info they always include)

## Correction History
(Leave blank — populated via feedback loop post-deployment)
```
[]
```

Keep this bounded. This is NOT impersonation — it is a constrained style reference."""

    track = bedrock_call(prompt, max_tokens=2000)
    out = Path(f"imprint_{slug}_persona.md")
    out.write_text(
        f"# Persona Track — {colleague_name}\n"
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
        + track,
        encoding="utf-8",
    )
    print(f"[DISTILL] Persona track → {out.name}")
    return track


# =====================================================================
# SECTION 4 — ARTIFACT WRITER (SKILL.md + manifest)
# =====================================================================

def write_skill_artifact(
    colleague_name: str,
    slug: str,
    capability_track: str,
    persona_track: str,
    source_files: List[str],
):
    """
    Writes the final SKILL.md package — the combined invokable artifact.
    Also writes manifest.json and meta.json for agent-host compatibility.
    """
    print("[ARTIFACT] Writing SKILL.md package...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = "1.0.0"

    skill_md = f"""---
name: {slug}-skill
description: >
  Person-grounded AI skill for {colleague_name}.
  Distilled from work traces via Project IMPRINT.
  Contains capability track (heuristics, triage, escalation) and
  bounded behavior track (communication style, interaction rules).
version: {version}
generated: {timestamp}
preset: colleague
user-invocable: true
source-files: {json.dumps(source_files)}
correction-count: 0
---

# {colleague_name} — IMPRINT Skill Package

> This is a bounded skill artifact. It encodes selected expertise and
> communication patterns from work traces. It does not simulate or
> replace {colleague_name} as a person.

---

# PART A — Capability Track

{capability_track}

---

# PART B — Behavior Track (Bounded)

{persona_track}

---

# PART C — Operating Rules

- Use Part A for task reasoning, triage decisions, and escalation judgment.
- Use Part B for response style only — do not treat it as identity simulation.
- If evidence for a decision is thin, flag it explicitly rather than inferring.
- When asked "How would [{slug}-skill] approach this?", apply Part A reasoning
  within Part B style constraints and stay within documented boundaries.
- This artifact can be corrected, versioned, and deleted by its subject at any time.
"""

    skill_path = Path(f"imprint_{slug}_SKILL.md")
    skill_path.write_text(skill_md, encoding="utf-8")
    print(f"[ARTIFACT] SKILL.md → {skill_path.name}")

    # manifest.json — agent-host install metadata
    manifest = {
        "name": f"{slug}-skill",
        "version": version,
        "generated": timestamp,
        "preset": "colleague",
        "entrypoints": {
            "full": f"imprint_{slug}_SKILL.md",
            "capability-only": f"imprint_{slug}_work.md",
            "persona-only": f"imprint_{slug}_persona.md",
        },
        "slash-commands": [
            f"/{slug}",
            f"/{slug}-work",
            f"/{slug}-persona",
        ],
        "compatible-hosts": ["claude-code", "kiro-ai", "any-agent-skills-v3"],
        "source-files": source_files,
    }
    manifest_path = Path(f"imprint_{slug}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ARTIFACT] manifest.json → {manifest_path.name}")

    # meta.json — lifecycle + governance metadata
    meta = {
        "schema-version": 3,
        "slug": slug,
        "colleague-name": colleague_name,
        "generated": timestamp,
        "version": version,
        "correction-count": 0,
        "rollback-history": [],
        "governance": {
            "consent-obtained": False,      # Set True after colleague sign-off
            "associate-owned": True,
            "deletable": True,
            "local-first-storage": True,
            "not-for-performance-mgmt": True,
        },
        "source-scope": source_files,
        "lifecycle-status": "draft",         # draft → reviewed → active → archived
    }
    meta_path = Path(f"imprint_{slug}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ARTIFACT] meta.json → {meta_path.name}")

    return skill_path.name


# =====================================================================
# SECTION 5 — CORRECTION HANDLER
# =====================================================================

def apply_correction(slug: str, scene: str, wrong: str, correct: str):
    """
    Apply a natural-language correction to the persona track.
    Appends a structured correction record and bumps version.
    Mirrors the correction lifecycle from COLLEAGUE.SKILL.
    """
    meta_path = Path(f"imprint_{slug}_meta.json")
    persona_path = Path(f"imprint_{slug}_persona.md")

    if not meta_path.exists() or not persona_path.exists():
        print(f"[ERROR] Cannot find artifact for slug '{slug}'. Run distillation first.")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Archive current persona before patching
    version = meta.get("version", "1.0.0")
    archive_path = Path(f"imprint_{slug}_persona_v{version}_archive.md")
    shutil.copy(persona_path, archive_path)
    print(f"[CORRECTION] Archived current persona → {archive_path.name}")

    # Append correction record to persona.md
    record = {
        "scene": scene,
        "wrong": wrong,
        "correct": correct,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    persona_text = persona_path.read_text(encoding="utf-8")

    # Find ## Correction History section and append there
    if "## Correction History" in persona_text:
        # Pull out current records, append new one
        parts = persona_text.split("## Correction History")
        existing_block = parts[1].strip()
        try:
            existing_list = json.loads(
                re.search(r"```\n(.*?)\n```", existing_block, re.DOTALL).group(1)
            )
        except Exception:
            existing_list = []
        existing_list.append(record)
        new_block = (
            "\n## Correction History\n```\n"
            + json.dumps(existing_list, indent=2)
            + "\n```\n"
        )
        persona_text = parts[0] + new_block
    else:
        persona_text += (
            "\n## Correction History\n```\n"
            + json.dumps([record], indent=2)
            + "\n```\n"
        )

    persona_path.write_text(persona_text, encoding="utf-8")

    # Bump version in meta
    parts = version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    new_version = ".".join(parts)
    meta["version"] = new_version
    meta["correction-count"] = meta.get("correction-count", 0) + 1
    meta["rollback-history"].append({"from-version": version, "archive": archive_path.name})
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[CORRECTION] Correction applied. Version {version} → {new_version}")
    print(f"[CORRECTION] Correction count: {meta['correction-count']}")


# =====================================================================
# SECTION 6 — ARCHIVE + CLEANUP
# =====================================================================

def archive_imprint(slug: str, colleague_name: str):
    """Move all imprint artifact files into a versioned output folder."""
    folder = Path(f"IMPRINT_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    folder.mkdir(exist_ok=True)

    moved = 0
    for f in Path(".").iterdir():
        if f.is_file() and f.name.startswith(f"imprint_{slug}"):
            try:
                shutil.move(str(f), folder / f.name)
                print(f"[ARCHIVE] {f.name} → {folder.name}/")
                moved += 1
            except Exception as e:
                print(f"[ERROR] Could not move {f.name}: {e}")

    print(f"[ARCHIVE] {moved} files archived to {folder.name}/")
    return str(folder)


def delete_imprint(slug: str):
    """Delete all artifacts for a slug (associate-owned deletion right)."""
    deleted = 0
    for f in Path(".").iterdir():
        if f.is_file() and f.name.startswith(f"imprint_{slug}"):
            f.unlink()
            print(f"[DELETE] Removed: {f.name}")
            deleted += 1
    # Also check archived folders
    for d in Path(".").iterdir():
        if d.is_dir() and f"IMPRINT_{slug}" in d.name:
            shutil.rmtree(d)
            print(f"[DELETE] Removed folder: {d.name}")
            deleted += 1
    print(f"[DELETE] {deleted} items removed for slug '{slug}'.")
