import os
import re
import json
from pathlib import Path
from typing import Dict, Any

# ── Graceful LangGraph stub (mirrors agents.py exactly) ──────────────────────
try:
    from langgraph.graph import StateGraph, END
except Exception:
    class StateGraph:
        def __init__(self, state_type):
            self.nodes = {}
            self.edges = {}
            self.entry = None

        def add_node(self, name, func):
            self.nodes[name] = func

        def set_entry_point(self, name):
            self.entry = name

        def add_edge(self, a, b):
            self.edges.setdefault(a, []).append(b)

        def compile(self):
            return self

        def invoke(self, state: Dict[str, Any]):
            cur = self.entry
            ctx = dict(state)
            while cur and cur != "END":
                fn = self.nodes[cur]
                print(f"\n[GRAPH] ── Running node: {cur} ──")
                out = fn(ctx)
                if isinstance(out, dict):
                    ctx.update(out)
                nexts = self.edges.get(cur, [])
                cur = nexts[0] if nexts else "END"
            return ctx

    END = "END"

from imprintAgents import (
    ingest_traces,
    save_raw_corpus,
    rag_distill_corpus,
    distill_capability_track,
    distill_persona_track,
    write_skill_artifact,
    apply_correction,
    archive_imprint,
    delete_imprint,
)

# =====================================================================
# PIPELINE NODES
# =====================================================================

def node_intake(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load all trace files from the input folder."""
    corpus = ingest_traces(state["trace_folder"])
    if not corpus:
        print("[ERROR] No trace files found. Aborting.")
        return {**state, "abort": True}
    save_raw_corpus(corpus, state["slug"])
    return {**state, "corpus": corpus, "source_files": list(corpus.keys())}


def node_rag(state: Dict[str, Any]) -> Dict[str, Any]:
    """FAISS-backed RAG distillation over all trace files."""
    if state.get("abort"):
        return state
    consolidated = rag_distill_corpus(state["corpus"], state["slug"])
    if not consolidated:
        print("[ERROR] RAG distillation produced no output.")
        return {**state, "abort": True}
    return {**state, "consolidated": consolidated}


def node_capability(state: Dict[str, Any]) -> Dict[str, Any]:
    """Distill capability track (heuristics, triage, escalation)."""
    if state.get("abort"):
        return state
    capability_track = distill_capability_track(
        state["consolidated"],
        state["colleague_name"],
        state["slug"],
    )
    return {**state, "capability_track": capability_track}


def node_persona(state: Dict[str, Any]) -> Dict[str, Any]:
    """Distill persona / behavior track (communication style, interaction rules)."""
    if state.get("abort"):
        return state
    persona_track = distill_persona_track(
        state["consolidated"],
        state["colleague_name"],
        state["slug"],
    )
    return {**state, "persona_track": persona_track}


def node_artifact(state: Dict[str, Any]) -> Dict[str, Any]:
    """Write SKILL.md, manifest.json, meta.json."""
    if state.get("abort"):
        return state
    skill_file = write_skill_artifact(
        state["colleague_name"],
        state["slug"],
        state["capability_track"],
        state["persona_track"],
        state["source_files"],
    )
    return {**state, "skill_file": skill_file}


def node_archive(state: Dict[str, Any]) -> Dict[str, Any]:
    """Move all artifact files into an output folder."""
    if state.get("abort"):
        return state
    folder = archive_imprint(state["slug"], state["colleague_name"])
    return {**state, "output_folder": folder}


# =====================================================================
# GRAPH CONSTRUCTION
# Intake → RAG → Capability → Persona → Artifact → Archive → END
# =====================================================================

def build_graph():
    graph = StateGraph(dict)

    graph.add_node("intake",     node_intake)
    graph.add_node("rag",        node_rag)
    graph.add_node("capability", node_capability)
    graph.add_node("persona",    node_persona)
    graph.add_node("artifact",   node_artifact)
    graph.add_node("archive",    node_archive)

    graph.set_entry_point("intake")
    graph.add_edge("intake",     "rag")
    graph.add_edge("rag",        "capability")
    graph.add_edge("capability", "persona")
    graph.add_edge("persona",    "artifact")
    graph.add_edge("artifact",   "archive")
    graph.add_edge("archive",    END)

    return graph.compile()


# =====================================================================
# RUN ORCHESTRATION
# =====================================================================

def run_imprint(colleague_name: str, trace_folder: str):
    slug = re.sub(r"[^a-z0-9]", "-", colleague_name.lower().strip())

    print("#--------------------------------------------------------#")
    print("               Project IMPRINT  v1.0                      ")
    print("    Capturing Operational Expertise as AI Skills           ")
    print("#--------------------------------------------------------#\n")
    print(f"  Colleague  : {colleague_name}")
    print(f"  Slug       : {slug}")
    print(f"  Trace Folder: {trace_folder}")
    print()

    state = {
        "colleague_name": colleague_name,
        "slug": slug,
        "trace_folder": trace_folder,
    }

    graph = build_graph()
    final = graph.invoke(state)

    print("\n#--------------------------------------------------------#")
    if final.get("abort"):
        print("  [IMPRINT] Pipeline aborted — check warnings above.")
    else:
        print(f"  [IMPRINT] Pipeline complete!")
        print(f"  Output folder : {final.get('output_folder', 'N/A')}")
        print(f"  SKILL.md      : {final.get('skill_file', 'N/A')}")
        print(f"  Source files  : {len(final.get('source_files', []))}")
    print("#--------------------------------------------------------#\n")

    return final


def run_correction(slug: str):
    """Interactive correction mode — apply feedback to a deployed artifact."""
    print("#--------------------------------------------------------#")
    print("          Project IMPRINT — Correction Mode               ")
    print("#--------------------------------------------------------#\n")
    print("Enter a correction for the persona/behavior track.")
    print("(Leave blank to cancel)\n")

    scene  = input("Scene (context where wrong behavior occurs): ").strip()
    if not scene:
        print("[CANCELLED]")
        return
    wrong  = input("Wrong (what the artifact currently does): ").strip()
    correct = input("Correct (what it should do instead): ").strip()

    if not wrong or not correct:
        print("[CANCELLED] Both 'wrong' and 'correct' are required.")
        return

    apply_correction(slug, scene, wrong, correct)
    print("\n[CORRECTION] Done. Re-run distillation to regenerate SKILL.md from updated tracks.")


def run_delete(slug: str):
    """Delete an artifact — associate-owned deletion right."""
    confirm = input(
        f"⚠️  Delete ALL artifacts for slug '{slug}'? "
        f"This cannot be undone. Type YES to confirm: "
    ).strip()
    if confirm == "YES":
        delete_imprint(slug)
    else:
        print("[CANCELLED] Deletion aborted.")


# =====================================================================
# CLI ENTRY POINT
# =====================================================================

def main():
    print("#--------------------------------------------------------#")
    print("               Project IMPRINT  v1.0                      ")
    print("#--------------------------------------------------------#\n")
    print("  [1]  Create new skill (full pipeline)")
    print("  [2]  Apply correction to existing skill")
    print("  [3]  Delete skill artifact")
    print()

    choice = input("Select an option [1/2/3]: ").strip()

    if choice == "1":
        colleague_name = input("\nColleague name (e.g. Ashwath Rajan): ").strip()
        if not colleague_name:
            print("[ERROR] Name cannot be empty.")
            return
        trace_folder = input(
            "Path to trace folder (txt/md/json/csv files): "
        ).strip()
        if not trace_folder:
            trace_folder = "traces"
        run_imprint(colleague_name, trace_folder)

    elif choice == "2":
        slug = input("\nSkill slug (e.g. ashwath-rajan): ").strip()
        if not slug:
            print("[ERROR] Slug cannot be empty.")
            return
        run_correction(slug)

    elif choice == "3":
        slug = input("\nSkill slug to delete: ").strip()
        if not slug:
            print("[ERROR] Slug cannot be empty.")
            return
        run_delete(slug)

    else:
        print("[ERROR] Invalid option.")


if __name__ == "__main__":
    main()
