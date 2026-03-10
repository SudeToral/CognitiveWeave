"""
CognitiveWeave — Live Demo
==========================
Shows the core value proposition in ~30 seconds:

  "Standard AI memory gets fooled by stale knowledge.
   CognitiveWeave doesn't."

Run:
  uv run python demo.py
"""
from __future__ import annotations

import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cognitiveweave.config.settings import Settings
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever

# ── helpers ─────────────────────────────────────────────────────────────────

def p(text: str = "") -> None:
    print(text, flush=True)

def step(n: int, title: str) -> None:
    p(f"\n\033[1;36mStep {n} — {title}\033[0m")

def ok(msg: str)   -> None: p(f"  \033[32m✓\033[0m  {msg}")
def warn(msg: str) -> None: p(f"  \033[33m!\033[0m  {msg}")
def bad(msg: str)  -> None: p(f"  \033[31m✗\033[0m  {msg}")
def info(msg: str) -> None: p(f"     {msg}")

def bar(score: float, width: int = 30) -> str:
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.0%}"

def ndcg(ids: list[str], relevant: set[str], k: int = 5) -> float:
    dcg  = sum(1/math.log2(i+2) for i, x in enumerate(ids[:k]) if x in relevant)
    idcg = sum(1/math.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg/idcg if idcg else 0.0

def mrr(ids: list[str], relevant: set[str]) -> float:
    for i, x in enumerate(ids):
        if x in relevant:
            return 1/(i+1)
    return 0.0

# ── constants ────────────────────────────────────────────────────────────────

QUERY    = "what brain mechanisms underlie memory consolidation?"
SEED     = ["c1_consolidation"]
RELEVANT = {"c3_swr", "c3_sleep_consolidation", "c3_hippocampus",
            "c1_protein_synthesis", "c1_reconsolidation", "c1_ltp"}

NOISE = {
    "noise_demo_1": "Deprecated model: engrams are stored exclusively in the prefrontal cortex, hippocampus is not involved.",
    "noise_demo_2": "Outdated hypothesis: REM sleep is the sole driver of consolidation, slow-wave sleep plays no role.",
    "noise_demo_3": "Superseded theory: memory consolidation completes within minutes via a single protein synthesis event.",
}

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p("\n\033[1;37m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    p("\033[1;37m  CognitiveWeave  ·  Temporal Knowledge Graph Demo\033[0m")
    p("\033[1;37m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")

    settings = Settings()
    neo4j = Neo4jClient(settings.neo4j)
    neo4j.connect()
    faiss = FAISSIndex(settings.faiss)
    faiss.load_model()
    from pathlib import Path
    if Path(settings.faiss.index_path).exists():
        faiss.load()
    else:
        faiss.build_index()
    retriever = HybridRetriever(faiss, neo4j)

    # ── Step 1: The problem ──────────────────────────────────────────────
    step(1, "The problem — stale knowledge enters the graph")
    p()
    info("An AI system has accumulated knowledge over months.")
    info("Some of it is now outdated — superseded theories, deprecated models.")
    info("We inject 3 stale nodes connected to 'memory consolidation':\n")

    for nid, content in NOISE.items():
        neo4j.upsert_node(nid, content=content, metadata={"cluster": "stale", "confidence": 0.2})
        faiss.add(nid, content, {"cluster": "stale"})
        neo4j.upsert_edge("c1_consolidation", nid, relation="RELATED_TO", weight=0.9)
        warn(content[:80] + "…")
    faiss.save()

    # reset all edges to fresh (no time has passed yet)
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        s.run("MATCH ()-[r]->() SET r.last_decay_at = toString(datetime())")

    # ── Step 2: Standard retrieval (no decay) ───────────────────────────
    step(2, "Standard retrieval — no temporal awareness")
    p()
    info(f"Query: \"{QUERY}\"")
    p()

    results_std = retriever.retrieve(QUERY, seed_node_ids=SEED, top_k=5)
    ids_std = [r.id for r in results_std]

    info("Top 5 results:")
    for i, r in enumerate(results_std):
        if r.id in RELEVANT:
            ok(f"#{i+1}  {r.id:<35}  rrf={r.rrf_score:.4f}")
        elif r.id in NOISE:
            bad(f"#{i+1}  {r.id:<35}  rrf={r.rrf_score:.4f}  ← STALE")
        else:
            info(f"   #{i+1}  {r.id:<35}  rrf={r.rrf_score:.4f}")

    n_std  = ndcg(ids_std, RELEVANT)
    m_std  = mrr(ids_std, RELEVANT)
    noise_std = sum(1 for x in ids_std if x in NOISE)
    p()
    bad(f"nDCG@5  {bar(n_std)}   ({noise_std} stale results in top 5)")

    time.sleep(0.8)

    # ── Step 3: Time passes ──────────────────────────────────────────────
    step(3, "Time passes — Ebbinghaus decay runs automatically")
    p()
    info("CognitiveWeave's CuratorAgent applies the forgetting curve:")
    info("  w_new = w_old × exp(−Δt / stability)")
    p()

    # Age noise edges 45 days, keep real edges fresh
    query_age = """
    UNWIND $ids AS nid
    MATCH ()-[r]->(b:KnowledgeNode {id: nid})
    SET r.last_decay_at = toString(datetime() - duration({days: 45}))
    """
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        s.run(query_age, ids=list(NOISE.keys()))

    updated = neo4j.decay_edge_weights()
    pruned  = neo4j.prune_weak_edges(threshold=0.3)

    # Show what happened to noise edge weights
    q_check = """
    UNWIND $ids AS nid
    MATCH ()-[r]->(b:KnowledgeNode {id: nid})
    RETURN b.id AS id, r.weight AS w
    """
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        remaining = {row["id"]: row["w"] for row in s.run(q_check, ids=list(NOISE.keys()))}

    for nid in NOISE:
        if nid in remaining:
            warn(f"{nid:<30}  edge weight → {remaining[nid]:.4f}  (was 0.9000)")
        else:
            ok(f"{nid:<30}  edge PRUNED  (decayed below threshold)")

    info(f"\n  {updated} edges decayed  ·  {pruned} edges pruned")

    time.sleep(0.8)

    # ── Step 4: CognitiveWeave retrieval (with decay) ────────────────────
    step(4, "CognitiveWeave retrieval — temporal awareness active")
    p()
    info(f"Query: \"{QUERY}\"")
    p()

    results_cw = retriever.retrieve(QUERY, seed_node_ids=SEED, top_k=5)
    ids_cw = [r.id for r in results_cw]

    info("Top 5 results:")
    for i, r in enumerate(results_cw):
        if r.id in RELEVANT:
            ok(f"#{i+1}  {r.id:<35}  rrf={r.rrf_score:.4f}")
        elif r.id in NOISE:
            bad(f"#{i+1}  {r.id:<35}  rrf={r.rrf_score:.4f}  ← STALE")
        else:
            info(f"   #{i+1}  {r.id:<35}  rrf={r.rrf_score:.4f}")

    n_cw  = ndcg(ids_cw, RELEVANT)
    m_cw  = mrr(ids_cw, RELEVANT)
    noise_cw = sum(1 for x in ids_cw if x in NOISE)
    p()
    ok(f"nDCG@5  {bar(n_cw)}   ({noise_cw} stale results in top 5)")

    # ── Step 5: Summary ──────────────────────────────────────────────────
    p()
    p("\033[1;37m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    p("\033[1;37m  Results\033[0m")
    p("\033[1;37m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    p()

    d_ndcg = (n_cw - n_std) / n_std * 100 if n_std else 0
    d_mrr  = (m_cw - m_std) / m_std * 100 if m_std else 0

    p(f"  {'Metric':<12} {'Standard':>14} {'CognitiveWeave':>16}  {'Δ':>8}")
    p(f"  {'─'*12} {'─'*14} {'─'*16}  {'─'*8}")
    p(f"  {'nDCG@5':<12} {n_std:>14.4f} {n_cw:>16.4f}  \033[32m{d_ndcg:>+7.1f}%\033[0m")
    p(f"  {'MRR':<12} {m_std:>14.4f} {m_cw:>16.4f}  \033[32m{d_mrr:>+7.1f}%\033[0m")
    p(f"  {'Noise@5':<12} {noise_std:>14} {noise_cw:>16}  \033[32m{'eliminated' if noise_cw == 0 else f'-{noise_std-noise_cw}':>8}\033[0m")
    p()
    p("  \033[1mKey insight:\033[0m  Standard retrieval returns whatever is \033[31mstructurally")
    p("  connected\033[0m, regardless of age. CognitiveWeave's Ebbinghaus decay")
    p("  automatically surfaces \033[32mfresh, relevant knowledge\033[0m and buries")
    p("  connections that have gone stale — without any manual curation.")
    p()
    p("\033[1;37m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")

    # Cleanup
    for nid in NOISE:
        neo4j.delete_node(nid)
    neo4j.close()


if __name__ == "__main__":
    main()
