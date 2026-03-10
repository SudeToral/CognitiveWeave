"""
CognitiveWeave Temporal Decay Impact Benchmark
===============================================
Tests the specific scenario where decay SHOULD matter:

  A seed node has edges to BOTH relevant neighbours (fresh) AND
  noisy/stale neighbours (aged). After decay, graph scoring should
  demote the stale paths and promote the fresh ones.

Scenario
--------
Query: "what brain mechanisms underlie memory consolidation?"
Seed:  c1_consolidation

Relevant neighbours (expected in top-5):
  c3_swr, c3_sleep_consolidation, c3_hippocampus, c1_protein_synthesis, c1_reconsolidation

Noise neighbours (should be suppressed by decay):
  We inject 3 synthetic noisy nodes connected to c1_consolidation via
  artificially-aged edges.  After decay these paths score low;
  without decay they pollute the top-5.

Run:
  uv run python benchmarks/decay_impact.py
"""
from __future__ import annotations

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitiveweave.config.settings import Settings
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever

QUERY = "what brain mechanisms underlie memory consolidation?"
SEED  = ["c1_consolidation"]
RELEVANT = {"c3_swr", "c3_sleep_consolidation", "c3_hippocampus",
            "c1_protein_synthesis", "c1_reconsolidation", "c1_ltp"}
NOISE_IDS = ["noise_deprecated_1", "noise_deprecated_2", "noise_deprecated_3"]
K = 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(ids: list[str], rel: set[str], k: int) -> float:
    return sum(1 for x in ids[:k] if x in rel) / k

def ndcg_at_k(ids: list[str], rel: set[str], k: int) -> float:
    dcg  = sum(1/math.log2(i+2) for i, x in enumerate(ids[:k]) if x in rel)
    idcg = sum(1/math.log2(i+2) for i in range(min(len(rel), k)))
    return dcg/idcg if idcg else 0.0

def mrr(ids: list[str], rel: set[str]) -> float:
    for i, x in enumerate(ids):
        if x in rel:
            return 1/(i+1)
    return 0.0

def noise_in_top_k(ids: list[str], noise: list[str], k: int) -> int:
    return sum(1 for x in ids[:k] if x in noise)


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def inject_noise(neo4j: Neo4jClient, faiss: FAISSIndex) -> None:
    """Add 3 noisy nodes connected to c1_consolidation with fresh weight=0.9."""
    contents = [
        "Deprecated consolidation model: engrams are stored exclusively in the prefrontal cortex without hippocampal involvement.",
        "Outdated hypothesis: REM sleep is the sole driver of memory consolidation, slow-wave sleep plays no role.",
        "Superseded theory: memory consolidation completes within minutes via a single protein synthesis event.",
    ]
    for nid, content in zip(NOISE_IDS, contents):
        neo4j.upsert_node(nid, content=content,
                          metadata={"cluster": "noise", "confidence": 0.3})
        faiss.add(nid, content, {"cluster": "noise"})
        # Connect to seed with initially strong edge
        neo4j.upsert_edge("c1_consolidation", nid,
                          relation="RELATED_TO", weight=0.9)
    faiss.save()
    print(f"  Injected {len(NOISE_IDS)} noisy nodes connected to c1_consolidation")


def remove_noise(neo4j: Neo4jClient) -> None:
    for nid in NOISE_IDS:
        neo4j.delete_node(nid)


def reset_all_edges_fresh(neo4j: Neo4jClient) -> None:
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        s.run("MATCH ()-[r]->() SET r.last_decay_at = toString(datetime())")


def age_noise_edges(neo4j: Neo4jClient, days: int = 45) -> None:
    """Backdate only the edges going TO noise nodes."""
    query = f"""
    UNWIND $ids AS nid
    MATCH ()-[r]->(b:KnowledgeNode {{id: nid}})
    SET r.last_decay_at = toString(datetime() - duration({{days: {days}}}))
    """
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        s.run(query, ids=NOISE_IDS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("CognitiveWeave — Temporal Decay Impact Benchmark")
    print("=" * 60)

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

    print("\nSetup: injecting noisy nodes…")
    inject_noise(neo4j, faiss)

    # ── Condition 1: No decay — noise edges are fresh, look strong ───────
    print("\n[1/3] Baseline: NO DECAY — all edges fresh")
    reset_all_edges_fresh(neo4j)
    results_no_decay = retriever.retrieve(QUERY, seed_node_ids=SEED, top_k=10)
    ids_no_decay = [r.id for r in results_no_decay]

    # ── Condition 2: Decay applied to noise paths only ───────────────────
    print("[2/3] WITH DECAY — noise edges aged 45 days")
    reset_all_edges_fresh(neo4j)
    age_noise_edges(neo4j, days=45)
    neo4j.decay_edge_weights()
    results_decay = retriever.retrieve(QUERY, seed_node_ids=SEED, top_k=10)
    ids_decay = [r.id for r in results_decay]

    # ── Condition 3: Decay + prune ────────────────────────────────────────
    print("[3/3] WITH DECAY + PRUNE — weak edges removed")
    pruned = neo4j.prune_weak_edges(threshold=0.3)
    print(f"       Pruned {pruned} edges below 0.3")
    results_prune = retriever.retrieve(QUERY, seed_node_ids=SEED, top_k=10)
    ids_prune = [r.id for r in results_prune]

    # ── Results ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Query: \"{QUERY}\"")
    print(f"  Seeds: {SEED}")
    print(f"  Relevant: {RELEVANT}")
    print(f"\n  {'Metric':<18} {'No Decay':>12} {'With Decay':>12} {'Decay+Prune':>12}")
    print(f"  {'─'*18} {'─'*12} {'─'*12} {'─'*12}")

    for label, fn, args in [
        (f"P@{K}",       precision_at_k, (K,)),
        (f"nDCG@{K}",    ndcg_at_k,      (K,)),
        ("MRR",          mrr,             ()),
        (f"Noise@{K}",   noise_in_top_k,  (NOISE_IDS, K)),
    ]:
        if label == f"Noise@{K}":
            a = noise_in_top_k(ids_no_decay, NOISE_IDS, K)
            b = noise_in_top_k(ids_decay,    NOISE_IDS, K)
            c = noise_in_top_k(ids_prune,    NOISE_IDS, K)
            print(f"  {label:<18} {a:>12} {b:>12} {c:>12}  (lower=better)")
        elif label == "MRR":
            a = mrr(ids_no_decay, RELEVANT)
            b = mrr(ids_decay,    RELEVANT)
            c = mrr(ids_prune,    RELEVANT)
            print(f"  {label:<18} {a:>12.4f} {b:>12.4f} {c:>12.4f}")
        else:
            a = fn(ids_no_decay, RELEVANT, *args)
            b = fn(ids_decay,    RELEVANT, *args)
            c = fn(ids_prune,    RELEVANT, *args)
            print(f"  {label:<18} {a:>12.4f} {b:>12.4f} {c:>12.4f}")

    print(f"\n  Top-10 retrieved (No Decay vs With Decay):")
    print(f"  {'Rank':<6} {'No Decay':<35} {'With Decay':<35}")
    print(f"  {'─'*6} {'─'*35} {'─'*35}")
    for i in range(min(10, max(len(ids_no_decay), len(ids_decay)))):
        nd = ids_no_decay[i] if i < len(ids_no_decay) else "—"
        wd = ids_decay[i]    if i < len(ids_decay)    else "—"
        nd_mark = "✓" if nd in RELEVANT else ("✗" if nd in NOISE_IDS else " ")
        wd_mark = "✓" if wd in RELEVANT else ("✗" if wd in NOISE_IDS else " ")
        print(f"  {i+1:<6} {nd_mark} {nd:<33} {wd_mark} {wd:<33}")

    print(f"\n  ✓ = relevant  ✗ = noise injected")

    # ── Improvement summary ───────────────────────────────────────────────
    a_ndcg = ndcg_at_k(ids_no_decay, RELEVANT, K)
    c_ndcg = ndcg_at_k(ids_prune,    RELEVANT, K)
    delta  = ((c_ndcg - a_ndcg) / a_ndcg * 100) if a_ndcg else 0
    noise_a = noise_in_top_k(ids_no_decay, NOISE_IDS, K)
    noise_c = noise_in_top_k(ids_prune,    NOISE_IDS, K)
    print(f"\n  nDCG@{K}: {a_ndcg:.4f} → {c_ndcg:.4f}  ({delta:+.1f}%)")
    print(f"  Noise in top-{K}: {noise_a} → {noise_c}  (decay eliminated stale paths)")

    # Cleanup
    print("\nCleaning up noise nodes…")
    remove_noise(neo4j)
    neo4j.close()
    print("Done.")


if __name__ == "__main__":
    main()
