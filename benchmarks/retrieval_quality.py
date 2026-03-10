"""
CognitiveWeave Retrieval Quality Benchmark
==========================================
Compares three retrieval strategies on the seeded cognitive science graph:

  A) Pure FAISS          — dense vector similarity only
  B) Hybrid RRF          — FAISS + Neo4j graph, NO temporal decay
  C) CognitiveWeave RRF  — FAISS + Neo4j graph, WITH temporal decay

Metrics: Precision@K, Recall@K, nDCG@K, MRR  (K=5)

Run:
  uv run python benchmarks/retrieval_quality.py
"""
from __future__ import annotations

import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitiveweave.config.settings import Settings
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Ground truth: for each query, which node_ids are genuinely relevant?
# (defined by domain experts, i.e. us)
# ---------------------------------------------------------------------------
QUERIES: list[dict] = [
    {
        "query": "How does synaptic strengthening happen at the molecular level?",
        "relevant": {"c1_ltp", "c1_nmda", "c1_camkii", "c1_ampa", "c1_hebbian", "c1_bdnf"},
        "seeds": None,
        "topic": "Molecular LTP",
    },
    {
        "query": "What happens in the brain during sleep to consolidate memories?",
        "relevant": {"c3_sleep_consolidation", "c3_swr", "c1_consolidation", "c3_hippocampus", "c3_theta"},
        "seeds": ["c3_sleep_consolidation"],
        "topic": "Sleep Consolidation",
    },
    {
        "query": "What study strategies improve long-term retention?",
        "relevant": {"c2_spacing", "c2_retrieval_practice", "c2_desirable_difficulty",
                     "c2_interleaving", "c2_elaborative", "c2_ebbinghaus"},
        "seeds": ["c2_spacing"],
        "topic": "Learning Strategies",
    },
    {
        "query": "How does the hippocampus support episodic memory?",
        "relevant": {"c3_hippocampus", "c3_ca1", "c3_ca3", "c3_dentate_gyrus",
                     "c2_episodic", "c1_engram", "c1_place_cells"},
        "seeds": ["c3_hippocampus"],
        "topic": "Hippocampus & Episodic",
    },
    {
        "query": "How does attention mechanism work in transformer models?",
        "relevant": {"c4_attention", "c4_transformer", "c4_kv_cache",
                     "c4_embedding", "c4_hopfield"},
        "seeds": None,
        "topic": "Transformer Attention",
    },
    {
        "query": "What is retrieval augmented generation and how does it work?",
        "relevant": {"c4_rag", "c4_vector_store", "c4_faiss", "c4_embedding",
                     "c4_graph_rag", "c4_rrf"},
        "seeds": ["c4_rag"],
        "topic": "RAG & Vector Retrieval",
    },
    {
        "query": "How do knowledge graphs differ from vector databases?",
        "relevant": {"c4_knowledge_graph", "c4_vector_store", "c4_faiss",
                     "c5_semantic_net", "c5_ontology", "c5_cypher"},
        "seeds": ["c4_knowledge_graph"],
        "topic": "Knowledge Graphs vs Vectors",
    },
    {
        "query": "What limits working memory capacity?",
        "relevant": {"c2_working_memory", "c2_phonological_loop", "c2_visuospatial",
                     "c2_central_executive", "c6_chunking", "c6_cognitive_load"},
        "seeds": ["c2_working_memory"],
        "topic": "Working Memory Limits",
    },
    {
        "query": "How does the brain distinguish similar memories to avoid confusion?",
        "relevant": {"c3_dentate_gyrus", "c3_ca3", "c3_ca1", "c2_source_monitoring",
                     "c1_engram", "c3_hippocampus"},
        "seeds": ["c3_dentate_gyrus"],
        "topic": "Pattern Separation",
    },
    {
        "query": "What is the Ebbinghaus forgetting curve and why does it matter?",
        "relevant": {"c2_ebbinghaus", "c2_spacing", "c2_retrieval_practice",
                     "c1_consolidation", "c2_desirable_difficulty"},
        "seeds": None,
        "topic": "Forgetting Curve",
    },
]

K = 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, r in enumerate(retrieved[:k]) if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0


def score_results(retrieved_ids: list[str], relevant: set[str]) -> dict:
    return {
        f"P@{K}":    round(precision_at_k(retrieved_ids, relevant, K), 4),
        f"R@{K}":    round(recall_at_k(retrieved_ids, relevant, K), 4),
        f"nDCG@{K}": round(ndcg_at_k(retrieved_ids, relevant, K), 4),
        "MRR":       round(mrr(retrieved_ids, relevant), 4),
    }


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def faiss_only(faiss: FAISSIndex, query: str, k: int) -> list[str]:
    """Pure FAISS — no graph, no decay."""
    results = faiss.search(query, top_k=k)
    return [r["id"] for r in results]


def simulate_decay_scenario(neo4j: Neo4jClient) -> None:
    """
    Simulate a realistic temporal scenario:
    - 'Core' memory/neuroscience edges are fresh (recently studied)
    - 'Peripheral' cross-domain edges (weak analogies) are aged 21 days
    This represents a researcher who has been actively working on neuroscience
    but hasn't revisited the AI-philosophy bridges in 3 weeks.
    """
    stale_sources = [
        "c5_holographic", "c5_conceptual_spaces", "c4_rrf",
        "c6_situated", "c6_dual_process", "c4_memory_augmented",
        "c6_transfer", "c5_frame",
    ]
    query = """
    UNWIND $ids AS nid
    MATCH (a:KnowledgeNode {id: nid})-[r]->()
    SET r.last_decay_at = toString(datetime() - duration({days: 21}))
    """
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        s.run(query, ids=stale_sources)

    neo4j.decay_edge_weights()


def reset_decay(neo4j: Neo4jClient) -> None:
    """Reset all edges to fresh (Δt = 0) for the no-decay baseline."""
    query = """
    MATCH ()-[r]->()
    SET r.last_decay_at = toString(datetime()),
        r.weight = coalesce(r.weight, 1.0)
    """
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        s.run(query)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    retriever: HybridRetriever,
    faiss: FAISSIndex,
    label: str,
    use_seeds: bool,
) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {f"P@{K}": [], f"R@{K}": [], f"nDCG@{K}": [], "MRR": []}

    for q in QUERIES:
        seeds = q["seeds"] if use_seeds else None
        retrieved = retriever.retrieve(q["query"], seed_node_ids=seeds, top_k=15)
        ids = [r.id for r in retrieved]
        s = score_results(ids, q["relevant"])
        for metric, val in s.items():
            scores[metric].append(val)

    return scores


def avg(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def print_table(results: dict[str, dict[str, list[float]]]) -> None:
    metrics = [f"P@{K}", f"R@{K}", f"nDCG@{K}", "MRR"]
    col_w = 14

    header = f"  {'System':<28}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(f"\n{'─' * (28 + col_w * len(metrics) + 2)}")
    print(header)
    print(f"{'─' * (28 + col_w * len(metrics) + 2)}")

    for system, scores in results.items():
        row = f"  {system:<28}" + "".join(f"{avg(scores[m]):>{col_w}.4f}" for m in metrics)
        print(row)

    print(f"{'─' * (28 + col_w * len(metrics) + 2)}")


def print_per_query(results: dict[str, dict[str, list[float]]]) -> None:
    metric = f"nDCG@{K}"
    print(f"\n  Per-query {metric}:")
    print(f"  {'Topic':<32} {'FAISS':>8} {'Hybrid':>8} {'CogWeave':>10}")
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*10}")
    systems = list(results.keys())
    for i, q in enumerate(QUERIES):
        vals = [f"{results[s][metric][i]:>8.4f}" for s in systems]
        print(f"  {q['topic']:<32} {vals[0]} {vals[1]} {vals[2]}")


def main() -> None:
    print("CognitiveWeave Retrieval Quality Benchmark")
    print("=" * 60)

    settings = Settings()

    print("\nConnecting…")
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

    # ── System A: Pure FAISS ─────────────────────────────────────────────
    print("\n[1/3] Running System A — Pure FAISS…")
    scores_a: dict[str, list[float]] = {f"P@{K}": [], f"R@{K}": [], f"nDCG@{K}": [], "MRR": []}
    for q in QUERIES:
        ids = faiss_only(faiss, q["query"], k=15)
        s = score_results(ids, q["relevant"])
        for metric, val in s.items():
            scores_a[metric].append(val)

    # ── System B: Hybrid RRF, no decay (fresh timestamps) ───────────────
    print("[2/3] Running System B — Hybrid RRF (no decay)…")
    reset_decay(neo4j)
    scores_b = run_benchmark(retriever, faiss, "Hybrid RRF (no decay)", use_seeds=True)

    # ── System C: CognitiveWeave — hybrid + decay ────────────────────────
    print("[3/3] Running System C — CognitiveWeave (with decay)…")
    simulate_decay_scenario(neo4j)
    scores_c = run_benchmark(retriever, faiss, "CognitiveWeave", use_seeds=True)

    # ── Results ──────────────────────────────────────────────────────────
    all_results = {
        "A  Pure FAISS":           scores_a,
        "B  Hybrid RRF (no decay)": scores_b,
        "C  CognitiveWeave":       scores_c,
    }

    print("\n\n" + "=" * 60)
    print("  RESULTS — averaged over 10 queries")
    print_table(all_results)
    print_per_query(all_results)

    # Improvement summary
    print("\n  IMPROVEMENT: A → C")
    for metric in [f"P@{K}", f"nDCG@{K}", "MRR"]:
        a = avg(scores_a[metric])
        c = avg(scores_c[metric])
        delta = ((c - a) / a * 100) if a > 0 else 0
        arrow = "▲" if delta > 0 else "▼"
        print(f"    {metric:<10} {a:.4f} → {c:.4f}   {arrow} {abs(delta):.1f}%")

    print("\n  IMPROVEMENT: B → C (decay effect only)")
    for metric in [f"P@{K}", f"nDCG@{K}", "MRR"]:
        b = avg(scores_b[metric])
        c = avg(scores_c[metric])
        delta = ((c - b) / b * 100) if b > 0 else 0
        arrow = "▲" if delta > 0 else "▼"
        print(f"    {metric:<10} {b:.4f} → {c:.4f}   {arrow} {abs(delta):.1f}%")

    neo4j.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
