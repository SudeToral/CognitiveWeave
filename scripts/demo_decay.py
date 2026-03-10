"""
Temporal Edge Decay demo.

Steps:
  1. Show current edge weights (baseline)
  2. Backdate a subset of edges to simulate aging (7, 14, 30 days old)
  3. Run Ebbinghaus decay
  4. Show before/after comparison
  5. Run again to show prune threshold in action

Run:
  uv run python scripts/demo_decay.py
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitiveweave.config.settings import Settings
from cognitiveweave.storage.neo4j_client import Neo4jClient


def show_weights(neo4j: Neo4jClient, label: str, limit: int = 15) -> None:
    query = """
    MATCH (a:KnowledgeNode)-[r]->(b:KnowledgeNode)
    RETURN a.id AS src, type(r) AS rel, b.id AS tgt,
           r.weight AS weight, r.stability AS stability,
           r.last_decay_at AS last_decay_at
    ORDER BY r.weight ASC
    LIMIT $limit
    """
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        rows = list(s.run(query, limit=limit))

    print(f"\n{'─'*70}")
    print(f"  {label}")
    print(f"{'─'*70}")
    print(f"  {'SRC':<25} {'REL':<28} {'TGT':<25} {'W':>6} {'STAB':>5}")
    print(f"  {'─'*25} {'─'*28} {'─'*25} {'─'*6} {'─'*5}")
    for r in rows:
        w = r["weight"]
        stab = r["stability"]
        print(f"  {str(r['src']):<25} {str(r['rel']):<28} {str(r['tgt']):<25} {w:>6.3f} {stab:>5.1f}")


def backdate_edges(neo4j: Neo4jClient, aging_map: dict[int, list[str]]) -> None:
    """Set last_decay_at to N days ago for edges whose source node is in the list."""
    for days_ago, node_ids in aging_map.items():
        query = f"""
        UNWIND $ids AS nid
        MATCH (a:KnowledgeNode {{id: nid}})-[r]->()
        SET r.last_decay_at = toString(datetime() - duration({{days: {days_ago}}}))
        """
        with neo4j._driver.session(database=neo4j._settings.database) as s:
            s.run(query, ids=node_ids)
        print(f"  Backdated edges from {node_ids} → {days_ago} days old")


def main() -> None:
    settings = Settings()
    neo4j = Neo4jClient(settings.neo4j)
    neo4j.connect()

    # ── Baseline ────────────────────────────────────────────────────────
    show_weights(neo4j, "BEFORE DECAY — lowest weight edges (freshly created, all ~1.0)")

    total_edges_q = "MATCH ()-[r]->() RETURN count(r) AS c"
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        total = s.run(total_edges_q).single()["c"]
    print(f"\n  Total edges: {total}")

    # ── Simulate aging ──────────────────────────────────────────────────
    print("\n  Simulating edge aging…")
    backdate_edges(neo4j, {
        # 30 days old — cross-cluster weak bridges (should decay significantly)
        30: ["c4_rrf", "c5_holographic", "c6_dual_process", "c6_situated"],
        # 14 days old — moderate links
        14: ["c2_priming", "c3_basal_ganglia", "c5_conceptual_spaces", "c4_memory_augmented"],
        # 7 days old — recently used links
        7:  ["c2_ebbinghaus", "c3_theta", "c1_synaptic_tagging", "c6_transfer"],
    })

    # ── First decay run ─────────────────────────────────────────────────
    print("\n  Running decay cycle 1…")
    updated = neo4j.decay_edge_weights()
    print(f"  → {updated} edges updated")

    show_weights(neo4j, "AFTER DECAY — edges now reflect Ebbinghaus forgetting (lowest first)")

    # Show the most decayed edges explicitly
    query_most_decayed = """
    MATCH (a:KnowledgeNode)-[r]->(b:KnowledgeNode)
    WHERE r.weight < 0.85
    RETURN a.id AS src, type(r) AS rel, b.id AS tgt,
           r.weight AS weight, r.stability AS stability
    ORDER BY r.weight ASC
    """
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        decayed = list(s.run(query_most_decayed))

    if decayed:
        print(f"\n  {'─'*70}")
        print(f"  DECAYED EDGES (weight < 0.85) — {len(decayed)} edges")
        print(f"  {'─'*70}")
        print(f"  {'SRC':<28} → {'REL':<30} {'W':>7}")
        for r in decayed:
            print(f"  {str(r['src']):<28} → {str(r['rel']):<30} {r['weight']:>7.4f}")
    else:
        print("\n  No edges below 0.85 threshold yet.")

    # ── Prune simulation ─────────────────────────────────────────────────
    pruned = neo4j.prune_weak_edges(threshold=0.3)
    print(f"\n  Prune (threshold=0.3): {pruned} edges deleted")

    with neo4j._driver.session(database=neo4j._settings.database) as s:
        remaining = s.run(total_edges_q).single()["c"]
    print(f"  Remaining edges: {remaining} / {total}")

    # ── Spaced repetition boost demo ─────────────────────────────────────
    print("\n  Running structural search from ['c1_consolidation'] → boosts traversed edges…")
    results = neo4j.structural_search(["c1_consolidation"], max_hops=2, limit=8)
    print(f"  Found {len(results)} neighbours via BFS")
    for r in results[:5]:
        print(f"    score={r['score']:.3f}  id={r['id']}")

    print("\n  Check stability boost on paths traversed from c1_consolidation:")
    query_boost = """
    MATCH (a:KnowledgeNode {id: 'c1_consolidation'})-[r]->(b:KnowledgeNode)
    RETURN b.id AS tgt, r.stability AS stability, r.weight AS weight
    ORDER BY r.stability DESC
    """
    with neo4j._driver.session(database=neo4j._settings.database) as s:
        boost_rows = list(s.run(query_boost))
    for r in boost_rows:
        print(f"    → {r['tgt']:<30} stability={r['stability']:.2f}  weight={r['weight']:.4f}")

    print("\n  Done.")
    neo4j.close()


if __name__ == "__main__":
    main()
