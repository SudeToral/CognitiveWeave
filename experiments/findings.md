# Experiment Findings

Observations from running the Belief Dynamics experiment layer on CognitiveWeave.
Each entry is a numbered finding with conditions and what was observed.

---

## F1 — Shared FAISS index is necessary for cross-pollination

**Date:** 2026-03-12
**Condition:** 3 agents (Mira, Kael, Sova), 5 cycles, Ollama enabled
**Setup:** Agents initialized *without* `faiss` parameter — experiment nodes were written to Neo4j only
**Observation:** Cross-pollination = 0 for all agents across all cycles
**Root cause:** Semantic search (FAISS) could not find any experiment nodes because they were never added to the index. BFS traversal alone was insufficient to bridge agents' separate graph neighborhoods.
**Fix:** Pass shared `FAISSIndex` to `SocietyAgent._write()` — adds each belief to FAISS immediately after writing.
**Result after fix:** Sova reached 24 cross-pollinations over 8 cycles.

---

## F2 — Emergent bridge agent from seed node position

**Date:** 2026-03-12
**Condition:** 3 agents, 8 cycles, Ollama enabled, shared FAISS
**Agent configuration:**

| Agent | Interest | Seed node | Domain |
|-------|----------|-----------|--------|
| Mira  | memory consolidation, hippocampus | `c3_hippocampus` | Neuroscience |
| Kael  | transformer attention, knowledge retrieval | `c4_attention` | AI/ML |
| Sova  | forgetting curves, long-term retention | `c2_ebbinghaus` | Learning Science |

**Observation:**
- Sova: 24 cross-pollinations, highest belief drift (~0.40 cosine distance from cycle-0)
- Mira: moderate cross-pollination, moderate drift (~0.30)
- Kael: near-zero cross-pollination, minimal drift (~0.05) — **echo chamber**

**Interpretation:** Sova's seed `c2_ebbinghaus` sits topologically between neuroscience and AI clusters. BFS from that node reaches both, making Sova a structural bridge. Kael's seed `c4_attention` is deep inside the AI cluster — BFS stays there.

**Open question:** Is Kael's echo chamber behavior caused by the seed node or by the interest topic? → See Hypothesis H1.

---

## H1 — Hypothesis: Seed node determines epistemic reach, not interest

**Status:** Untested
**Experiment:** Change Kael's seed from `c4_attention` to `c2_ebbinghaus` (Sova's seed), keep interest unchanged.
**Prediction:** If seed is the determining factor, Kael's cross-pollination will rise. If interest is the determining factor, Kael will remain isolated.
**Significance:** If confirmed — AI multi-agent system designers can control epistemic diversity by choosing agent starting positions in the knowledge graph, independently of what each agent is "curious about."

---

## Planned Experiments

| ID | Description | Status |
|----|-------------|--------|
| F1 | FAISS requirement for cross-pollination | Complete |
| F2 | Emergent bridge agent | Complete |
| H1 | Seed vs interest determinism | Pending |
| H2 | Bandwidth effect on consensus formation speed | Not started |
| H3 | Population size vs belief convergence rate | Not started |
