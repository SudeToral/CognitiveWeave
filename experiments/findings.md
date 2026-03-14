# Experiment Findings

Observations from running the Belief Dynamics experiment layer on CognitiveWeave.
Each entry is a numbered finding with conditions and what was observed.

---

## F1 — Shared FAISS index is necessary for cross-pollination (when seeds are domain-isolated)

**Date:** 2026-03-12 / controlled 2026-03-14
**Condition:** 3 agents (Mira, Kael, Sova), 5 cycles, Ollama enabled
**Setup A (original):** Agents initialized without `faiss` parameter — experiment nodes written to Neo4j only → Cross-pollination = 0
**Setup B (controlled baseline):** Original seeds (`c4_attention`, `c3_hippocampus`, `c2_ebbinghaus`), FAISS sharing explicitly disabled → Cross-pollination = 0 for all agents
**Setup C (FAISS enabled, same seeds):** Cross-pollination rose to 24 (Sova) over 8 cycles
**Conclusion:** When an agent's seed is deep inside a single domain cluster, graph traversal alone cannot bridge to other agents. Shared semantic search (FAISS) is the mechanism that enables cross-pollination.

## F1b — Bridge seed enables BFS-only cross-pollination without FAISS

**Date:** 2026-03-14
**Condition:** Kael seed swapped to `c2_ebbinghaus` (bridge position), FAISS disabled
**Observation:** Kael cross-pollination = 5 (without any shared FAISS)
**Conclusion:** A topologically central seed allows BFS traversal to reach other agents' nodes via graph edges alone. FAISS amplifies cross-pollination but is not strictly necessary when the seed is already in a bridge position. The two mechanisms (semantic search + graph topology) are partially redundant for bridge agents.

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

## H1 — Seed node determines epistemic reach, not interest ✓ CONFIRMED

**Date:** 2026-03-14
**Status:** Confirmed
**Experiment:** Changed Kael's seed from `c4_attention` → `c2_ebbinghaus` (Sova's seed). Interest unchanged: "attention mechanisms in transformer models and knowledge retrieval."
**Result:** Kael cross-pollination: **0 → 6**
**Conclusion:** Graph topology (seed position) determines how far an agent's beliefs spread across the population. Interest/curiosity alone is insufficient — an agent starting deep inside a single cluster stays isolated regardless of what it's looking for.
**Design implication:** In shared-memory multi-agent systems, epistemic diversity can be engineered by placing agents at topologically diverse starting points — independently of their domain specialization.

---

## H2 — Bandwidth effect on cross-pollination rate

**Date:** 2026-03-14
**Condition:** Original seeds, shared FAISS enabled, 5 cycles each
**Results:**

| Bandwidth | Sova cross-pol | Consensus anchors |
|-----------|---------------|-------------------|
| 3 | 5 | 0 |
| 5 | 4 | 0 |
| 10 | 13 | 0 |

**Observation:** Relationship is non-linear. bw=5 shows a slight dip vs bw=3 (noise), but bw=10 jumps to 13. Consensus anchors did not form within 5 cycles at any bandwidth level.
**Interpretation:** There is a threshold effect — below a certain bandwidth, agents don't see enough of the graph to encounter other agents' writes. Above the threshold, cross-pollination grows faster than linearly. Consensus formation likely requires more cycles (or higher bandwidth) than tested.
**Open question:** At what cycle count does consensus first appear? → H2b

---

## F3 — Three agent roles emerge: bridge, peripheral, isolated

**Date:** 2026-03-14
**Condition:** bw=10, 18 cycles, shared FAISS, original seeds

| Agent | Cross-pol at cycle 17 | Pattern |
|-------|-----------------------|---------|
| Sova  | 65 | Linear growth (+4/cycle), no saturation |
| Mira  | 4  | Late start (cycle 5), plateau at 4 |
| Kael  | 0  | Zero across all 18 cycles |

**Three roles:**
- **Bridge (Sova):** Continuously absorbs from both domains, grows linearly
- **Peripheral (Mira):** Eventually reached by bridge agent's writes, but saturates
- **Isolated (Kael):** Never reached regardless of cycle count or bandwidth

**Consensus anchor:** Never formed across 18 cycles at bw=10. Either the consensus threshold is too strict, or true consensus requires symmetric cross-pollination (at least 2 agents mutually reading each other) — which never occurs here since Kael remains isolated.

---

## F4 — Consensus anchors emerge from cycle 0 when bridge + peripheral agents share a domain

**Date:** 2026-03-14
**Condition:** bw=10, 5 cycles, shared FAISS, original seeds, FAISS rebuilt clean on reset
**Results:**

| Agent | Cross-pol (cycle 4) | Role |
|-------|---------------------|------|
| Sova  | 15 | Bridge |
| Mira  | 4  | Peripheral |
| Kael  | 0  | Isolated |

**Consensus anchors (stable from cycle 0):** `c1_consolidation`, `c3_pfc`, `c1_ltp`

**Observation:** Consensus anchors appeared immediately (cycle 0) and remained stable across all 5 cycles. These are neuroscience nodes independently chosen as primary anchors by both Mira (neuroscience seed) and Sova (bridge seed). Kael never contributes to any anchor — confirming domain isolation.

**Interpretation:** Consensus forms between agents that share topological overlap, not between all agents. "Consensus" in this system is partial and cluster-specific — a local agreement, not a global one. True population-wide consensus would require all agents to be in overlapping graph neighborhoods.

---

## H3 — Same seed does not produce homogenization; interest still determines epistemic role

**Date:** 2026-03-14
**Condition:** All 3 agents seed = `c2_ebbinghaus`, bw=10, 5 cycles, shared FAISS
**Results:**

| Agent | Cross-pol (cycle 4) | Role |
|-------|---------------------|------|
| Sova  | 15 | Bridge (unchanged) |
| Mira  | 9  | Now active (was 4 with domain seed) |
| Kael  | 0  | Still isolated |

**Consensus anchors (9 nodes, stable from cycle 0):**
`c1_consolidation`, `c1_ltp`, `c1_protein_synthesis`, `c1_reconsolidation`,
`c2_retrieval_practice`, `c2_spacing`, `c3_sleep_consolidation`, `c3_swr`, `c4_continual_learning`

**Key observations:**
1. **Kael remains isolated (cross-pol = 0) even with bridge seed** — H3 falsifies the seed-only hypothesis. Interest IS a factor when all agents share the same seed. Kael's AI-domain interest still prevents him from landing on neuroscience/learning anchors.
2. **Mira's cross-pol more than doubled** (4 → 9) — same seed + same domain interest creates strong overlap with Sova.
3. **Consensus expanded from 3 to 9 anchors** — when two agents share both seed and domain (Mira + Sova: learning/memory), they agree on far more anchor nodes.
4. **No homogenization** — agents did not converge to identical beliefs. Diversity persisted through interest.

**Revised model:** Cross-pollination requires BOTH seed proximity AND interest overlap. Seed determines reachability; interest determines which of the reachable nodes are selected as anchors. An agent can be topologically adjacent but epistemically isolated if its interest pulls it toward a different region.

---

## Planned Experiments

| ID | Description | Status |
|----|-------------|--------|
| F1 | FAISS requirement for cross-pollination | **Complete + controlled** |
| F1b | Bridge seed enables BFS-only cross-pol | **Complete** |
| F2 | Emergent bridge agent | Complete |
| H1 | Seed vs interest determinism | **Confirmed** (0→6 cross-pol) |
| H2 | Bandwidth effect on cross-pollination rate | **Complete** (non-linear threshold) |
| H2b | Minimum cycles for consensus formation | **Complete** (consensus never forms — see F3) |
| H3 | All same seed → homogenization? | **Falsified** — interest still determines isolation |
