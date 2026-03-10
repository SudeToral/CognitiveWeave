"""
Seed CognitiveWeave with a rich Cognitive Science + AI knowledge graph.

Domains:
  C1 - Memory Biology      (synaptic, molecular, cellular)
  C2 - Memory Psychology   (forgetting, learning strategies)
  C3 - Neural Systems      (brain regions, oscillations)
  C4 - AI / ML Parallels   (transformers, RAG, vector stores)
  C5 - Knowledge Repr.     (ontologies, semantic nets, graphs)
  C6 - Cognitive Science   (load, metacognition, dual-process)

Run:
  cd /Users/sudetoral/Desktop/CognitiveWeave
  uv run python scripts/seed_knowledge.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cognitiveweave.config.settings import Settings
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.storage.faiss_index import FAISSIndex

# ---------------------------------------------------------------------------
# Node definitions: (id, content, cluster, confidence)
# ---------------------------------------------------------------------------
NODES: list[tuple[str, str, str, float]] = [
    # ── C1 · Memory Biology ─────────────────────────────────────────────
    ("c1_ltp", "Long-term potentiation (LTP) is a persistent increase in synaptic strength following high-frequency stimulation, considered the cellular basis of learning and memory.", "memory_biology", 0.95),
    ("c1_ltd", "Long-term depression (LTD) is a sustained decrease in synaptic efficacy, providing the counter-mechanism to LTP and enabling synaptic refinement.", "memory_biology", 0.90),
    ("c1_hebbian", "Hebbian plasticity encapsulates the rule 'neurons that fire together wire together', describing activity-dependent synaptic strengthening.", "memory_biology", 0.92),
    ("c1_nmda", "NMDA receptors act as coincidence detectors requiring both pre- and postsynaptic activity, making them the molecular gate for LTP induction.", "memory_biology", 0.93),
    ("c1_ampa", "AMPA receptor trafficking — insertion during LTP, removal during LTD — is the primary mechanism for rapidly changing synaptic strength.", "memory_biology", 0.91),
    ("c1_camkii", "CaMKII (calcium/calmodulin-dependent protein kinase II) autophosphorylates after LTP induction and maintains synaptic potentiation long-term.", "memory_biology", 0.88),
    ("c1_synaptic_tagging", "Synaptic tagging and capture: a briefly activated synapse sets a molecular 'tag' that captures plasticity-related proteins synthesized elsewhere in the neuron.", "memory_biology", 0.85),
    ("c1_consolidation", "Memory consolidation is the process by which newly encoded memories are stabilised — first at the synaptic level, then through systems-level reorganisation during sleep.", "memory_biology", 0.94),
    ("c1_reconsolidation", "Memory reconsolidation occurs when a retrieved memory becomes transiently labile and must be restabilised, creating a window for memory updating or erasure.", "memory_biology", 0.87),
    ("c1_bdnf", "Brain-derived neurotrophic factor (BDNF) promotes synaptic growth and is required for the late phase of LTP, linking neurotrophin signalling to long-term memory.", "memory_biology", 0.86),
    ("c1_protein_synthesis", "Late-phase LTP requires new protein synthesis; blocking it within hours of learning impairs long-term but not short-term memory.", "memory_biology", 0.84),
    ("c1_engram", "An engram is the physical substrate of a memory — a sparse ensemble of neurons whose reactivation is sufficient to trigger recall of the original experience.", "memory_biology", 0.89),
    ("c1_place_cells", "Place cells in the hippocampal CA1/CA3 fire selectively when an animal occupies specific spatial locations, providing a neural map of the environment.", "memory_biology", 0.87),
    ("c1_grid_cells", "Grid cells in the entorhinal cortex fire in a hexagonal lattice pattern, providing a metric coordinate system that interfaces with hippocampal place cells.", "memory_biology", 0.86),

    # ── C2 · Memory Psychology ───────────────────────────────────────────
    ("c2_ebbinghaus", "Ebbinghaus forgetting curve shows that memory retention decays exponentially after initial learning, with most forgetting occurring within the first 24 hours.", "memory_psychology", 0.96),
    ("c2_spacing", "The spacing effect: distributing practice sessions across time yields far better long-term retention than massing the same study time into a single session.", "memory_psychology", 0.95),
    ("c2_retrieval_practice", "Retrieval practice (the testing effect) shows that actively recalling information strengthens memory more than re-reading or passive review.", "memory_psychology", 0.94),
    ("c2_interleaving", "Interleaved practice — alternating between different problem types or subjects — improves long-term learning and discrimination despite feeling harder during study.", "memory_psychology", 0.90),
    ("c2_elaborative", "Elaborative interrogation prompts learners to generate 'why' and 'how' explanations, activating prior knowledge and forming richer memory traces.", "memory_psychology", 0.88),
    ("c2_dual_coding", "Dual coding theory (Paivio) proposes that verbal and visual information are encoded in separate but interconnected systems, and using both together enhances memory.", "memory_psychology", 0.91),
    ("c2_working_memory", "Working memory (Baddeley & Hitch model) is the limited-capacity system for temporarily holding and manipulating information during cognitive tasks.", "memory_psychology", 0.95),
    ("c2_phonological_loop", "The phonological loop stores sound-based information for ~2 seconds and refreshes it via inner speech, explaining word-length and phonological similarity effects.", "memory_psychology", 0.89),
    ("c2_visuospatial", "The visuospatial sketchpad temporarily maintains visual and spatial information, supporting tasks such as mental rotation and navigation imagery.", "memory_psychology", 0.88),
    ("c2_central_executive", "The central executive coordinates the slave systems of working memory, controls attention, and switches between tasks — associated with prefrontal cortex function.", "memory_psychology", 0.90),
    ("c2_episodic", "Episodic memory stores personally experienced events with their temporal and spatial context, allowing mental time travel to past experiences.", "memory_psychology", 0.94),
    ("c2_semantic", "Semantic memory holds general world knowledge and facts independent of personal experience — the 'knowing that' system.", "memory_psychology", 0.93),
    ("c2_procedural", "Procedural memory underlies skill acquisition and habit formation; it is implicit, robust to amnesia, and depends on striatum and cerebellum rather than hippocampus.", "memory_psychology", 0.92),
    ("c2_priming", "Priming is an implicit memory effect where exposure to a stimulus influences response to a subsequent related stimulus, often without conscious awareness.", "memory_psychology", 0.89),
    ("c2_source_monitoring", "Source monitoring is the cognitive process of attributing memories to their origins; failures cause confusions such as cryptomnesia and false memories.", "memory_psychology", 0.85),
    ("c2_desirable_difficulty", "Desirable difficulties are learning conditions that slow apparent progress but enhance long-term retention and transfer (e.g., retrieval practice, interleaving, spacing).", "memory_psychology", 0.91),

    # ── C3 · Neural Systems ───────────────────────────────────────────────
    ("c3_hippocampus", "The hippocampus is critical for forming new episodic and semantic memories; damage causes anterograde amnesia while leaving procedural memory intact.", "neural_systems", 0.96),
    ("c3_ca1", "Hippocampal CA1 receives processed information from CA3 via Schaffer collaterals and from entorhinal cortex directly, acting as the main output stage for memory encoding.", "neural_systems", 0.90),
    ("c3_ca3", "Hippocampal CA3 contains recurrent connections enabling pattern completion — the reconstruction of a full memory from a partial cue.", "neural_systems", 0.89),
    ("c3_dentate_gyrus", "The dentate gyrus performs pattern separation, transforming similar inputs into distinct neural representations to minimise interference between memories.", "neural_systems", 0.88),
    ("c3_entorhinal", "The entorhinal cortex is the primary interface between hippocampus and neocortex, relaying sensory and associative information into hippocampal circuits.", "neural_systems", 0.87),
    ("c3_pfc", "The prefrontal cortex supports working memory maintenance, strategic retrieval, and source monitoring; it modulates hippocampal activity during encoding and recall.", "neural_systems", 0.93),
    ("c3_amygdala", "The amygdala enhances consolidation of emotionally significant memories by modulating norepinephrine release in the hippocampus, explaining flashbulb memory intensity.", "neural_systems", 0.92),
    ("c3_basal_ganglia", "The basal ganglia implement reward-based procedural learning and habit formation through dopamine-gated synaptic plasticity in the striatum.", "neural_systems", 0.91),
    ("c3_cerebellum", "The cerebellum underlies motor learning and timing through long-term depression at parallel fibre–Purkinje cell synapses driven by climbing fibre error signals.", "neural_systems", 0.90),
    ("c3_theta", "Hippocampal theta oscillations (4–8 Hz) coordinate encoding and retrieval phases: encoding is favoured at theta peaks, retrieval at troughs.", "neural_systems", 0.87),
    ("c3_swr", "Sharp-wave ripples (SWRs, ~80–100 Hz) during slow-wave sleep replay waking experience, driving hippocampal-to-cortical transfer and systems consolidation.", "neural_systems", 0.88),
    ("c3_gamma", "Gamma oscillations (30–80 Hz) support local cortical computation and binding; hippocampal gamma couples with theta to organise sequential memory encoding.", "neural_systems", 0.85),
    ("c3_dmn", "The default mode network (medial PFC, posterior cingulate, angular gyrus) activates during memory retrieval, future simulation, and self-referential thought.", "neural_systems", 0.89),
    ("c3_sleep_consolidation", "During NREM slow-wave sleep, sharp-wave ripples replay memories and synchronise with cortical slow oscillations and thalamic spindles to consolidate declarative memory.", "neural_systems", 0.92),

    # ── C4 · AI / ML Parallels ────────────────────────────────────────────
    ("c4_transformer", "Transformer architecture uses self-attention to compute context-dependent representations, replacing recurrence with parallel processing of sequences.", "ai_ml", 0.96),
    ("c4_attention", "Scaled dot-product attention computes weighted sums of values using query-key similarity scores, allowing models to selectively focus on relevant input positions.", "ai_ml", 0.95),
    ("c4_kv_cache", "Key-value cache in transformer inference stores computed attention states, enabling efficient autoregressive generation without recomputing past tokens.", "ai_ml", 0.90),
    ("c4_rag", "Retrieval-Augmented Generation (RAG) augments language model generation by retrieving relevant documents from an external store, grounding responses in factual knowledge.", "ai_ml", 0.95),
    ("c4_vector_store", "Vector stores index dense embeddings for approximate nearest-neighbour search, forming the retrieval backbone of RAG and semantic search systems.", "ai_ml", 0.93),
    ("c4_faiss", "FAISS (Facebook AI Similarity Search) provides efficient exact and approximate nearest-neighbour search over billions of vectors using IVF, HNSW and PQ indexing.", "ai_ml", 0.92),
    ("c4_embedding", "Dense vector embeddings map text (or other modalities) into a continuous semantic space where cosine similarity approximates semantic relatedness.", "ai_ml", 0.94),
    ("c4_knowledge_graph", "Knowledge graphs represent entities as nodes and relations as typed edges, enabling structured reasoning and graph traversal beyond what flat vector search provides.", "ai_ml", 0.93),
    ("c4_hopfield", "Modern Hopfield networks (Ramsauer et al., 2020) reformulate associative memory with exponential storage capacity and show formal equivalence to self-attention.", "ai_ml", 0.88),
    ("c4_in_context", "In-context learning allows large language models to acquire new behaviours from examples in the prompt without gradient updates, analogous to fast episodic binding.", "ai_ml", 0.91),
    ("c4_continual_learning", "Continual learning studies how neural networks can acquire new knowledge without catastrophic forgetting of old knowledge — the stability-plasticity dilemma.", "ai_ml", 0.90),
    ("c4_rrf", "Reciprocal Rank Fusion (RRF) combines ranked lists from multiple retrieval systems (e.g., dense vector + sparse BM25 + graph) into a single fused ranking.", "ai_ml", 0.87),
    ("c4_graph_rag", "Graph RAG extends standard RAG by traversing knowledge-graph neighbourhoods around retrieved entities, enriching context with structured relational information.", "ai_ml", 0.89),
    ("c4_memory_augmented", "Memory-augmented neural networks (e.g., Neural Turing Machine, Differentiable Neural Computer) equip models with an external addressable memory matrix.", "ai_ml", 0.86),

    # ── C5 · Knowledge Representation ────────────────────────────────────
    ("c5_semantic_net", "Semantic networks represent knowledge as labelled graphs of concepts and relations, originating from Collins & Quillian's spreading-activation model of memory.", "knowledge_repr", 0.91),
    ("c5_spreading_activation", "Spreading activation in semantic networks explains priming: activating one node propagates excitation to related nodes, lowering their retrieval threshold.", "knowledge_repr", 0.90),
    ("c5_ontology", "Formal ontologies define concepts, categories, and their relations in a machine-interpretable way — the backbone of the Semantic Web and knowledge engineering.", "knowledge_repr", 0.89),
    ("c5_schema", "Schema theory (Bartlett) proposes that knowledge is organised in structured frameworks that guide encoding, storage, and reconstruction of new information.", "knowledge_repr", 0.88),
    ("c5_conceptual_spaces", "Conceptual spaces (Gärdenfors) represent concepts as convex regions in a geometric quality space, bridging symbolic and sub-symbolic representations.", "knowledge_repr", 0.83),
    ("c5_holographic", "Holographic reduced representations (Plate) encode structured compositional knowledge in fixed-width distributed vectors using circular convolution.", "knowledge_repr", 0.80),
    ("c5_frame", "Frames (Minsky) are structured representations for stereotyped situations with slots for typical attribute values, precursors to modern ontology classes.", "knowledge_repr", 0.85),
    ("c5_rdf", "RDF (Resource Description Framework) represents knowledge as subject–predicate–object triples, enabling machine-readable linked data on the Web.", "knowledge_repr", 0.84),
    ("c5_sparql", "SPARQL is the query language for RDF knowledge graphs, analogous to Cypher for property graphs like Neo4j.", "knowledge_repr", 0.82),
    ("c5_cypher", "Cypher is Neo4j's declarative graph query language; MATCH patterns express structural traversals that are impossible in flat SQL or vector retrieval alone.", "knowledge_repr", 0.86),

    # ── C6 · Cognitive Science ────────────────────────────────────────────
    ("c6_cognitive_load", "Cognitive load theory (Sweller) distinguishes intrinsic, extraneous, and germane load; effective instruction minimises extraneous load to free resources for schema formation.", "cognitive_science", 0.93),
    ("c6_dual_process", "Dual-process theory (Kahneman) posits System 1 (fast, automatic, heuristic) and System 2 (slow, deliberate, rule-based) as two modes of cognition.", "cognitive_science", 0.92),
    ("c6_metacognition", "Metacognition is the ability to monitor and regulate one's own cognitive processes; accurate metacognitive monitoring predicts effective use of study strategies.", "cognitive_science", 0.91),
    ("c6_embodied", "Embodied cognition holds that mental processes are fundamentally shaped by the body's sensorimotor systems and its interaction with the environment.", "cognitive_science", 0.85),
    ("c6_situated", "Situated learning (Lave & Wenger) argues that knowledge is inseparable from the contexts in which it is acquired and used — opposing abstract, decontextualised instruction.", "cognitive_science", 0.84),
    ("c6_transfer", "Transfer of learning is the application of knowledge or skills acquired in one context to novel situations; near transfer is easier than far transfer.", "cognitive_science", 0.90),
    ("c6_executive_function", "Executive functions (working memory, inhibition, cognitive flexibility) are prefrontally mediated control processes that regulate goal-directed behaviour.", "cognitive_science", 0.92),
    ("c6_attention", "Selective attention filters the continuous stream of sensory input, prioritising task-relevant information and gating what enters working memory.", "cognitive_science", 0.93),
    ("c6_chunking", "Chunking (Miller) is the process of grouping individual items into meaningful units, dramatically expanding effective working memory capacity.", "cognitive_science", 0.91),
    ("c6_expertise", "Expertise involves reorganisation of knowledge into larger, more integrated chunks and automated retrieval, freeing working memory for higher-level reasoning.", "cognitive_science", 0.90),
]

# ---------------------------------------------------------------------------
# Edge definitions: (source, target, relation, weight)
# ---------------------------------------------------------------------------
EDGES: list[tuple[str, str, str, float]] = [
    # C1 internal — molecular cascade
    ("c1_nmda", "c1_ltp", "TRIGGERS", 0.95),
    ("c1_nmda", "c1_camkii", "ACTIVATES", 0.90),
    ("c1_camkii", "c1_ampa", "PHOSPHORYLATES", 0.88),
    ("c1_ampa", "c1_ltp", "MEDIATES", 0.92),
    ("c1_ltp", "c1_hebbian", "INSTANTIATES", 0.90),
    ("c1_ltd", "c1_hebbian", "INSTANTIATES", 0.85),
    ("c1_ltp", "c1_consolidation", "ENABLES", 0.93),
    ("c1_bdnf", "c1_ltp", "REQUIRED_FOR_LATE_PHASE", 0.87),
    ("c1_protein_synthesis", "c1_consolidation", "REQUIRED_FOR", 0.89),
    ("c1_synaptic_tagging", "c1_protein_synthesis", "CAPTURES", 0.83),
    ("c1_consolidation", "c1_reconsolidation", "PRECEDES", 0.80),
    ("c1_engram", "c1_place_cells", "IMPLEMENTED_BY", 0.85),
    ("c1_place_cells", "c1_grid_cells", "RECEIVES_INPUT_FROM", 0.88),
    ("c1_engram", "c1_ltp", "ENCODED_BY", 0.90),

    # C2 internal — learning strategies
    ("c2_ebbinghaus", "c2_spacing", "MOTIVATES", 0.95),
    ("c2_spacing", "c2_retrieval_practice", "SYNERGISES_WITH", 0.90),
    ("c2_retrieval_practice", "c2_desirable_difficulty", "EXEMPLIFIES", 0.88),
    ("c2_interleaving", "c2_desirable_difficulty", "EXEMPLIFIES", 0.87),
    ("c2_elaborative", "c2_desirable_difficulty", "EXEMPLIFIES", 0.83),
    ("c2_dual_coding", "c2_working_memory", "LEVERAGES", 0.85),
    ("c2_working_memory", "c2_phonological_loop", "CONTAINS", 0.92),
    ("c2_working_memory", "c2_visuospatial", "CONTAINS", 0.91),
    ("c2_working_memory", "c2_central_executive", "CONTROLLED_BY", 0.93),
    ("c2_episodic", "c2_semantic", "ABSTRACTS_INTO", 0.85),
    ("c2_source_monitoring", "c2_episodic", "APPLIED_TO", 0.80),
    ("c2_priming", "c2_semantic", "OPERATES_ON", 0.82),

    # C3 internal — neural circuits
    ("c3_entorhinal", "c3_dentate_gyrus", "PROJECTS_TO", 0.90),
    ("c3_entorhinal", "c3_ca3", "PROJECTS_TO", 0.88),
    ("c3_entorhinal", "c3_ca1", "PROJECTS_TO", 0.89),
    ("c3_dentate_gyrus", "c3_ca3", "PROJECTS_TO", 0.87),
    ("c3_ca3", "c3_ca1", "PROJECTS_VIA_SCHAFFER", 0.91),
    ("c3_ca1", "c3_entorhinal", "BACK_PROJECTS", 0.80),
    ("c3_hippocampus", "c3_ca1", "CONTAINS", 0.95),
    ("c3_hippocampus", "c3_ca3", "CONTAINS", 0.95),
    ("c3_hippocampus", "c3_dentate_gyrus", "CONTAINS", 0.95),
    ("c3_theta", "c3_ca3", "COORDINATES", 0.85),
    ("c3_theta", "c3_ca1", "COORDINATES", 0.85),
    ("c3_swr", "c3_sleep_consolidation", "DRIVES", 0.92),
    ("c3_gamma", "c3_theta", "NESTED_WITHIN", 0.83),
    ("c3_amygdala", "c3_hippocampus", "MODULATES", 0.88),
    ("c3_pfc", "c3_hippocampus", "BIDIRECTIONALLY_CONNECTED", 0.87),
    ("c3_basal_ganglia", "c3_cerebellum", "PARALLEL_SYSTEM_TO", 0.70),
    ("c3_dmn", "c3_hippocampus", "ANCHORED_BY", 0.85),

    # C4 internal — AI architecture
    ("c4_attention", "c4_transformer", "CORE_MECHANISM_OF", 0.95),
    ("c4_kv_cache", "c4_attention", "OPTIMISES", 0.88),
    ("c4_embedding", "c4_vector_store", "INDEXED_IN", 0.93),
    ("c4_faiss", "c4_vector_store", "IMPLEMENTS", 0.92),
    ("c4_vector_store", "c4_rag", "ENABLES", 0.93),
    ("c4_knowledge_graph", "c4_graph_rag", "ENABLES", 0.91),
    ("c4_rag", "c4_graph_rag", "EXTENDED_BY", 0.87),
    ("c4_rrf", "c4_rag", "IMPROVES_RANKING_IN", 0.85),
    ("c4_hopfield", "c4_attention", "FORMALLY_EQUIVALENT_TO", 0.90),
    ("c4_memory_augmented", "c4_transformer", "PREDATES", 0.75),
    ("c4_in_context", "c4_transformer", "EMERGENT_IN", 0.88),
    ("c4_continual_learning", "c4_memory_augmented", "ADDRESSED_BY", 0.82),

    # C5 internal — knowledge representation
    ("c5_semantic_net", "c5_spreading_activation", "USES", 0.92),
    ("c5_ontology", "c5_rdf", "SERIALISED_AS", 0.88),
    ("c5_rdf", "c5_sparql", "QUERIED_BY", 0.90),
    ("c5_cypher", "c5_sparql", "ANALOGOUS_TO", 0.80),
    ("c5_schema", "c5_frame", "PRECURSOR_TO", 0.82),
    ("c5_frame", "c5_ontology", "FORMALISED_INTO", 0.78),
    ("c5_conceptual_spaces", "c5_holographic", "RELATED_TO", 0.65),
    ("c5_semantic_net", "c5_ontology", "EVOLVED_INTO", 0.85),

    # C6 internal — cognitive science
    ("c6_cognitive_load", "c2_working_memory", "CONSTRAINED_BY", 0.90),
    ("c6_executive_function", "c2_central_executive", "OVERLAPS_WITH", 0.88),
    ("c6_attention", "c2_working_memory", "GATES", 0.91),
    ("c6_chunking", "c2_working_memory", "EXPANDS_CAPACITY_OF", 0.89),
    ("c6_expertise", "c6_chunking", "RELIES_ON", 0.88),
    ("c6_metacognition", "c6_dual_process", "INVOLVES", 0.82),
    ("c6_transfer", "c6_situated", "CHALLENGED_BY", 0.78),
    ("c6_dual_process", "c6_attention", "MEDIATED_BY", 0.80),

    # Cross-cluster bridges — C1 ↔ C3
    ("c1_ltp", "c3_hippocampus", "OCCURS_IN", 0.94),
    ("c1_ltp", "c3_ca1", "STUDIED_AT", 0.90),
    ("c1_ltp", "c3_ca3", "STUDIED_AT", 0.88),
    ("c1_engram", "c3_hippocampus", "LOCATED_IN", 0.92),
    ("c1_consolidation", "c3_sleep_consolidation", "ENABLED_BY", 0.93),
    ("c1_consolidation", "c3_swr", "DRIVEN_BY", 0.88),
    ("c1_place_cells", "c3_ca1", "FOUND_IN", 0.91),
    ("c1_grid_cells", "c3_entorhinal", "FOUND_IN", 0.91),
    ("c3_amygdala", "c1_bdnf", "RELEASES_VIA", 0.78),

    # Cross-cluster bridges — C2 ↔ C1
    ("c2_ebbinghaus", "c1_consolidation", "DESCRIBES_FAILURE_OF", 0.88),
    ("c2_spacing", "c1_ltp", "OPTIMISES", 0.85),
    ("c2_retrieval_practice", "c1_reconsolidation", "TRIGGERS", 0.82),
    ("c2_retrieval_practice", "c1_engram", "REACTIVATES", 0.84),

    # Cross-cluster bridges — C2 ↔ C3
    ("c2_working_memory", "c3_pfc", "IMPLEMENTED_BY", 0.93),
    ("c2_episodic", "c3_hippocampus", "DEPENDS_ON", 0.95),
    ("c2_semantic", "c3_hippocampus", "INITIALLY_DEPENDS_ON", 0.85),
    ("c2_procedural", "c3_basal_ganglia", "DEPENDS_ON", 0.91),
    ("c2_priming", "c3_entorhinal", "MEDIATED_BY", 0.72),
    ("c3_theta", "c2_episodic", "TEMPORALLY_ORGANISES", 0.80),

    # Cross-cluster bridges — C4 ↔ C1/C2/C3
    ("c4_hopfield", "c1_hebbian", "FORMALISES", 0.88),
    ("c4_hopfield", "c1_ltp", "BIOLOGICALLY_ANALOGOUS_TO", 0.82),
    ("c4_attention", "c2_working_memory", "FUNCTIONALLY_ANALOGOUS_TO", 0.86),
    ("c4_in_context", "c2_episodic", "ANALOGOUS_TO", 0.80),
    ("c4_continual_learning", "c1_consolidation", "INSPIRED_BY", 0.83),
    ("c4_embedding", "c1_engram", "ARTIFICIAL_ANALOGUE_OF", 0.78),
    ("c4_graph_rag", "c3_hippocampus", "ARCHITECTURALLY_INSPIRED_BY", 0.75),
    ("c4_rrf", "c2_spacing", "COMPUTATIONAL_ANALOGUE_OF", 0.65),

    # Cross-cluster bridges — C5 ↔ C4
    ("c5_semantic_net", "c4_knowledge_graph", "ANCESTOR_OF", 0.88),
    ("c5_spreading_activation", "c4_attention", "CONCEPTUALLY_RELATED_TO", 0.80),
    ("c5_ontology", "c4_knowledge_graph", "FORMALISED_IN", 0.87),
    ("c5_cypher", "c4_graph_rag", "USED_IN", 0.85),
    ("c5_rdf", "c4_knowledge_graph", "ALTERNATIVE_REPR_IN", 0.78),
    ("c5_holographic", "c4_embedding", "PRECURSOR_TO", 0.72),

    # Cross-cluster bridges — C6 ↔ C2/C3/C4
    ("c6_cognitive_load", "c2_desirable_difficulty", "CONSTRAINS", 0.88),
    ("c6_metacognition", "c2_retrieval_practice", "IMPROVES_USE_OF", 0.85),
    ("c6_expertise", "c2_semantic", "REORGANISES", 0.83),
    ("c6_chunking", "c2_episodic", "COMPRESSES", 0.78),
    ("c6_executive_function", "c3_pfc", "IMPLEMENTED_BY", 0.92),
    ("c6_attention", "c3_theta", "CORRELATED_WITH", 0.75),
    ("c6_dual_process", "c4_in_context", "ANALOGOUS_TO", 0.70),
    ("c6_transfer", "c4_continual_learning", "HUMAN_ANALOGUE_OF", 0.75),
]


def main() -> None:
    settings = Settings()

    print("Connecting to Neo4j…")
    neo4j = Neo4jClient(settings.neo4j)
    neo4j.connect()
    neo4j.create_constraints()

    print("Loading FAISS model…")
    faiss = FAISSIndex(settings.faiss)
    faiss.load_model()
    from pathlib import Path
    if Path(settings.faiss.index_path).exists():
        faiss.load()
    else:
        faiss.build_index()

    print(f"Inserting {len(NODES)} nodes…")
    created = 0
    for node_id, content, cluster, confidence in NODES:
        neo4j.upsert_node(
            node_id,
            content=content,
            metadata={"cluster": cluster, "confidence": confidence, "source": "seed_v1"},
        )
        faiss.add(node_id, content, {"cluster": cluster, "confidence": confidence})
        created += 1
        if created % 10 == 0:
            print(f"  {created}/{len(NODES)} nodes…")

    print("Saving FAISS index…")
    faiss.save()

    print(f"Inserting {len(EDGES)} edges…")
    for i, (src, tgt, rel, weight) in enumerate(EDGES, 1):
        try:
            neo4j.upsert_edge(src, tgt, relation=rel, weight=weight)
        except Exception as e:
            print(f"  WARN edge ({src}→{tgt}): {e}")
        if i % 20 == 0:
            print(f"  {i}/{len(EDGES)} edges…")

    # Final stats
    with neo4j._driver.session(database=settings.neo4j.database) as s:
        nc = s.run("MATCH (n:KnowledgeNode) RETURN count(n) AS c").single()["c"]
        ec = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    print("\n" + "="*50)
    print(f"  Nodes in Neo4j : {nc}")
    print(f"  Edges in Neo4j : {ec}")
    print(f"  FAISS vectors  : {len(faiss)}")
    print("="*50)
    print("Done. Seed complete.")

    neo4j.close()


if __name__ == "__main__":
    main()
