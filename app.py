"""
CognitiveWeave — Streamlit Dashboard

Run:
  uv run streamlit run app.py
"""
from __future__ import annotations

import math, sys, os, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_agraph import agraph, Node, Edge, Config

from cognitiveweave.config.settings import Settings
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever
from cognitiveweave.llm.ollama_client import OllamaClient
from cognitiveweave.experiments.agent import SocietyAgent
from cognitiveweave.experiments.epistemic_monitor import EpistemicMonitor
from cognitiveweave.experiments.population import AgentPopulation
from cognitiveweave.experiments.observer import ExperimentObserver
from cognitiveweave.telemetry import setup_telemetry

# Telemetry: Jaeger varsa gönder, yoksa sessizce devam et
try:
    setup_telemetry(console_fallback=False)
except Exception:
    pass

st.set_page_config(
    page_title="CognitiveWeave",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  div[data-testid="metric-container"] {
    background: #1e1e2e; border-radius: 10px; padding: 0.8rem 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Epistemic Graph Visualization
# ---------------------------------------------------------------------------

_AGENT_COLORS = {
    "Mira": "#89b4fa",   # blue
    "Kael": "#a6e3a1",   # green
    "Sova": "#fab387",   # orange
}
_BASE_COLOR    = "#585b70"   # muted gray for knowledge nodes
_SYSTEM_COLOR  = "#f38ba8"   # red for system state node
_CROSS_COLOR   = "#f9e2af"   # yellow for cross-pollination edges


def build_epistemic_graph(
    neo4j: Neo4jClient,
    agent_colors: dict[str, str],
    max_nodes: int = 80,
) -> tuple[list[Node], list[Edge]]:
    """Query Neo4j and build agraph Node/Edge lists for the live graph view."""

    # 1 — fetch experiment nodes
    exp_query = """
    MATCH (n:KnowledgeNode {cluster: 'experiment'})
    RETURN n.id AS id, n.source AS source, n.content AS content,
           n.confidence AS confidence, n.cycle AS cycle
    ORDER BY n.cycle ASC
    LIMIT $limit
    """
    # 2 — fetch base knowledge nodes referenced by experiment nodes
    base_query = """
    MATCH (exp:KnowledgeNode {cluster: 'experiment'})-[:DERIVED_FROM]->(base:KnowledgeNode)
    WHERE base.cluster <> 'experiment'
    RETURN DISTINCT base.id AS id, base.content AS content
    LIMIT 40
    """
    # 3 — fetch edges
    edge_query = """
    MATCH (a:KnowledgeNode {cluster: 'experiment'})-[r:DERIVED_FROM]->(b:KnowledgeNode)
    RETURN a.id AS src, b.id AS tgt, a.source AS src_agent, b.source AS tgt_agent,
           b.cluster AS tgt_cluster
    LIMIT 200
    """
    # 4 — system state node
    system_query = """
    MATCH (n:KnowledgeNode {cluster: 'system'})
    RETURN n.id AS id, n.entropy AS entropy, n.regime AS regime
    LIMIT 1
    """

    nodes: list[Node] = []
    edges: list[Edge] = []
    seen_ids: set[str] = set()

    with neo4j._session() as s:
        # Experiment nodes
        for row in s.run(exp_query, limit=max_nodes):
            nid     = row["id"]
            source  = row["source"] or "unknown"
            conf    = float(row["confidence"] or 0.6)
            content = (row["content"] or "")[:60]
            color   = agent_colors.get(source, "#cdd6f4")
            size    = 14 + int(conf * 10)
            nodes.append(Node(
                id=nid, label=source,
                title=f"[{source}] {content}",
                size=size, color=color,
                font={"color": "#cdd6f4", "size": 10},
            ))
            seen_ids.add(nid)

        # Base knowledge nodes
        for row in s.run(base_query):
            nid     = row["id"]
            content = (row["content"] or nid)[:40]
            if nid not in seen_ids:
                nodes.append(Node(
                    id=nid, label=nid.split("_")[-1],
                    title=content,
                    size=8, color=_BASE_COLOR,
                    font={"color": "#9399b2", "size": 9},
                ))
                seen_ids.add(nid)

        # System state node
        sys_row = s.run(system_query).single()
        if sys_row:
            entropy = float(sys_row["entropy"] or 0)
            regime  = sys_row["regime"] or "healthy"
            nodes.append(Node(
                id="system:epistemic_state",
                label=f"Σ {regime}",
                title=f"System entropy: {entropy:.3f} | {regime}",
                size=22, color=_SYSTEM_COLOR,
                font={"color": "#cdd6f4", "size": 11, "bold": True},
            ))
            seen_ids.add("system:epistemic_state")

        # Edges
        for row in s.run(edge_query):
            src, tgt = row["src"], row["tgt"]
            if src not in seen_ids or tgt not in seen_ids:
                continue
            src_agent = row["src_agent"] or ""
            tgt_agent = row["tgt_agent"] or ""
            is_cross  = src_agent and tgt_agent and src_agent != tgt_agent
            edges.append(Edge(
                source=src, target=tgt,
                color=_CROSS_COLOR if is_cross else "#45475a",
                width=2.5 if is_cross else 1.0,
            ))

    return nodes, edges


def entropy_gauge(entropy: float, regime: str) -> go.Figure:
    """Plotly indicator gauge for system entropy."""
    color = {"homogenizing": "#89b4fa", "healthy": "#a6e3a1", "diverging": "#f38ba8"}.get(
        regime, "#cdd6f4"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(entropy, 3),
        title={"text": f"Epistemic Entropy<br><span style='font-size:0.8em;color:{color}'>{regime}</span>"},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#cdd6f4"},
            "bar":  {"color": color},
            "steps": [
                {"range": [0.00, 0.15], "color": "#313244"},
                {"range": [0.15, 0.60], "color": "#1e1e2e"},
                {"range": [0.60, 1.00], "color": "#313244"},
            ],
            "threshold": {
                "line": {"color": "#f9e2af", "width": 3},
                "thickness": 0.75,
                "value": entropy,
            },
        },
        number={"font": {"color": "#cdd6f4"}},
    ))
    fig.update_layout(
        height=220, margin=dict(t=60, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#cdd6f4",
    )
    return fig


TOPICS = {
    "Memory Biology": {
        "description": "Molecular & cellular mechanisms of memory — synaptic plasticity, LTP, engrams",
        "examples": [
            ("How does LTP work?",                              "c1_ltp"),
            ("What is synaptic plasticity?",                   "c1_hebbian"),
            ("How are memories physically stored in neurons?", "c1_engram"),
            ("What role does sleep play in memory?",           "c1_consolidation"),
        ],
    },
    "Learning Science": {
        "description": "Evidence-based study strategies — spaced repetition, retrieval practice, forgetting curves",
        "examples": [
            ("Why do we forget things over time?",                        "c2_ebbinghaus"),
            ("What is the best way to study for long-term retention?",    "c2_spacing"),
            ("Why does testing yourself work better than re-reading?",    "c2_retrieval_practice"),
            ("What is working memory and why does it matter?",            "c2_working_memory"),
        ],
    },
    "Brain Regions": {
        "description": "Neuroscience of memory — hippocampus, prefrontal cortex, sleep oscillations",
        "examples": [
            ("What does the hippocampus do?",                             "c3_hippocampus"),
            ("How does the brain consolidate memories during sleep?",     "c3_sleep_consolidation"),
            ("What is the role of the prefrontal cortex in memory?",      "c3_pfc"),
            ("How do brain waves relate to memory?",                      "c3_theta"),
        ],
    },
    "AI and Machine Learning": {
        "description": "How AI systems handle knowledge — transformers, RAG, vector search, knowledge graphs",
        "examples": [
            ("How does attention work in AI models?",                     "c4_attention"),
            ("What is retrieval-augmented generation?",                   "c4_rag"),
            ("How does FAISS find similar documents?",                    "c4_faiss"),
            ("What is a knowledge graph and why use one?",                "c4_knowledge_graph"),
        ],
    },
    "Knowledge Representation": {
        "description": "How knowledge is structured and stored — semantic networks, ontologies, schemas",
        "examples": [
            ("What is a semantic network?",                               "c5_semantic_net"),
            ("How does spreading activation explain memory?",             "c5_spreading_activation"),
            ("What is an ontology in computer science?",                  "c5_ontology"),
        ],
    },
    "Cognitive Science": {
        "description": "How humans think and learn — cognitive load, metacognition, expertise, attention",
        "examples": [
            ("What is cognitive load and how does it affect learning?",   "c6_cognitive_load"),
            ("What is metacognition?",                                    "c6_metacognition"),
            ("How does chunking expand memory capacity?",                 "c6_chunking"),
            ("What is the difference between System 1 and System 2?",    "c6_dual_process"),
        ],
    },
}

ALL_EXAMPLES: dict[str, str] = {}
for topic_data in TOPICS.values():
    for question, seed in topic_data["examples"]:
        ALL_EXAMPLES[question] = seed


def ndcg(ids: list[str], relevant: set[str], k: int = 5) -> float:
    dcg  = sum(1/math.log2(i+2) for i, x in enumerate(ids[:k]) if x in relevant)
    idcg = sum(1/math.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg/idcg if idcg else 0.0

def mrr(ids: list[str], relevant: set[str]) -> float:
    for i, x in enumerate(ids):
        if x in relevant:
            return 1/(i+1)
    return 0.0


@st.cache_resource(show_spinner="Connecting to knowledge graph…")
def init_backend():
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
    return neo4j, faiss, retriever, settings

neo4j, faiss, retriever, settings = init_backend()

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## CognitiveWeave")
    st.caption("Temporal Knowledge Graph")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Ask the Graph", "Add Knowledge", "Decay Demo", "Graph Stats", "Belief Dynamics"],
        label_visibility="collapsed",
    )

    st.divider()

    with neo4j._driver.session(database=settings.neo4j.database) as s:
        node_count = s.run("MATCH (n:KnowledgeNode) RETURN count(n) AS c").single()["c"]
        edge_count = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    st.caption("Graph size")
    c1, c2 = st.columns(2)
    c1.metric("Concepts", node_count)
    c2.metric("Relations", edge_count)

# ── page: Ask the Graph ───────────────────────────────────────────────────────

if page == "Ask the Graph":
    st.title("Ask the Knowledge Graph")
    st.markdown(
        "This graph contains **99 concepts** across 6 domains of cognitive science and AI. "
        "Ask any question below — the system finds the most relevant concepts using both "
        "semantic similarity and graph connections."
    )

    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""
    if "current_seed" not in st.session_state:
        st.session_state["current_seed"] = None

    query = st.text_input(
        "Your question",
        value=st.session_state["current_query"],
        placeholder="e.g. How does the brain consolidate memories during sleep?",
    )

    st.markdown("**Or pick an example question:**")
    for topic_name, topic_data in TOPICS.items():
        with st.expander(f"{topic_name} — {topic_data['description']}"):
            cols = st.columns(2)
            for i, (question, seed) in enumerate(topic_data["examples"]):
                if cols[i % 2].button(question, key=f"btn_{question}", use_container_width=True):
                    st.session_state["current_query"] = question
                    st.session_state["current_seed"]  = seed
                    st.rerun()

    seed_id = st.session_state.get("current_seed") or ALL_EXAMPLES.get(query)
    seeds   = [seed_id] if seed_id else None

    with st.expander("Advanced settings"):
        top_k = st.slider("How many results to show", 3, 15, 8)
        st.caption(
            f"**Starting point for graph traversal:** `{seed_id or 'none — semantic only'}`  \n"
            "When a starting concept is known, the system also follows graph connections "
            "to surface related ideas that a keyword search would miss."
        )

    if st.button("Search", type="primary", use_container_width=True) and query:
        st.session_state["current_query"] = query
        with st.spinner("Searching…"):
            results = retriever.retrieve(query, seed_node_ids=seeds, top_k=top_k)
        st.session_state["search_results"] = results
        st.session_state["search_query"]   = query

    if st.session_state.get("search_results"):
        results = st.session_state["search_results"]
        st.divider()
        st.markdown(f"### Results for: _{st.session_state['search_query']}_")

        graph_assisted = sum(1 for r in results if r.graph_rank is not None)
        if graph_assisted:
            st.success(
                f"Graph traversal surfaced **{graph_assisted}** concepts that "
                f"keyword/semantic search alone would have ranked much lower."
            )

        ids    = [r.id for r in results]
        f_comp = [1/(60 + (r.faiss_rank or 999)) for r in results]
        g_comp = [1/(60 + (r.graph_rank or 999)) if r.graph_rank else 0 for r in results]

        fig = go.Figure()
        fig.add_bar(name="Semantic similarity", x=ids, y=f_comp, marker_color="#89b4fa")
        fig.add_bar(name="Graph connections",   x=ids, y=g_comp, marker_color="#a6e3a1")
        fig.update_layout(
            barmode="stack",
            title="Why each concept was ranked here (semantic vs graph signal)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd6f4",
            height=280,
            margin=dict(t=40, b=10),
            xaxis_tickangle=-30,
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True)

        for i, r in enumerate(results):
            node    = neo4j.get_node(r.id)
            content = (node or {}).get("content", "") if node else ""
            cluster = r.metadata.get("cluster", "").replace("_", " ").title()
            how_found = []
            if r.faiss_rank and r.faiss_rank <= 10:
                how_found.append(f"semantic match #{r.faiss_rank}")
            if r.graph_rank:
                how_found.append(f"graph neighbour #{r.graph_rank}")
            how_str = " · ".join(how_found) if how_found else "retrieved"

            with st.expander(f"**#{i+1}** {r.id}   `{cluster}`   —   {how_str}", expanded=(i < 3)):
                st.markdown(f"> {content}" if content else "_No content_")


# ── page: Add Knowledge ──────────────────────────────────────────────────────

elif page == "Add Knowledge":
    st.title("Add Knowledge")
    st.markdown(
        "Anything you add here is immediately searchable. "
        "The system embeds it as a vector, stores it in the graph, and connects it to related concepts."
    )

    with st.form("add_knowledge_form"):
        content = st.text_area(
            "What do you want to add?",
            placeholder="e.g. The default mode network activates during mind-wandering and is suppressed during focused attention tasks.",
            height=120,
        )

        col1, col2 = st.columns(2)
        with col1:
            domain = st.selectbox(
                "Domain",
                options=[
                    "memory_biology", "memory_psychology", "neural_systems",
                    "ai_ml", "knowledge_repr", "cognitive_science", "other",
                ],
                format_func=lambda x: {
                    "memory_biology":    "Memory Biology",
                    "memory_psychology": "Learning Science",
                    "neural_systems":    "Brain Regions",
                    "ai_ml":             "AI and Machine Learning",
                    "knowledge_repr":    "Knowledge Representation",
                    "cognitive_science": "Cognitive Science",
                    "other":             "Other",
                }[x],
            )
        with col2:
            confidence = st.slider(
                "How confident are you in this?",
                0.0, 1.0, 0.8, 0.05,
                help="1.0 = certain fact · 0.5 = hypothesis · 0.2 = speculation",
            )

        connect_to = st.text_input(
            "Connect to an existing concept (optional)",
            placeholder="e.g. c3_hippocampus",
            help="If you know a related concept ID, the system will draw an edge between them.",
        )

        submitted = st.form_submit_button("Add to Knowledge Graph", type="primary", use_container_width=True)

    if submitted:
        if not content.strip():
            st.warning("Please enter some content.")
        else:
            with st.spinner("Adding to graph…"):
                import uuid
                node_id = str(uuid.uuid4())[:8]

                neo4j.upsert_node(
                    node_id,
                    content=content.strip(),
                    metadata={"cluster": domain, "confidence": confidence, "source": "ui"},
                )
                faiss.add(node_id, content.strip(), {"cluster": domain, "confidence": confidence})
                faiss.save()

                if connect_to.strip():
                    existing = neo4j.get_node(connect_to.strip())
                    if existing:
                        neo4j.upsert_edge(node_id, connect_to.strip(),
                                          relation="RELATED_TO", weight=confidence)
                        st.success(f"Added and connected to `{connect_to.strip()}`. Node ID: `{node_id}`")
                    else:
                        st.warning(f"Node `{connect_to.strip()}` not found — added without connection. Node ID: `{node_id}`")
                else:
                    st.success(f"Added to graph. Node ID: `{node_id}`")

            st.markdown("**Similar concepts already in the graph:**")
            results = retriever.retrieve(content.strip(), top_k=5)
            for i, r in enumerate(results[:5]):
                node = neo4j.get_node(r.id)
                c = (node or {}).get("content", "") if node else ""
                st.info(f"#{i+1} **{r.id}** — {c[:120]}{'…' if len(c)>120 else ''}")

            st.caption("Go to Ask the Graph to query your new knowledge.")


# ── page: Decay Demo ─────────────────────────────────────────────────────────

elif page == "Decay Demo":
    st.title("The Forgetting Curve in Action")

    st.markdown("""
    ### The problem with AI memory

    Most AI systems treat all knowledge as equally important, forever.
    If outdated or incorrect information enters the knowledge base,
    it stays at the top of search results indefinitely.

    **CognitiveWeave automatically suppresses stale knowledge** using the
    same mathematical curve that describes human forgetting (Ebbinghaus, 1885):

    > *retention = e^(−time / stability)*

    Frequently-accessed knowledge decays slowly. Forgotten knowledge fades away.
    """)

    st.divider()

    DEMO_QUERY    = "what brain mechanisms underlie memory consolidation?"
    DEMO_SEED     = ["c1_consolidation"]
    DEMO_RELEVANT = {"c3_swr", "c3_sleep_consolidation", "c3_hippocampus",
                     "c1_protein_synthesis", "c1_reconsolidation", "c1_ltp"}
    NOISE_IDS     = ["noise_st_1", "noise_st_2", "noise_st_3"]
    NOISE_CONTENT = [
        "Deprecated model: engrams are stored exclusively in the prefrontal cortex — hippocampus is not involved.",
        "Outdated hypothesis: REM sleep is the sole driver of consolidation, slow-wave sleep plays no role.",
        "Superseded theory: memory consolidation completes within minutes via a single protein synthesis event.",
    ]

    st.markdown(f"**Demo query:** _{DEMO_QUERY}_")
    st.markdown("**Correct answers** the system should surface:")
    for r in sorted(DEMO_RELEVANT):
        st.markdown(f"- `{r}`")

    st.divider()
    st.markdown("### Step 1 — Add some outdated information")
    st.markdown(
        "We add 3 pieces of **incorrect/superseded science** to the graph — "
        "the kind of thing that accumulates in any real knowledge base over time:"
    )
    for c in NOISE_CONTENT:
        st.warning(c)

    if st.button("Add stale knowledge and see what happens", use_container_width=True):
        for nid, content in zip(NOISE_IDS, NOISE_CONTENT):
            neo4j.upsert_node(nid, content=content,
                              metadata={"cluster": "stale", "confidence": 0.2})
            faiss.add(nid, content, {"cluster": "stale"})
            neo4j.upsert_edge("c1_consolidation", nid, relation="RELATED_TO", weight=0.9)
        faiss.save()
        with neo4j._driver.session(database=settings.neo4j.database) as s:
            s.run("MATCH ()-[r]->() SET r.last_decay_at = toString(datetime())")

        results_std = retriever.retrieve(DEMO_QUERY, seed_node_ids=DEMO_SEED, top_k=5)
        st.session_state.update({
            "ids_std": [r.id for r in results_std],
            "results_std": results_std,
            "noise_injected": True,
            "decay_done": False,
        })

    if st.session_state.get("noise_injected"):
        ids_std     = st.session_state["ids_std"]
        results_std = st.session_state["results_std"]
        n_std       = ndcg(ids_std, DEMO_RELEVANT)
        noise_std   = sum(1 for x in ids_std if x in NOISE_IDS)

        st.markdown("#### Without decay — what the system returns:")
        for i, r in enumerate(results_std):
            if r.id in NOISE_IDS:
                st.error(f"#{i+1}  wrong  **{r.id}** — this is stale/incorrect")
            elif r.id in DEMO_RELEVANT:
                st.success(f"#{i+1}  correct  {r.id}")
            else:
                st.info(f"#{i+1}  {r.id}")

        c1, c2 = st.columns(2)
        c1.metric("Accuracy (nDCG@5)", f"{n_std:.0%}", help="How well the top 5 match correct answers")
        c2.metric("Stale results in top 5", noise_std,
                  delta="problem" if noise_std > 0 else "ok",
                  delta_color="inverse")

        st.divider()
        st.markdown("### Step 2 — Time passes, decay runs")
        st.markdown(
            "In a real deployment, the Curator agent runs automatically on a schedule. "
            "Here we simulate **45 days** of no access to the stale nodes — "
            "their connections to the rest of the graph decay and are pruned."
        )

        if st.button("Simulate 45 days of decay", type="primary", use_container_width=True):
            q = """
            UNWIND $ids AS nid
            MATCH ()-[r]->(b:KnowledgeNode {id: nid})
            SET r.last_decay_at = toString(datetime() - duration({days: 45}))
            """
            with neo4j._driver.session(database=settings.neo4j.database) as s:
                s.run(q, ids=NOISE_IDS)
            updated = neo4j.decay_edge_weights()
            pruned  = neo4j.prune_weak_edges(threshold=0.3)

            results_cw = retriever.retrieve(DEMO_QUERY, seed_node_ids=DEMO_SEED, top_k=5)
            ids_cw = [r.id for r in results_cw]
            st.session_state.update({
                "ids_cw": ids_cw,
                "results_cw": results_cw,
                "updated": updated,
                "pruned": pruned,
                "decay_done": True,
            })
            for nid in NOISE_IDS:
                neo4j.delete_node(nid)
            st.session_state["noise_injected"] = False

    if st.session_state.get("decay_done"):
        ids_cw     = st.session_state["ids_cw"]
        results_cw = st.session_state["results_cw"]
        ids_std    = st.session_state["ids_std"]

        n_std    = ndcg(ids_std, DEMO_RELEVANT)
        n_cw     = ndcg(ids_cw,  DEMO_RELEVANT)
        noise_cw = sum(1 for x in ids_cw if x in NOISE_IDS)
        noise_std = sum(1 for x in ids_std if x in NOISE_IDS)

        st.markdown(
            f"**{st.session_state['updated']} edges decayed · "
            f"{st.session_state['pruned']} stale connections pruned automatically**"
        )
        st.markdown("#### With CognitiveWeave — after decay:")
        for i, r in enumerate(results_cw):
            if r.id in NOISE_IDS:
                st.error(f"#{i+1}  wrong  **{r.id}** — stale")
            elif r.id in DEMO_RELEVANT:
                st.success(f"#{i+1}  correct  {r.id}")
            else:
                st.info(f"#{i+1}  {r.id}")

        st.divider()
        st.markdown("### Result")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{n_cw:.0%}",
                  delta=f"{(n_cw-n_std)/n_std*100:+.0f}%",
                  help="nDCG@5 — how well the top 5 match correct answers")
        c2.metric("First correct result at rank",
                  f"#{next((i+1 for i,x in enumerate(ids_cw) if x in DEMO_RELEVANT), '—')}",
                  delta="was #{0}".format(next((i+1 for i,x in enumerate(ids_std) if x in DEMO_RELEVANT), '—')))
        c3.metric("Stale results removed", f"{noise_std - noise_cw} of {noise_std}",
                  delta="automatically" if noise_cw < noise_std else "none")

        fig = go.Figure(data=[
            go.Bar(name="Standard (no decay)", x=["Accuracy (nDCG@5)", "MRR"],
                   y=[n_std, mrr(ids_std, DEMO_RELEVANT)], marker_color="#f38ba8"),
            go.Bar(name="CognitiveWeave",      x=["Accuracy (nDCG@5)", "MRR"],
                   y=[n_cw,  mrr(ids_cw,  DEMO_RELEVANT)], marker_color="#a6e3a1"),
        ])
        fig.update_layout(
            barmode="group",
            title="Retrieval quality: before vs after decay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd6f4",
            height=320,
            yaxis=dict(range=[0, 1.1], tickformat=".0%"),
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── page: Graph Stats ────────────────────────────────────────────────────────

elif page == "Graph Stats":
    st.title("What's in the Graph")
    st.markdown(
        "The knowledge graph currently contains **99 concepts** across 6 domains, "
        "connected by **115+ typed relationships**. "
        "Edge weights decay over time — frequently accessed connections stay strong, "
        "unused ones fade."
    )

    col1, col2 = st.columns(2)

    with col1:
        with neo4j._driver.session(database=settings.neo4j.database) as s:
            rows = list(s.run(
                "MATCH (n:KnowledgeNode) RETURN n.cluster AS cluster, count(n) AS cnt ORDER BY cnt DESC"
            ))
        clusters = [r["cluster"].replace("_", " ").title() if r["cluster"] else "Other" for r in rows]
        counts   = [r["cnt"] for r in rows]

        import pandas as pd
        fig1 = px.bar(
            pd.DataFrame({"Domain": clusters, "Concepts": counts}),
            x="Domain", y="Concepts",
            title="Concepts per domain",
            color="Domain",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd6f4", showlegend=False, height=320,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        with neo4j._driver.session(database=settings.neo4j.database) as s:
            weights = [r["w"] for r in s.run("MATCH ()-[r]->() RETURN r.weight AS w") if r["w"] is not None]

        fig2 = px.histogram(x=weights, nbins=25,
                            labels={"x": "Edge weight", "y": "Count"},
                            title="Edge weight distribution (1.0 = strong, 0.0 = decayed)")
        fig2.update_traces(marker_color="#89b4fa")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cdd6f4", height=320, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("### Domain map")
    for topic_name, topic_data in TOPICS.items():
        st.markdown(f"**{topic_name}** — {topic_data['description']}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run decay cycle now", use_container_width=True,
                     help="Applies Ebbinghaus decay to all edges based on time since last access"):
            updated = neo4j.decay_edge_weights()
            pruned  = neo4j.prune_weak_edges(threshold=0.1)
            st.success(f"{updated} edges updated · {pruned} weak edges pruned")
            st.rerun()
    with col2:
        if st.button("Reset edge timestamps (demo reset)", use_container_width=True,
                     help="Sets all edges back to 'just accessed' for demo purposes"):
            with neo4j._driver.session(database=settings.neo4j.database) as s:
                s.run("MATCH ()-[r]->() SET r.last_decay_at = toString(datetime())")
            st.success("All edges reset to fresh")
            st.rerun()


# ── page: Belief Dynamics ─────────────────────────────────────────────────────

elif page == "Belief Dynamics":
    st.title("Belief Dynamics Experiment")
    st.markdown(
        "A closed population of LLM agents shares this knowledge graph as their "
        "only communication channel. Each agent has **asymmetric initial knowledge** "
        "and a **limited bandwidth** — it can only read a few nodes per cycle. "
        "We observe how shared beliefs form, drift, and stabilize without a ground truth authority."
    )

    # ── agent presets ────────────────────────────────────────────────────────
    AGENT_PRESETS = [
        {
            "name":     "Mira",
            "interest": "memory consolidation and the role of the hippocampus in sleep",
            "seed":     "c2_ebbinghaus",  # H3: all agents same bridge seed
            "color":    "#89b4fa",
        },
        {
            "name":     "Kael",
            "interest": "attention mechanisms in transformer models and knowledge retrieval",
            "seed":     "c2_ebbinghaus",  # H3: all agents same bridge seed
            "color":    "#a6e3a1",
        },
        {
            "name":     "Sova",
            "interest": "forgetting curves and long-term memory retention strategies",
            "seed":     "c2_ebbinghaus",
            "color":    "#fab387",
        },
    ]

    # ── sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.caption("Experiment controls")
        bandwidth   = st.slider("Bandwidth (nodes/cycle)", 3, 10, 5,
                                help="How many nodes each agent can read per cycle")
        n_cycles    = st.number_input("Cycles to run", min_value=1, max_value=20, value=3)
        use_ollama  = st.checkbox("Use Ollama for synthesis", value=True,
                                  help="Uncheck to use heuristic fallback (faster, no LLM)")
        shared_faiss = st.checkbox("Shared FAISS index", value=True,
                                   help="Uncheck = isolated mode (baseline): agents cannot find each other's writes via semantic search")

    # ── mode badge ────────────────────────────────────────────────────────────
    if st.session_state.get("exp_shared_faiss", True):
        st.success("Mode: **Shared FAISS** — agents can find each other's writes", icon="🔗")
    elif "exp_population" in st.session_state:
        st.warning("Mode: **Isolated** (baseline) — agents are semantically blind to each other", icon="🚫")

    # ── init / reset ─────────────────────────────────────────────────────────
    col_init, col_reset = st.columns(2)

    with col_init:
        if st.button("Initialize agents", use_container_width=True, type="primary"):
            ollama = OllamaClient(settings.ollama)
            agents = [
                SocietyAgent(
                    name     = p["name"],
                    interest = p["interest"],
                    retriever= retriever,
                    neo4j    = neo4j,
                    ollama   = ollama,
                    bandwidth= bandwidth,
                    seed_node_id=p["seed"],
                    faiss    = faiss if shared_faiss else None,
                )
                for p in AGENT_PRESETS
            ]
            st.session_state["exp_shared_faiss"] = shared_faiss
            observer   = ExperimentObserver(neo4j, database=settings.neo4j.database)
            monitor    = EpistemicMonitor(neo4j)
            population = AgentPopulation(agents, monitor=monitor)
            st.session_state["exp_population"] = population
            st.session_state["exp_observer"]   = observer
            st.session_state["exp_snapshots"]  = []
            st.session_state["exp_cycle"]      = 0
            st.success(f"Initialized {len(agents)} agents: {', '.join(p['name'] for p in AGENT_PRESETS)}")

    with col_reset:
        if st.button("Reset experiment", use_container_width=True):
            if "exp_observer" in st.session_state:
                deleted = st.session_state["exp_observer"].cleanup()
                st.info(f"Deleted {deleted} experiment nodes from graph.")
            # Rebuild FAISS from base nodes only — removes all stale experiment vectors
            with neo4j._driver.session(database=settings.neo4j.database) as _s:
                base_nodes = list(_s.run(
                    "MATCH (n:KnowledgeNode) WHERE n.cluster <> 'experiment' OR n.cluster IS NULL "
                    "RETURN n.id AS id, n.content AS content, n.cluster AS cluster"
                ))
            faiss.build_index()
            faiss.add_batch([
                {"id": r["id"], "text": r["content"], "metadata": {"cluster": r["cluster"] or ""}}
                for r in base_nodes if r["id"] and r["content"]
            ])
            for key in ["exp_population", "exp_observer", "exp_snapshots", "exp_cycle", "exp_shared_faiss"]:
                st.session_state.pop(key, None)
            st.rerun()

    if "exp_population" not in st.session_state:
        # Auto-restore from Neo4j if experiment data exists (e.g. after Streamlit restart)
        _restore_observer = ExperimentObserver(neo4j, database=settings.neo4j.database)
        if _restore_observer.has_experiment_data():
            _agent_names = [p["name"] for p in AGENT_PRESETS]
            _snapshots   = _restore_observer.rebuild_snapshots(_agent_names)
            _max_cycle   = max((s.cycle for s in _snapshots), default=0)
            st.session_state["exp_observer"]   = _restore_observer
            st.session_state["exp_snapshots"]  = _snapshots
            st.session_state["exp_cycle"]      = _max_cycle + 1
            # Stub population (read-only, no agents) — Initialize to run new cycles
            st.session_state["exp_population"] = AgentPopulation([])
            st.info("Previous experiment restored from graph. Click **Initialize agents** to run more cycles.")
        else:
            st.info("Click **Initialize agents** to start.")
            st.stop()

    population: AgentPopulation = st.session_state["exp_population"]
    observer:   ExperimentObserver = st.session_state["exp_observer"]
    snapshots:  list = st.session_state["exp_snapshots"]

    # ── agent cards ───────────────────────────────────────────────────────────
    if population.agents:
        st.divider()
        st.markdown("### Agents")
        agent_cols = st.columns(len(AGENT_PRESETS))
        for i, (agent, preset) in enumerate(zip(population.agents, AGENT_PRESETS)):
            with agent_cols[i]:
                st.markdown(
                    f"<div style='border-left: 4px solid {preset['color']}; "
                    f"padding: 0.5rem 0.8rem; border-radius:4px; background:#1e1e2e'>"
                    f"<b>{agent.name}</b><br>"
                    f"<small style='color:#cdd6f4'>{agent.interest[:80]}…</small><br>"
                    f"<small>seed: <code>{agent._seed or 'none'}</code> · bw: {agent.bandwidth}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── run cycles ────────────────────────────────────────────────────────────
    st.divider()
    current_cycle = st.session_state.get("exp_cycle", 0)
    st.markdown(f"### Run cycles  —  current cycle: **{current_cycle}**")

    if population.agents and st.button(f"Run {n_cycles} cycle(s)", use_container_width=True, type="primary"):
        progress = st.progress(0.0, text="Running…")
        for i in range(int(n_cycles)):
            result = population.run_cycle(current_cycle + i)
            snap   = observer.snapshot(current_cycle + i, population.agent_names)
            snapshots.append(snap)
            progress.progress((i + 1) / int(n_cycles),
                              text=f"Cycle {current_cycle + i} done — "
                                   f"{len(result.successful_writes)} writes")
        st.session_state["exp_cycle"]     = current_cycle + int(n_cycles)
        st.session_state["exp_snapshots"] = snapshots
        progress.empty()
        st.rerun()

    if not snapshots:
        st.info("No cycles run yet.")
        st.stop()

    # ── live epistemic graph ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Live Epistemic Graph")
    st.caption(
        "Colored nodes = agent beliefs. Gray nodes = base knowledge. "
        "**Yellow edges** = cross-pollination (one agent built on another's belief). "
        "Red node = system entropy state."
    )

    graph_col, gauge_col = st.columns([3, 1])

    with graph_col:
        try:
            g_nodes, g_edges = build_epistemic_graph(
                neo4j, _AGENT_COLORS, max_nodes=80
            )
            if g_nodes:
                config = Config(
                    width="100%", height=480,
                    directed=True, physics=True,
                    hierarchical=False,
                    nodeHighlightBehavior=True,
                    highlightColor="#f9e2af",
                    backgroundColor="#1e1e2e",
                    node={"labelProperty": "label"},
                    link={"renderLabel": False},
                )
                agraph(nodes=g_nodes, edges=g_edges, config=config)
            else:
                st.info("Run at least one cycle to see the graph.")
        except Exception as e:
            st.warning(f"Graph unavailable: {e}")

    with gauge_col:
        # Entropy gauge — read from monitor history or graph
        from cognitiveweave.experiments.epistemic_monitor import read_epistemic_signal
        signal = read_epistemic_signal(neo4j)
        entropy = signal.get("entropy", 0.0)
        regime  = signal.get("regime", "healthy")
        st.plotly_chart(
            entropy_gauge(entropy, regime),
            use_container_width=True,
        )

        # Agent status cards
        st.markdown("**Agent status**")
        latest_snap = snapshots[-1] if snapshots else None
        for preset in AGENT_PRESETS:
            name  = preset["name"]
            color = preset["color"]
            count = latest_snap.node_counts.get(name, 0) if latest_snap else 0
            cross = latest_snap.cross_reads.get(name, 0) if latest_snap else 0
            st.markdown(
                f"<div style='background:#313244;border-left:4px solid {color};"
                f"padding:8px 12px;border-radius:6px;margin-bottom:6px'>"
                f"<b style='color:{color}'>{name}</b><br>"
                f"<span style='font-size:0.8em;color:#cdd6f4'>"
                f"Beliefs: {count} &nbsp;|&nbsp; Cross-pol: {cross}</span></div>",
                unsafe_allow_html=True,
            )

    # ── metrics over time ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Belief production over time")

    agent_names  = population.agent_names or [p["name"] for p in AGENT_PRESETS]
    agent_colors = {p["name"]: p["color"] for p in AGENT_PRESETS}
    cycles_axis  = [s.cycle for s in snapshots]

    fig_prod = go.Figure()
    for name in agent_names:
        fig_prod.add_scatter(
            name=name,
            x=cycles_axis,
            y=[s.node_counts.get(name, 0) for s in snapshots],
            mode="lines+markers",
            line=dict(color=agent_colors.get(name, "#cdd6f4"), width=2),
        )
    fig_prod.update_layout(
        title="Cumulative nodes written per agent",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cdd6f4", height=280, margin=dict(t=40, b=10),
        xaxis_title="Cycle", yaxis_title="Nodes written",
    )
    st.plotly_chart(fig_prod, width="stretch")

    fig_conf = go.Figure()
    for name in agent_names:
        fig_conf.add_scatter(
            name=name,
            x=cycles_axis,
            y=[s.avg_confidence.get(name, 0.0) for s in snapshots],
            mode="lines+markers",
            line=dict(color=agent_colors.get(name, "#cdd6f4"), width=2, dash="dot"),
        )
    fig_conf.update_layout(
        title="Average belief confidence per agent",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cdd6f4", height=260, margin=dict(t=40, b=10),
        xaxis_title="Cycle", yaxis_title="Avg confidence",
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(fig_conf, width="stretch")

    # ── cross-pollination ─────────────────────────────────────────────────────
    latest = snapshots[-1]

    st.divider()
    st.markdown("### Cross-pollination")
    st.caption(
        "How many times has each agent built a belief on top of *another agent's* write? "
        "A rising count means knowledge is spreading across the population."
    )
    cross_cols = st.columns(len(agent_names))
    for i, name in enumerate(agent_names):
        cross_cols[i].metric(
            name,
            latest.cross_reads.get(name, 0),
            help="Times this agent derived from another agent's node",
        )

    # ── belief drift ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Belief drift")
    st.caption(
        "Cosine distance between each agent's cycle-0 belief and each subsequent cycle. "
        "A rising line means the agent's language is moving away from its starting point — "
        "absorbing concepts from the shared graph."
    )
    drift_series: dict[str, tuple[list[int], list[float]]] = {}
    for name in agent_names:
        beliefs_by_cycle = observer.get_agent_beliefs_by_cycle(name)
        if len(beliefs_by_cycle) < 2:
            continue
        # last belief written each cycle
        cycle_map: dict[int, str] = {}
        for cyc, text in beliefs_by_cycle:
            cycle_map[cyc] = text
        sorted_cycles = sorted(cycle_map.keys())
        texts = [cycle_map[c] for c in sorted_cycles]
        vecs = faiss.encode(texts)
        base = vecs[0]
        dists = [float(1.0 - float(base @ vecs[i])) for i in range(len(vecs))]
        drift_series[name] = (sorted_cycles, dists)

    if drift_series:
        fig_drift = go.Figure()
        for name, (cycles, dists) in drift_series.items():
            fig_drift.add_scatter(
                x=cycles, y=dists, mode="lines+markers", name=name,
            )
        fig_drift.update_layout(
            xaxis_title="Cycle", yaxis_title="Drift from cycle 0 (cosine distance)",
            yaxis=dict(range=[0, 1]), height=300, margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_drift, width="stretch")

    # ── consensus anchors ─────────────────────────────────────────────────────
    if latest.consensus_anchors:
        st.divider()
        st.markdown("### Emerging consensus anchors")
        st.caption(
            "These are knowledge nodes that **multiple agents independently** chose "
            "as the foundation for their beliefs — focal points crystallizing without "
            "any central authority."
        )
        for anchor_id in latest.consensus_anchors:
            node = neo4j.get_node(anchor_id)
            content = (node or {}).get("content", "")
            st.success(f"`{anchor_id}` — {content[:200]}")

    # ── save run ──────────────────────────────────────────────────────────────
    st.divider()
    col_save, col_label = st.columns([1, 3])
    with col_label:
        run_label = st.text_input(
            "Run label (optional)",
            placeholder="e.g. baseline_3agents_8cycles",
            label_visibility="collapsed",
        )
    with col_save:
        if st.button("Save run to disk", use_container_width=True):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            label = run_label.strip().replace(" ", "_") or "run"
            filename = f"experiments/runs/{timestamp}_{label}.json"
            run_data = {
                "saved_at": timestamp,
                "label": label,
                "agent_presets": AGENT_PRESETS,
                "total_cycles": st.session_state.get("exp_cycle", 0),
                "snapshots": [
                    {
                        "cycle": s.cycle,
                        "node_counts": s.node_counts,
                        "avg_confidence": s.avg_confidence,
                        "total_nodes": s.total_nodes,
                        "consensus_anchors": s.consensus_anchors,
                        "cross_reads": s.cross_reads,
                    }
                    for s in snapshots
                ],
                "belief_log": observer.get_all_experiment_nodes(),
                "drift_series": {
                    name: {"cycles": cyc, "distances": dists}
                    for name, (cyc, dists) in drift_series.items()
                },
            }
            os.makedirs("experiments/runs", exist_ok=True)
            with open(filename, "w") as f:
                json.dump(run_data, f, indent=2, default=str)
            st.success(f"Saved → `{filename}`")

    # ── belief log ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Belief log")
    nodes = observer.get_all_experiment_nodes()
    if nodes:
        for n in reversed(nodes[-30:]):
            agent_name = n.get("source", "?")
            color = agent_colors.get(agent_name, "#cdd6f4")
            conf  = float(n.get("confidence") or 0)
            cycle = n.get("cycle", "?")
            belief = n.get("content", "")
            st.markdown(
                f"<div style='border-left: 3px solid {color}; padding: 0.4rem 0.8rem; "
                f"margin-bottom:0.3rem; background:#1e1e2e; border-radius:4px'>"
                f"<small><b>{agent_name}</b> · cycle {cycle} · conf {conf:.2f}</small><br>"
                f"{belief}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No experiment nodes written yet.")
