"""
CognitiveWeave — Streamlit Dashboard

Run:
  uv run streamlit run app.py
"""
from __future__ import annotations

import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from cognitiveweave.config.settings import Settings
from cognitiveweave.storage.neo4j_client import Neo4jClient
from cognitiveweave.storage.faiss_index import FAISSIndex
from cognitiveweave.retrieval.hybrid_retriever import HybridRetriever

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
        ["Ask the Graph", "Add Knowledge", "Decay Demo", "Graph Stats"],
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
            content = r.metadata.get("content", "")
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
                c = r.metadata.get("content", "")
                st.info(f"#{i+1} **{r.id}** — {c[:100]}{'…' if len(c)>100 else ''}")

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

        fig1 = px.bar(x=clusters, y=counts,
                      labels={"x": "", "y": "Concepts"},
                      title="Concepts per domain",
                      color=clusters, color_discrete_sequence=px.colors.qualitative.Pastel)
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
