"""Ditwah Cyclone Chatbot — RAG-powered Q&A with claim stance analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.styling import apply_page_style
from components.interpretations import generate_stance_interpretation
from src.db import get_db
from src.llm import get_llm, EmbeddingClient

# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title="Ditwah Chatbot - Sri Lanka Media Bias Detector",
    page_icon="💬",
    layout="wide",
)

apply_page_style()

# ============================================================================
# Cached resources
# ============================================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name: str):
    """Load a sentence-transformer model. EmbeddingGemma uses 'retrieval' task."""
    is_gemma = "embeddinggemma" in model_name.lower()
    return EmbeddingClient(
        provider="local",
        model=model_name,
        task="retrieval" if is_gemma else None,
    )


@st.cache_resource(show_spinner="Connecting to LLM...")
def load_llm():
    return get_llm()


@st.cache_resource(show_spinner="Loading NLI stance model...")
def load_nli_analyzer():
    from src.nli_stance import NLIStanceAnalyzer
    return NLIStanceAnalyzer()


# ============================================================================
# Data loaders
# ============================================================================

@st.cache_data(ttl=600)
def get_available_embedding_models() -> list:
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT e.embedding_model, COUNT(*) AS article_count
                FROM {schema}.embeddings e
                JOIN {schema}.news_articles n ON e.article_id = n.id
                WHERE n.is_ditwah_cyclone = 1
                GROUP BY e.embedding_model
                ORDER BY article_count DESC
            """)
            return cur.fetchall()


@st.cache_data(ttl=600)
def get_claims_versions_with_stance() -> list:
    """Return ditwah_claims versions that have both sentiment and stance data."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT rv.id, rv.name, rv.created_at,
                       COUNT(DISTINCT dc.id)              AS claims_count,
                       COUNT(DISTINCT cs_sent.article_id) AS covered_articles,
                       COUNT(DISTINCT cst.id)             AS stance_rows
                FROM {schema}.result_versions rv
                JOIN {schema}.ditwah_claims  dc     ON dc.result_version_id = rv.id
                JOIN {schema}.claim_sentiment cs_sent ON cs_sent.claim_id   = dc.id
                JOIN {schema}.claim_stance    cst     ON cst.claim_id       = dc.id
                GROUP BY rv.id, rv.name, rv.created_at
                ORDER BY stance_rows DESC
            """)
            return cur.fetchall()


@st.cache_data(ttl=600)
def get_total_ditwah_count() -> int:
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) AS count
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
            """)
            return cur.fetchone()["count"]


def _embedding_to_pg_str(embedding: list) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


def search_ditwah_articles(question_embedding: list, embedding_model: str, top_k: int) -> list:
    """Semantic search: top-K Ditwah articles by cosine similarity."""
    pg_vec = _embedding_to_pg_str(question_embedding)
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                SELECT n.id, n.title, n.content, n.source_id, n.date_posted, n.url,
                       1 - (e.embedding <=> %s::vector) AS similarity
                FROM {schema}.embeddings e
                JOIN {schema}.news_articles n ON e.article_id = n.id
                WHERE n.is_ditwah_cyclone = 1
                  AND e.embedding_model = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (pg_vec, embedding_model, pg_vec, top_k),
            )
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claims_for_articles(article_ids: tuple, claims_version_id: str, stance_model: str = None) -> list:
    """
    Find the general claims linked to the retrieved articles.

    Primary path: news_articles → ditwah_article_claims (general_claim_id) → ditwah_claims.
    Each article is assigned to exactly one general claim via the two-step pipeline.
    Only returns claims that have at least 20 total articles and have stance data.
    Falls back to claim_sentiment if no results from the primary path.

    stance_model: if provided, only returns claims that have stance data for that specific model.
    """
    if not article_ids:
        return []

    def model_exists_clause(schema, model):
        if model:
            return f"EXISTS (SELECT 1 FROM {schema}.claim_stance cst WHERE cst.claim_id = dc.id AND cst.llm_model = %s)"
        return f"EXISTS (SELECT 1 FROM {schema}.claim_stance cst WHERE cst.claim_id = dc.id)"

    with get_db() as db:
        schema = db.config["schema"]
        exists_clause = model_exists_clause(schema, stance_model)

        with db.cursor() as cur:
            # Primary: use direct article→claim assignment (general_claim_id)
            params = [list(article_ids), claims_version_id]
            if stance_model:
                params.append(stance_model)
            cur.execute(
                f"""
                SELECT
                    dc.id,
                    dc.claim_text,
                    dc.claim_category,
                    dc.article_count,
                    COUNT(DISTINCT dac.article_id) AS matched_articles
                FROM {schema}.ditwah_article_claims dac
                JOIN {schema}.ditwah_claims dc ON dac.general_claim_id = dc.id
                WHERE dac.article_id = ANY(%s::uuid[])
                  AND dc.result_version_id = %s
                  AND {exists_clause}
                GROUP BY dc.id, dc.claim_text, dc.claim_category, dc.article_count
                ORDER BY matched_articles DESC, dc.article_count DESC
                """,
                params,
            )
            rows = cur.fetchall()

        if rows:
            return rows

        # Fallback: use claim_sentiment (older data without general_claim_id)
        with db.cursor() as cur:
            params = [list(article_ids), claims_version_id]
            if stance_model:
                params.append(stance_model)
            cur.execute(
                f"""
                SELECT
                    dc.id,
                    dc.claim_text,
                    dc.claim_category,
                    dc.article_count,
                    COUNT(DISTINCT cs_sent.article_id) AS matched_articles
                FROM {schema}.claim_sentiment cs_sent
                JOIN {schema}.ditwah_claims dc ON cs_sent.claim_id = dc.id
                WHERE cs_sent.article_id = ANY(%s::uuid[])
                  AND dc.result_version_id = %s
                  AND {exists_clause}
                GROUP BY dc.id, dc.claim_text, dc.claim_category, dc.article_count
                ORDER BY matched_articles DESC, dc.article_count DESC
                """,
                params,
            )
            return cur.fetchall()


@st.cache_data(ttl=600)
def get_available_stance_models() -> list:
    """Return distinct llm_model values stored in claim_stance, sorted alphabetically."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT llm_model
                FROM {schema}.claim_stance
                WHERE llm_model IS NOT NULL
                ORDER BY llm_model
            """)
            rows = cur.fetchall()
            return [r["llm_model"] for r in rows]


@st.cache_data(ttl=300)
def load_stance_by_source(
    claim_id: str,
    stance_model: str = None,
    article_ids: tuple = None,
) -> list:
    """
    Stance breakdown (Agree / Neutral / Disagree %) per source for one claim.

    Uses stance_score threshold: >0.2 = agree, <-0.2 = disagree, else neutral.
    Optionally filter by article_ids (tuple of UUID strings) and/or stance_model.
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            article_clause = "AND article_id = ANY(%s::uuid[])" if article_ids else ""
            model_clause = "AND llm_model = %s" if stance_model else ""
            params = [claim_id]
            if article_ids:
                params.append(list(article_ids))
            if stance_model:
                params.append(stance_model)
            cur.execute(
                f"""
                SELECT
                    source_id,
                    COUNT(*)                                                             AS total,
                    SUM(CASE WHEN stance_score >  0.2 THEN 1 ELSE 0 END)::int          AS agree_count,
                    SUM(CASE WHEN stance_score >  0.2 THEN 1 ELSE 0 END) * 100.0
                        / COUNT(*)                                                       AS agree_pct,
                    SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END)::int
                                                                                         AS neutral_count,
                    SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END) * 100.0
                        / COUNT(*)                                                       AS neutral_pct,
                    SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END)::int          AS disagree_count,
                    SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END) * 100.0
                        / COUNT(*)                                                       AS disagree_pct
                FROM {schema}.claim_stance
                WHERE claim_id = %s {article_clause} {model_clause}
                GROUP BY source_id
                ORDER BY agree_pct DESC
                """,
                params,
            )
            return cur.fetchall()


def _stance_rows_to_df(stance_rows: list) -> pd.DataFrame:
    """Convert raw stance DB rows to a DataFrame with source_name and count columns."""
    records = []
    for row in stance_rows:
        records.append({
            "source_name":    SOURCE_NAMES.get(row["source_id"], str(row["source_id"])),
            "total":          int(row["total"]),
            "agree_pct":      float(row["agree_pct"]),
            "neutral_pct":    float(row["neutral_pct"]),
            "disagree_pct":   float(row["disagree_pct"]),
            "agree_count":    int(row["agree_count"]),
            "neutral_count":  int(row["neutral_count"]),
            "disagree_count": int(row["disagree_count"]),
        })
    return pd.DataFrame(records)


def generate_answer(question: str, articles: list, llm) -> str:
    """LLM generates a grounded answer from the top-10 retrieved article excerpts."""
    context_parts = []
    for i, art in enumerate(articles[:10], 1):
        source_name = SOURCE_NAMES.get(art["source_id"], art["source_id"])
        date_str = art["date_posted"].strftime("%d %B %Y") if art["date_posted"] else "Unknown date"
        snippet = (art["content"] or "").strip()[:800]
        context_parts.append(
            f"[Article {i}] {source_name} — {date_str}\n"
            f"Title: {art['title']}\n"
            f"Excerpt: {snippet}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a knowledgeable analyst who answers questions about Cyclone Ditwah "
        "based solely on Sri Lankan newspaper articles provided to you. "
        "Answer factually and concisely. Cite the source newspapers when relevant. "
        "If the articles do not contain enough information, say so clearly."
    )
    prompt = (
        f"Using only the following Sri Lankan newspaper article excerpts about Cyclone Ditwah, "
        f"answer this question:\n\nQuestion: {question}\n\nArticles:\n{context}\n\nAnswer:"
    )
    return llm.generate(prompt, system_prompt=system_prompt).content.strip()


def generate_hypothesis(question: str, llm) -> str:
    """Convert a user question into a declarative hypothesis for NLI stance analysis.

    e.g. "Did the government take proper actions?" →
         "The government took proper actions towards the disaster."
    """
    system_prompt = (
        "You are a media bias analyst. Convert a user's question into a single, "
        "clear declarative statement (hypothesis) that can be used to measure "
        "whether different media outlets agree or disagree with it."
    )
    prompt = (
        f"Convert this question into a concise declarative hypothesis statement "
        f"suitable for media bias analysis. The hypothesis should be a positive "
        f"assertion — do not use negations or hedging language.\n\n"
        f"Question: {question}\n\n"
        f"Return ONLY the hypothesis statement, nothing else."
    )
    response = llm.generate(prompt, system_prompt=system_prompt)
    return response.content.strip().strip('"').strip("'")


def generate_claims_from_articles(question: str, articles: list, llm) -> list:
    """Use LLM to extract 3-5 bias-revealing claims from the retrieved articles.

    Claims are tightly scoped to the reader's question and framed to expose
    how different media outlets cover and frame the same specific topic.
    """
    import json as _json

    context_parts = []
    for i, art in enumerate(articles, 1):
        source_name = SOURCE_NAMES.get(art["source_id"], art["source_id"])
        snippet = (art["content"] or "").strip()[:900]
        context_parts.append(
            f"[Article {i} — {source_name}]\n"
            f"Title: {art['title']}\n"
            f"Content: {snippet}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a senior media bias analyst studying how different Sri Lankan "
        "newspapers cover Cyclone Ditwah. You extract precise, question-focused "
        "claims from news articles to measure how different outlets agree, stay "
        "neutral, or push back on the same assertion — revealing editorial bias."
    )
    prompt = (
        f"READER'S QUESTION: \"{question}\"\n\n"
        f"NEWSPAPER ARTICLES:\n{context}\n\n"
        f"═══ INSTRUCTIONS ═══\n\n"
        f"Your goal is to generate 3–5 claims for MEDIA BIAS ANALYSIS. "
        f"Each claim will be scored by an NLI model against each article to "
        f"reveal which outlets agree, stay neutral, or disagree — exposing "
        f"editorial differences in coverage.\n\n"
        f"STEP 1 — Understand the question:\n"
        f"Identify the specific aspect the reader is asking about "
        f"(e.g. government response, relief operations, casualties, "
        f"infrastructure damage, international aid, economic impact).\n\n"
        f"STEP 2 — Extract claims that satisfy ALL of the following:\n"
        f"  ✔ ON-TOPIC: The claim must directly address the same specific "
        f"aspect as the reader's question. Do NOT include claims about "
        f"unrelated aspects of the cyclone.\n"
        f"  ✔ DECLARATIVE: A confident, positive assertion — not a question, "
        f"not hedged with 'may' or 'might', not a negation.\n"
        f"  ✔ SPECIFIC: Contains concrete details — named actors, institutions, "
        f"numbers, dates, locations, or specific actions.\n"
        f"  ✔ GROUNDED: Only from what the articles explicitly report. "
        f"Do not invent or infer facts not present in the text.\n"
        f"  ✔ BIAS-REVEALING: Different outlets should plausibly take "
        f"different stances on this claim — making it useful for detecting "
        f"how media frames the story.\n\n"
        f"GOOD example (for a question about government response):\n"
        f'  "The government declared a national emergency within 24 hours of '
        f'Cyclone Ditwah making landfall."\n'
        f'  "Relief supplies reached affected districts within 48 hours of '
        f'the disaster."\n\n'
        f"BAD example (too vague, off-topic, or not grounded):\n"
        f'  "The cyclone caused widespread damage."  ← too vague\n'
        f'  "Fishermen lost their boats."  ← off-topic if question is about govt response\n\n'
        f"Return ONLY a JSON array of claim strings — no explanation, "
        f"no numbering, no extra text:\n"
        f'["claim 1", "claim 2", "claim 3"]'
    )

    response = llm.generate(prompt, system_prompt=system_prompt)
    raw = response.content.strip()

    # Strip markdown code fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        claims = _json.loads(raw.strip())
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if c]
    except Exception:
        pass
    return []


def compute_stance_by_source(claim_text: str, articles: list, nli_analyzer) -> list:
    """Run NLI for one claim against all retrieved articles.

    Returns per-source stance rows in the same format as load_stance_by_source()
    so render_stance_chart() can be reused unchanged.
    """
    from collections import defaultdict

    premises = [
        f"{a['title']}\n{(a.get('content') or '').strip()}"
        for a in articles
    ]
    results = nli_analyzer.predict_batch(premises, claim_text)

    source_scores = defaultdict(list)
    for article, result in zip(articles, results):
        source_scores[article["source_id"]].append(result["stance_score"])

    rows = []
    for source_id, scores in source_scores.items():
        total          = len(scores)
        agree_count    = sum(1 for s in scores if s > 0.2)
        neutral_count  = sum(1 for s in scores if -0.2 <= s <= 0.2)
        disagree_count = sum(1 for s in scores if s < -0.2)
        rows.append({
            "source_id":      source_id,
            "total":          total,
            "agree_count":    agree_count,
            "agree_pct":      agree_count    * 100.0 / total,
            "neutral_count":  neutral_count,
            "neutral_pct":    neutral_count  * 100.0 / total,
            "disagree_count": disagree_count,
            "disagree_pct":   disagree_count * 100.0 / total,
        })

    return sorted(rows, key=lambda r: r["agree_pct"], reverse=True)


def render_stance_chart(stance_rows: list):
    """
    Render a vertical stacked bar chart of stance by source (agree / neutral / disagree).
    Returns a stance DataFrame for use in interpretation, or None if no data.
    """
    if not stance_rows:
        st.info("No stance data available for this claim.")
        return None

    stance_df = _stance_rows_to_df(stance_rows)

    fig = go.Figure()

    stance_categories = [
        ("agree_pct",    "agree_count",    "Agree",    "#2D6A4F"),
        ("neutral_pct",  "neutral_count",  "Neutral",  "#FFD93D"),
        ("disagree_pct", "disagree_count", "Disagree", "#C9184A"),
    ]

    for pct_col, count_col, label, color in stance_categories:
        fig.add_trace(go.Bar(
            name=label,
            x=stance_df["source_name"],
            y=stance_df[pct_col],
            marker_color=color,
            text=stance_df[pct_col].apply(lambda v: f"{v:.1f}%" if v >= 5 else ""),
            textposition="inside",
            textfont=dict(size=11, color="white" if label != "Neutral" else "black"),
            hovertemplate=(
                "<b>%{x}</b><br>"
                + label + ": %{y:.1f}%<br>"
                + "Count: " + stance_df[count_col].astype(str)
                + "<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        yaxis_title="Percentage of Articles (%)",
        xaxis_title="Source",
        height=400,
        yaxis_range=[0, 100],
        showlegend=True,
        hovermode="x unified",
        legend=dict(
            title="Stance",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    most_supportive = stance_df.loc[stance_df["agree_pct"].idxmax()]
    most_critical   = stance_df.loc[stance_df["disagree_pct"].idxmax()]
    st.caption(
        f"💡 **Most supportive:** {most_supportive['source_name']} "
        f"({most_supportive['agree_pct']:.1f}% agree) | "
        f"**Most critical:** {most_critical['source_name']} "
        f"({most_critical['disagree_pct']:.1f}% disagree)"
    )

    return stance_df


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("Settings")

    # --- Embedding model ---
    emb_models = get_available_embedding_models()
    if not emb_models:
        st.error("No Ditwah embeddings found.")
        st.stop()

    model_options = {
        f"{m['embedding_model']} ({m['article_count']} articles)": m["embedding_model"]
        for m in emb_models
    }
    selected_model = model_options[
        st.selectbox("Embedding model", list(model_options.keys()), index=0,
                     help="Question and articles are embedded with the same model.")
    ]

    st.divider()

    top_k = st.slider("Articles to retrieve", min_value=3, max_value=15, value=8, step=1,
                      help="Top-K articles by semantic similarity. Top 10 used for the answer.")
    show_full_excerpt = st.checkbox("Show longer excerpts", value=False)

    st.divider()
    total = get_total_ditwah_count()
    st.metric("Total Ditwah articles", total)


# ============================================================================
# Main UI
# ============================================================================

st.title("💬 Ditwah Cyclone Chatbot")
st.caption(
    "Ask any question about Cyclone Ditwah. Get an answer grounded in newspaper articles, "
    "then explore how each source stands on the related claims."
)

# --- Chat input ---
question = st.chat_input("Ask a question about Cyclone Ditwah…")

# When a new question is submitted, run the RAG pipeline and store in session_state
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Embedding question…"):
            embed_client = load_embedding_model(selected_model)
            q_embedding = embed_client.embed_single(question)

        with st.spinner("Searching articles…"):
            articles = search_ditwah_articles(q_embedding, selected_model, top_k)

        if not articles:
            st.warning("No relevant Ditwah articles found. Try rephrasing your question.")
            st.stop()

        with st.spinner("Generating answer…"):
            llm = load_llm()
            answer = generate_answer(question, articles, llm)

        with st.spinner("Generating hypothesis and bias-analysis claims…"):
            hypothesis = generate_hypothesis(question, llm)
            dynamic_claims = generate_claims_from_articles(question, articles, llm)

        # Persist in session state so reruns don't lose results
        st.session_state["chatbot_question"] = question
        st.session_state["chatbot_answer"] = answer
        st.session_state["chatbot_articles"] = [dict(a) for a in articles]
        st.session_state["chatbot_hypothesis"] = hypothesis
        st.session_state["chatbot_dynamic_claims"] = dynamic_claims
        st.session_state["chatbot_stance_cache"] = {}   # reset cache for new question

        st.markdown(answer)

# If we have a stored result (either just generated or from previous run), show everything
if "chatbot_question" in st.session_state:
    stored_question = st.session_state["chatbot_question"]
    stored_answer   = st.session_state["chatbot_answer"]
    stored_articles = st.session_state["chatbot_articles"]

    # If this is a rerun (claim selection) and not a fresh question, re-show the answer block
    if question is None:
        with st.chat_message("user"):
            st.markdown(stored_question)
        with st.chat_message("assistant"):
            st.markdown(stored_answer)

    article_ids = tuple(str(a["id"]) for a in stored_articles)
    n = len(stored_articles)

    st.divider()

    # ---- Article count + cards ----
    st.markdown(
        f"**{n} related article{'s' if n != 1 else ''} found** "
        f"(out of {total} total Ditwah articles)"
    )

    with st.expander("View source articles", expanded=False):
        st.caption(
            "Ranked by semantic similarity. "
            "Links may require access to the original news sources."
        )
        excerpt_len = 400 if show_full_excerpt else 200
        for rank, art in enumerate(stored_articles, 1):
            source_name  = SOURCE_NAMES.get(art["source_id"], art["source_id"])
            source_color = SOURCE_COLORS.get(source_name, "#888888")
            date_str = (
                art["date_posted"].strftime("%d %B %Y")
                if art.get("date_posted") else "Unknown date"
            )
            sim_pct = round(float(art["similarity"]) * 100, 1)
            raw     = (art.get("content") or "").strip()
            excerpt = raw[:excerpt_len] + ("…" if len(raw) > excerpt_len else "")

            with st.container(border=True):
                col_meta, col_sim = st.columns([5, 1])
                with col_meta:
                    st.markdown(
                        f"**#{rank}** &nbsp;"
                        f'<span style="background:{source_color};color:#fff;'
                        f'padding:2px 8px;border-radius:10px;font-size:0.8em;">'
                        f"{source_name}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{art['title']}**")
                    st.caption(date_str)
                with col_sim:
                    st.metric("Match", f"{sim_pct}%")
                st.markdown(excerpt)
                if art.get("url"):
                    st.markdown(f"[View full article]({art['url']})")
                else:
                    st.caption("No URL available")

    # ---- Question stance section ----
    hypothesis = st.session_state.get("chatbot_hypothesis")
    if hypothesis:
        st.divider()
        st.subheader("🔍 How do media outlets stand on your question?")
        st.caption(
            "Your question has been converted into a hypothesis. "
            "NLI (roberta-large-mnli) then scores each retrieved article against it, "
            "showing whether each newspaper source agrees, stays neutral, or disagrees."
        )
        st.info(f"**Hypothesis:** {hypothesis}")

        stance_cache = st.session_state.setdefault("chatbot_stance_cache", {})
        cache_key = f"__hypothesis__{hypothesis}"
        if cache_key not in stance_cache:
            with st.spinner("Running NLI stance analysis on hypothesis…"):
                nli_analyzer = load_nli_analyzer()
                stance_cache[cache_key] = compute_stance_by_source(
                    hypothesis, stored_articles, nli_analyzer
                )

        q_stance_df = render_stance_chart(stance_cache[cache_key])

        if q_stance_df is not None:
            st.markdown("---")
            st.subheader("📖 What Do These Charts Mean?")
            tab_q, = st.tabs(["⚖️ Stance Interpretation"])
            with tab_q:
                st.markdown(f"**Hypothesis:** *{hypothesis}*")
                st.markdown("---")
                st.markdown(generate_stance_interpretation(q_stance_df.copy(), hypothesis))

    # ---- Dynamic claims section ----
    dynamic_claims = st.session_state.get("chatbot_dynamic_claims", [])

    if dynamic_claims:
        st.divider()
        st.subheader("📋 Claims extracted for bias analysis")
        st.caption(
            "These claims were extracted from the retrieved articles specifically to "
            "reveal how different media outlets frame this topic. "
            "Select a claim to see the stance of each newspaper source."
        )

        selected_claim_text = st.selectbox(
            f"Choose a claim ({len(dynamic_claims)} extracted):",
            options=dynamic_claims,
            key="dynamic_claim_selector",
        )

        st.info(f"**Claim:** {selected_claim_text}")

        st.subheader("⚖️ Stance Distribution: Do sources agree or disagree with this claim?")
        st.caption(
            "NLI (roberta-large-mnli) scores each retrieved article against this claim. "
            "Shows whether each newspaper source agrees, stays neutral, or disagrees."
        )

        # Cache NLI results per claim to avoid recomputing on every widget rerun
        stance_cache = st.session_state.setdefault("chatbot_stance_cache", {})
        if selected_claim_text not in stance_cache:
            with st.spinner("Running NLI stance analysis…"):
                nli_analyzer = load_nli_analyzer()
                stance_cache[selected_claim_text] = compute_stance_by_source(
                    selected_claim_text, stored_articles, nli_analyzer
                )

        stance_rows = stance_cache[selected_claim_text]
        stance_df = render_stance_chart(stance_rows)

        if stance_df is not None:
            st.markdown("---")
            st.subheader("📖 What Do These Charts Mean?")
            tab_stance, = st.tabs(["⚖️ Stance Interpretation"])
            with tab_stance:
                st.markdown(f"**Claim:** *{selected_claim_text}*")
                st.markdown("---")
                st.markdown(generate_stance_interpretation(stance_df.copy(), selected_claim_text))

# Welcome state (no question asked yet)
elif "chatbot_question" not in st.session_state:
    st.info(
        "Ask a question in the chat box below — for example:\n\n"
        "- *How many people were displaced by Cyclone Ditwah?*\n"
        "- *What was the government's response to the cyclone?*\n"
        "- *Which areas were most affected by flooding?*\n"
        "- *What international aid was provided?*"
    )
