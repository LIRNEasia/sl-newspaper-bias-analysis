"""Ditwah Claims Analysis Page."""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from components.source_mapping import SOURCE_NAMES
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style
from components.interpretations import (
    generate_sentiment_interpretation,
    generate_stance_interpretation,
    generate_combined_bias_interpretation
)
from src.db import get_db
from src.llm import get_embeddings_client
from src.config import load_config

# Page config
st.set_page_config(
    page_title="Ditwah Claims - Sri Lanka Media Bias Detector",
    page_icon="🌀",
    layout="wide"
)

apply_page_style()


# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_data(ttl=300)
def load_ditwah_claims(version_id: str):
    """Load ALL claims for a version that have sentiment data."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT dc.*, COUNT(DISTINCT cs.article_id) as sentiment_count
                FROM {schema}.ditwah_claims dc
                LEFT JOIN {schema}.claim_sentiment cs ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id
                HAVING COUNT(DISTINCT cs.article_id) > 0
                ORDER BY dc.claim_order, dc.article_count DESC NULLS LAST
            """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def search_articles_by_content(question: str, version_id: str) -> list:
    """
    Search articles that contain the question keywords.

    Args:
        question: User's search query
        version_id: Version ID to filter articles

    Returns:
        List of article IDs that match the search
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Try to find articles via claim_sentiment (works for all pipeline versions)
            # This scopes the search to articles that are part of this version's analysis
            cur.execute(f"""
                SELECT DISTINCT n.id
                FROM {schema}.news_articles n
                JOIN {schema}.claim_sentiment cs ON n.id = cs.article_id
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                  AND (n.title ILIKE %s OR n.content ILIKE %s)
            """, (version_id, f"%{question}%", f"%{question}%"))
            results = [row['id'] for row in cur.fetchall()]

            # Fallback: if no results from claim_sentiment, search all ditwah articles
            if not results:
                cur.execute(f"""
                    SELECT DISTINCT n.id
                    FROM {schema}.news_articles n
                    WHERE n.is_ditwah_cyclone = 1
                      AND (n.title ILIKE %s OR n.content ILIKE %s)
                """, (f"%{question}%", f"%{question}%"))
                results = [row['id'] for row in cur.fetchall()]

            return results


@st.cache_data(ttl=300)
def filter_claims_by_question(claims: list, question: str, config: dict, version_id: str) -> list:
    """
    Filter claims using article-first search: find articles matching the question,
    then show claims from those articles. Combines with semantic similarity for ranking.

    Args:
        claims: List of claim dictionaries
        question: User's natural language question
        config: Application config
        version_id: Version ID for article search

    Returns:
        List of all relevant claims sorted by relevance (most relevant first)
    """
    if not question or not claims:
        return claims

    # Step 1: Article-First Search - Find articles matching the question
    matching_article_ids = search_articles_by_content(question, version_id)

    if not matching_article_ids:
        # No articles match the keyword search, fall back to semantic search only
        st.warning("🔍 No articles found with exact keyword matches. Trying semantic search...")

    # Step 2: Find claims associated with matching articles
    claim_article_counts = {}
    if matching_article_ids:
        with get_db() as db:
            schema = db.config["schema"]
            with db.cursor() as cur:
                # Count how many matching articles discuss each claim
                cur.execute(f"""
                    SELECT
                        claim_id,
                        COUNT(DISTINCT article_id) as matching_article_count
                    FROM {schema}.claim_sentiment
                    WHERE article_id = ANY(%s::uuid[])
                    GROUP BY claim_id
                """, (matching_article_ids,))

                for row in cur.fetchall():
                    claim_article_counts[row['claim_id']] = row['matching_article_count']

    # Step 3: Semantic similarity for additional ranking
    embeddings_client = get_embeddings_client(config, 'claims_analysis')
    question_embedding = embeddings_client.embed([question])[0]
    claim_texts = [c['claim_text'] for c in claims]
    claim_embeddings = embeddings_client.embed(claim_texts)
    similarities = cosine_similarity([question_embedding], claim_embeddings)[0]

    # Step 4: Combine article match count with semantic similarity
    claims_with_scores = []
    for claim, semantic_score in zip(claims, similarities):
        claim_copy = claim.copy()
        article_match_count = claim_article_counts.get(claim['id'], 0)

        # Calculate combined relevance score
        # - Article matches are weighted heavily (0.7)
        # - Semantic similarity provides additional context (0.3)
        # - Normalize article count by dividing by max(10, to keep it in reasonable range)
        article_score = min(article_match_count / 10.0, 1.0) if article_match_count > 0 else 0
        combined_score = (0.7 * article_score) + (0.3 * semantic_score)

        claim_copy['relevance_score'] = float(combined_score)
        claim_copy['matching_articles'] = article_match_count
        claim_copy['semantic_score'] = float(semantic_score)
        claims_with_scores.append(claim_copy)

    # Step 5: Filter by threshold - lower threshold since we're using article matches
    # Accept claims with either good article matches OR good semantic similarity
    relevant_claims = [
        c for c in claims_with_scores
        if c['matching_articles'] > 0 or c['semantic_score'] > 0.25
    ]

    # Sort by combined relevance score descending
    relevant_claims.sort(key=lambda x: x['relevance_score'], reverse=True)

    # Return all relevant claims (no limit)
    return relevant_claims


@st.cache_data(ttl=300)
def load_claim_sentiment_by_source(claim_id: str, article_ids: list = None):
    """Get average sentiment by source for a claim, optionally filtered by article IDs."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if article_ids:
                # Filter by specific article IDs when question is asked
                cur.execute(f"""
                    SELECT
                        cs.source_id,
                        AVG(cs.sentiment_score) as avg_sentiment,
                        STDDEV(cs.sentiment_score) as stddev_sentiment,
                        COUNT(*) as article_count
                    FROM {schema}.claim_sentiment cs
                    WHERE cs.claim_id = %s AND cs.article_id = ANY(%s::uuid[])
                    GROUP BY cs.source_id
                    ORDER BY avg_sentiment DESC
                """, (claim_id, article_ids))
            else:
                # Load all articles for this claim
                cur.execute(f"""
                    SELECT
                        cs.source_id,
                        AVG(cs.sentiment_score) as avg_sentiment,
                        STDDEV(cs.sentiment_score) as stddev_sentiment,
                        COUNT(*) as article_count
                    FROM {schema}.claim_sentiment cs
                    WHERE cs.claim_id = %s
                    GROUP BY cs.source_id
                    ORDER BY avg_sentiment DESC
                """, (claim_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_sentiment_breakdown(claim_id: str, article_ids: list = None):
    """Get sentiment distribution (very negative to very positive percentages) by source, optionally filtered by article IDs."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if article_ids:
                # Filter by specific article IDs when question is asked
                cur.execute(f"""
                    SELECT
                        source_id,
                        COUNT(*) as total,
                        SUM(CASE WHEN sentiment_score <= -3 THEN 1 ELSE 0 END)::int as very_negative_count,
                        SUM(CASE WHEN sentiment_score <= -3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as very_negative_pct,
                        SUM(CASE WHEN sentiment_score > -3 AND sentiment_score <= -1 THEN 1 ELSE 0 END)::int as negative_count,
                        SUM(CASE WHEN sentiment_score > -3 AND sentiment_score <= -1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as negative_pct,
                        SUM(CASE WHEN sentiment_score > -1 AND sentiment_score < 1 THEN 1 ELSE 0 END)::int as neutral_count,
                        SUM(CASE WHEN sentiment_score > -1 AND sentiment_score < 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as neutral_pct,
                        SUM(CASE WHEN sentiment_score >= 1 AND sentiment_score < 3 THEN 1 ELSE 0 END)::int as positive_count,
                        SUM(CASE WHEN sentiment_score >= 1 AND sentiment_score < 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as positive_pct,
                        SUM(CASE WHEN sentiment_score >= 3 THEN 1 ELSE 0 END)::int as very_positive_count,
                        SUM(CASE WHEN sentiment_score >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as very_positive_pct
                    FROM {schema}.claim_sentiment
                    WHERE claim_id = %s AND article_id = ANY(%s::uuid[])
                    GROUP BY source_id
                """, (claim_id, article_ids))
            else:
                # Load all articles for this claim
                cur.execute(f"""
                    SELECT
                        source_id,
                        COUNT(*) as total,
                        SUM(CASE WHEN sentiment_score <= -3 THEN 1 ELSE 0 END)::int as very_negative_count,
                        SUM(CASE WHEN sentiment_score <= -3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as very_negative_pct,
                        SUM(CASE WHEN sentiment_score > -3 AND sentiment_score <= -1 THEN 1 ELSE 0 END)::int as negative_count,
                        SUM(CASE WHEN sentiment_score > -3 AND sentiment_score <= -1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as negative_pct,
                        SUM(CASE WHEN sentiment_score > -1 AND sentiment_score < 1 THEN 1 ELSE 0 END)::int as neutral_count,
                        SUM(CASE WHEN sentiment_score > -1 AND sentiment_score < 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as neutral_pct,
                        SUM(CASE WHEN sentiment_score >= 1 AND sentiment_score < 3 THEN 1 ELSE 0 END)::int as positive_count,
                        SUM(CASE WHEN sentiment_score >= 1 AND sentiment_score < 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as positive_pct,
                        SUM(CASE WHEN sentiment_score >= 3 THEN 1 ELSE 0 END)::int as very_positive_count,
                        SUM(CASE WHEN sentiment_score >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as very_positive_pct
                    FROM {schema}.claim_sentiment
                    WHERE claim_id = %s
                    GROUP BY source_id
                """, (claim_id,))
            return cur.fetchall()


@st.cache_data(ttl=600)
def get_available_stance_models() -> list:
    """Return distinct llm_model values in claim_stance, sorted alphabetically."""
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
def load_claim_stance_breakdown(claim_id: str, article_ids: list = None, stance_model: str = None):
    """Get stance distribution (agree/neutral/disagree percentages) by source.

    Optionally filter by article IDs and/or stance_model (llm_model column).
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            model_clause = "AND llm_model = %s" if stance_model else ""
            if article_ids:
                params = [claim_id, article_ids] + ([stance_model] if stance_model else [])
                cur.execute(f"""
                    SELECT
                        source_id,
                        COUNT(*) as total,
                        SUM(CASE WHEN stance_score > 0.2 THEN 1 ELSE 0 END)::int as agree_count,
                        SUM(CASE WHEN stance_score > 0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as agree_pct,
                        SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END)::int as neutral_count,
                        SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as neutral_pct,
                        SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END)::int as disagree_count,
                        SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as disagree_pct
                    FROM {schema}.claim_stance
                    WHERE claim_id = %s AND article_id = ANY(%s::uuid[]) {model_clause}
                    GROUP BY source_id
                """, params)
            else:
                params = [claim_id] + ([stance_model] if stance_model else [])
                cur.execute(f"""
                    SELECT
                        source_id,
                        COUNT(*) as total,
                        SUM(CASE WHEN stance_score > 0.2 THEN 1 ELSE 0 END)::int as agree_count,
                        SUM(CASE WHEN stance_score > 0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as agree_pct,
                        SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END)::int as neutral_count,
                        SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as neutral_pct,
                        SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END)::int as disagree_count,
                        SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as disagree_pct
                    FROM {schema}.claim_stance
                    WHERE claim_id = %s {model_clause}
                    GROUP BY source_id
                """, params)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_articles(claim_id: str, limit: int = 10, article_ids: list = None):
    """Get sample articles for a claim with sentiment/stance scores, optionally filtered by article IDs."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if article_ids:
                # Filter by specific article IDs when question is asked
                cur.execute(f"""
                    SELECT
                        n.id,
                        n.title,
                        n.content,
                        n.date_posted,
                        n.url,
                        n.source_id,
                        cs_sentiment.sentiment_score,
                        cs_stance.stance_score,
                        cs_stance.stance_label,
                        cs_stance.supporting_quotes
                    FROM {schema}.claim_sentiment cs_sentiment
                    JOIN {schema}.claim_stance cs_stance
                        ON cs_sentiment.article_id = cs_stance.article_id
                        AND cs_sentiment.claim_id = cs_stance.claim_id
                    JOIN {schema}.news_articles n ON n.id = cs_sentiment.article_id
                    WHERE cs_sentiment.claim_id = %s AND cs_sentiment.article_id = ANY(%s::uuid[])
                    ORDER BY n.date_posted DESC
                    LIMIT %s
                """, (claim_id, article_ids, limit))
            else:
                # Load all articles for this claim
                cur.execute(f"""
                    SELECT
                        n.id,
                        n.title,
                        n.content,
                        n.date_posted,
                        n.url,
                        n.source_id,
                        cs_sentiment.sentiment_score,
                        cs_stance.stance_score,
                        cs_stance.stance_label,
                        cs_stance.supporting_quotes
                    FROM {schema}.claim_sentiment cs_sentiment
                    JOIN {schema}.claim_stance cs_stance
                        ON cs_sentiment.article_id = cs_stance.article_id
                        AND cs_sentiment.claim_id = cs_stance.claim_id
                    JOIN {schema}.news_articles n ON n.id = cs_sentiment.article_id
                    WHERE cs_sentiment.claim_id = %s
                    ORDER BY n.date_posted DESC
                    LIMIT %s
                """, (claim_id, limit))
            return cur.fetchall()


# ============================================================================
# Main Page
# ============================================================================

st.title("🌀 Cyclone Ditwah - Claims Analysis")
st.markdown("Analyze how different newspapers cover claims about Cyclone Ditwah")

# Version selector
version_id = render_version_selector('ditwah_claims')
render_create_version_button('ditwah_claims')

# Stance model selector
with st.sidebar:
    st.divider()
    _stance_models = get_available_stance_models()
    if _stance_models:
        selected_stance_model = st.selectbox(
            "Stance model",
            options=_stance_models,
            index=0,
            help="Which stance model's results to display. "
                 "roberta-large-mnli = NLI-based; others = LLM-based.",
        )
    else:
        selected_stance_model = None

if not version_id:
    st.info("👆 Select or create a ditwah_claims version to view analysis")
    st.stop()

st.markdown("---")

# Question-based search
col1, col2 = st.columns([3, 1])
with col1:
    user_question = st.text_input(
        "❓ Ask a question about Cyclone Ditwah coverage",
        placeholder="e.g., What aid did Sri Lanka receive? How did the government respond?",
        help="Ask any question and we'll find relevant claims and filter articles",
        key="ditwah_claims_question"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Question", key="ditwah_claims_clear"):
        # Clear the question input by clearing session state
        if "ditwah_claims_question" in st.session_state:
            del st.session_state["ditwah_claims_question"]
        st.rerun()

# Load all claims
all_claims = load_ditwah_claims(version_id)

# Claims are already filtered by sentiment_count > 0 in the SQL query

if not all_claims:
    st.warning("⚠️ No claims found with analysis data. Run the claim generation pipeline first.")
    with st.expander("🛠️ How to generate claims"):
        st.code("""
# 1. Mark Ditwah articles
python3 scripts/ditwah_claims/01_mark_ditwah_articles.py

# 2. Generate claims
python3 scripts/ditwah_claims/02_generate_claims.py --version-id <version-id>
        """)
    st.stop()

# Initialize variables for filtering
matching_article_ids = None
claims = all_claims

# Filter claims by semantic relevance if user asked a question
if user_question:
    config = load_config()

    # Get matching articles for the question
    matching_article_ids = search_articles_by_content(user_question, version_id)

    # Filter claims based on the question
    claims = filter_claims_by_question(all_claims, user_question, config, version_id)

    if not claims:
        st.warning(f"⚠️ No claims found relevant to: '{user_question}'. Try rephrasing your question.")
        st.info("💡 Tip: Try broader questions like 'What happened?' or 'How did sources respond?'")
        st.stop()

    st.success(f"Found {len(claims)} claims relevant to your question (from {len(all_claims)} total)")

    if matching_article_ids:
        st.info(f"📄 Filtering analysis to {len(matching_article_ids)} articles that match your question")

    # Show relevance scores for transparency
    with st.expander("🔍 Relevance Scores", expanded=False):
        st.markdown("**Relevance Score** = 70% article matches + 30% semantic similarity")
        for claim in claims[:5]:
            articles_text = f" ({claim['matching_articles']} matching articles)" if claim.get('matching_articles', 0) > 0 else ""
            st.write(f"**{claim['relevance_score']:.3f}**{articles_text} - {claim['claim_text'][:100]}...")
else:
    st.success(f"Showing all {len(claims)} claims")

# Claim selector with relevance indicators
claim_options = {}
for c in claims:
    label = f"{c['claim_text'][:100]}{'...' if len(c['claim_text']) > 100 else ''}"

    # Add relevance indicator if question was asked
    if user_question and 'relevance_score' in c:
        label += f" [{c['relevance_score']:.2f} relevance]"

    article_cnt = c['article_count'] if c.get('article_count') else c.get('sentiment_count', 0)
    label += f" ({article_cnt} articles, {c['claim_category'].replace('_', ' ').title()})"
    claim_options[label] = c['id']

selected_claim_label = st.selectbox(
    "📋 Select a claim to explore",
    options=list(claim_options.keys()),
    help="Choose a claim to see how different sources cover it",
    key="ditwah_claims_selector"
)

if not selected_claim_label:
    st.stop()

claim_id = claim_options[selected_claim_label]
claim = next(c for c in claims if c['id'] == claim_id)

# Display claim details
st.markdown("---")

# Show filtering indicator if active
if user_question and matching_article_ids:
    st.info(f"🔍 **Active Filter:** Showing analysis for articles matching: \"{user_question}\" ({len(matching_article_ids)} articles)")

st.subheader("Claim Details")
st.info(f"**{claim['claim_text']}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Category", claim['claim_category'].replace('_', ' ').title())
with col2:
    if user_question and matching_article_ids:
        st.metric("Matching Articles", len(matching_article_ids), help="Articles relevant to your question")
    else:
        st.metric("Articles Mentioning", claim['article_count'] if claim['article_count'] else 0)
with col3:
    # Calculate unique sources (filtered if question is asked)
    sentiment_data = load_claim_sentiment_by_source(claim_id, matching_article_ids)
    sources_count = len(sentiment_data) if sentiment_data else 0
    st.metric("Sources Covering", sources_count)

st.markdown("---")

# Initialize dataframes and interpretations (populated by chart sections below)
sent_df = pd.DataFrame()
stance_df = pd.DataFrame()
_sentiment_interpretation = None
_stance_interpretation = None

# Visualization 1: Sentiment Distribution (100% Stacked Bar)
st.subheader("📊 Sentiment Distribution: How do sources feel about this claim?")
if user_question and matching_article_ids:
    st.caption(f"Shows sentiment for articles matching your question: '{user_question}'. Hover over bars to see exact counts.")
else:
    st.caption("Shows what percentage of each source's articles fall into each sentiment category. Hover over bars to see exact counts.")

sentiment_breakdown = load_claim_sentiment_breakdown(claim_id, matching_article_ids)

if sentiment_breakdown and len(sentiment_breakdown) > 0:
    sent_df = pd.DataFrame(sentiment_breakdown)
    sent_df['source_name'] = sent_df['source_id'].map(lambda x: SOURCE_NAMES.get(x, f"Source {x}"))

    # Create 100% stacked bar chart using Plotly Graph Objects for better control
    fig = go.Figure()

    sentiment_categories = [
        ('very_negative_pct', 'very_negative_count', 'Very Negative', '#8B0000'),
        ('negative_pct', 'negative_count', 'Negative', '#FF6B6B'),
        ('neutral_pct', 'neutral_count', 'Neutral', '#FFD93D'),
        ('positive_pct', 'positive_count', 'Positive', '#6BCF7F'),
        ('very_positive_pct', 'very_positive_count', 'Very Positive', '#2D6A4F')
    ]

    for pct_col, count_col, label, color in sentiment_categories:
        fig.add_trace(go.Bar(
            name=label,
            x=sent_df['source_name'],
            y=sent_df[pct_col],
            marker_color=color,
            text=sent_df[pct_col].apply(lambda x: f'{x:.1f}%' if x >= 5 else ''),
            textposition='inside',
            textfont=dict(size=11, color='white'),
            hovertemplate='<b>%{x}</b><br>' +
                          label + ': %{y:.1f}%<br>' +
                          'Count: ' + sent_df[count_col].astype(str) + '<extra></extra>'
        ))

    fig.update_layout(
        barmode='stack',
        yaxis_title="Percentage of Articles (%)",
        xaxis_title="Source",
        height=400,
        yaxis_range=[0, 100],
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            title="Sentiment",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary insights
    if len(sent_df) > 0:
        # Find most positive source (highest positive + very_positive)
        sent_df['total_positive'] = sent_df['positive_pct'] + sent_df['very_positive_pct']
        sent_df['total_negative'] = sent_df['negative_pct'] + sent_df['very_negative_pct']

        most_positive = sent_df.loc[sent_df['total_positive'].idxmax()]
        most_negative = sent_df.loc[sent_df['total_negative'].idxmax()]

        st.caption(f"💡 **Most positive coverage:** {most_positive['source_name']} "
                   f"({most_positive['total_positive']:.1f}% positive) | "
                   f"**Most negative coverage:** {most_negative['source_name']} "
                   f"({most_negative['total_negative']:.1f}% negative)")

        # Store for interpretation section below
        _sentiment_interpretation = generate_sentiment_interpretation(sent_df, claim['claim_text'])
else:
    if user_question and matching_article_ids:
        st.warning(f"⚠️ No sentiment data found for articles matching your question: '{user_question}'. Try selecting a different claim or rephrasing your question.")
    else:
        st.warning("⚠️ No sentiment data available for this claim")

st.markdown("---")

# Visualization 2: Stance Distribution (100% Stacked Bar)
st.subheader("⚖️ Stance Distribution: Do sources agree or disagree with this claim?")
if user_question and matching_article_ids:
    st.caption(f"Shows stance for articles matching your question: '{user_question}'. Hover over bars to see exact counts.")
else:
    st.caption("Shows what percentage of each source's articles agree, are neutral, or disagree with the claim. Hover over bars to see exact counts.")

stance_breakdown = load_claim_stance_breakdown(claim_id, matching_article_ids, selected_stance_model)

if stance_breakdown and len(stance_breakdown) > 0:
    stance_df = pd.DataFrame(stance_breakdown)
    stance_df['source_name'] = stance_df['source_id'].map(lambda x: SOURCE_NAMES.get(x, f"Source {x}"))

    # Create 100% stacked bar chart using Plotly Graph Objects for better control
    fig = go.Figure()

    stance_categories = [
        ('agree_pct', 'agree_count', 'Agree', '#2D6A4F'),
        ('neutral_pct', 'neutral_count', 'Neutral', '#FFD93D'),
        ('disagree_pct', 'disagree_count', 'Disagree', '#C9184A')
    ]

    for pct_col, count_col, label, color in stance_categories:
        fig.add_trace(go.Bar(
            name=label,
            x=stance_df['source_name'],
            y=stance_df[pct_col],
            marker_color=color,
            text=stance_df[pct_col].apply(lambda x: f'{x:.1f}%' if x >= 5 else ''),
            textposition='inside',
            textfont=dict(size=11, color='white' if label != 'Neutral' else 'black'),
            hovertemplate='<b>%{x}</b><br>' +
                          label + ': %{y:.1f}%<br>' +
                          'Count: ' + stance_df[count_col].astype(str) + '<extra></extra>'
        ))

    fig.update_layout(
        barmode='stack',
        yaxis_title="Percentage of Articles (%)",
        xaxis_title="Source",
        height=400,
        yaxis_range=[0, 100],
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            title="Stance",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary insights
    if len(stance_df) > 0:
        most_supportive = stance_df.loc[stance_df['agree_pct'].idxmax()]
        most_critical = stance_df.loc[stance_df['disagree_pct'].idxmax()]

        st.caption(f"💡 **Most supportive:** {most_supportive['source_name']} "
                   f"({most_supportive['agree_pct']:.1f}% agree) | "
                   f"**Most critical:** {most_critical['source_name']} "
                   f"({most_critical['disagree_pct']:.1f}% disagree)")

        # Store for interpretation section below
        _stance_interpretation = generate_stance_interpretation(stance_df, claim['claim_text'])
else:
    if user_question and matching_article_ids:
        st.warning(f"⚠️ No stance data found for articles matching your question: '{user_question}'. Try selecting a different claim or rephrasing your question.")
    else:
        st.warning("⚠️ No stance data available for this claim")

st.markdown("---")

# Plain Language Interpretation Section
st.subheader("📖 What Do These Charts Mean?")
st.caption("Plain language explanations of the sentiment and stance patterns shown above.")

has_sentiment = _sentiment_interpretation is not None
has_stance = _stance_interpretation is not None

if has_sentiment or has_stance:
    if has_sentiment and has_stance:
        tab_sent, tab_stance, tab_combined = st.tabs(["😊 Sentiment Interpretation", "⚖️ Stance Interpretation", "🎯 Combined Bias Analysis"])
    elif has_sentiment:
        tab_sent, = st.tabs(["😊 Sentiment Interpretation"])
    else:
        tab_stance, = st.tabs(["⚖️ Stance Interpretation"])

    if has_sentiment:
        with tab_sent:
            st.markdown(f"**Claim:** *{claim['claim_text']}*")
            st.markdown("---")
            st.markdown(_sentiment_interpretation)

    if has_stance:
        with tab_stance:
            st.markdown(f"**Claim:** *{claim['claim_text']}*")
            st.markdown("---")
            st.markdown(_stance_interpretation)

    if has_sentiment and has_stance:
        with tab_combined:
            st.markdown(f"**Claim:** *{claim['claim_text']}*")
            st.markdown("---")
            bias_interpretation = generate_combined_bias_interpretation(sent_df, stance_df, claim['claim_text'])
            st.markdown(bias_interpretation)
else:
    st.info("Interpretations will appear here once sentiment and stance data are available for this claim.")

st.markdown("---")

# Sample Articles
st.subheader("📰 Sample Articles Mentioning This Claim")
if user_question and matching_article_ids:
    st.caption(f"Showing articles that match your question: '{user_question}'")

articles = load_claim_articles(claim_id, limit=5, article_ids=matching_article_ids)
if articles and len(articles) > 0:
    for article in articles:
        source_name = SOURCE_NAMES.get(article['source_id'], article['source_id'])
        with st.expander(f"**{source_name}** - {article['title'][:100]}..."):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Sentiment", f"{article['sentiment_score']:.2f}")
                st.metric("Stance", f"{article['stance_score']:.2f}")
            with col2:
                st.markdown(f"**Published:** {article['date_posted'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Excerpt:** {article['content'][:300]}...")
                if article['supporting_quotes']:
                    quotes = article['supporting_quotes'] if isinstance(article['supporting_quotes'], list) else []
                    if quotes:
                        st.markdown("**Key Quotes:**")
                        for quote in quotes[:2]:
                            st.markdown(f"> {quote}")
                st.markdown(f"[Read full article]({article['url']})")
else:
    if user_question and matching_article_ids:
        st.info(f"ℹ️ No articles found that match your question: '{user_question}' for this specific claim. Try selecting a different claim.")
    else:
        st.info("ℹ️ No articles found with complete sentiment and stance data for this claim.")
