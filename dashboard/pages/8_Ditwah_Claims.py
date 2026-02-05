"""Ditwah Claims Analysis Page."""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from components.source_mapping import SOURCE_NAMES
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style
from src.db import get_db

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
def load_ditwah_claims(version_id: str, keyword: Optional[str] = None):
    """Load claims, optionally filtered by keyword."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if keyword:
                keyword_pattern = f"%{keyword.lower()}%"
                cur.execute(f"""
                    SELECT * FROM {schema}.ditwah_claims
                    WHERE result_version_id = %s
                      AND LOWER(claim_text) LIKE %s
                    ORDER BY claim_order, article_count DESC
                    LIMIT 50
                """, (version_id, keyword_pattern))
            else:
                cur.execute(f"""
                    SELECT * FROM {schema}.ditwah_claims
                    WHERE result_version_id = %s
                    ORDER BY claim_order, article_count DESC
                """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_sentiment_by_source(claim_id: str):
    """Get average sentiment by source for a claim."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
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
def load_claim_sentiment_breakdown(claim_id: str):
    """Get sentiment distribution (very negative to very positive percentages) by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
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


@st.cache_data(ttl=300)
def load_claim_stance_breakdown(claim_id: str):
    """Get stance distribution (agree/neutral/disagree percentages) by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
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
                WHERE claim_id = %s
                GROUP BY source_id
            """, (claim_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_articles(claim_id: str, limit: int = 10):
    """Get sample articles for a claim with sentiment/stance scores."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
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

if not version_id:
    st.info("👆 Select or create a ditwah_claims version to view analysis")
    st.stop()

st.markdown("---")

# Keyword search
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input(
        "🔍 Search claims by keyword",
        placeholder="e.g., government, aid, casualties, damage",
        help="Enter keywords to filter claims",
        key="ditwah_claims_search"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear Search", key="ditwah_claims_clear"):
        st.rerun()

# Load claims
claims = load_ditwah_claims(version_id, search_query if search_query else None)

if not claims:
    st.warning("⚠️ No claims found. Run the claim generation pipeline first.")
    with st.expander("🛠️ How to generate claims"):
        st.code("""
# 1. Mark Ditwah articles
python3 scripts/ditwah_claims/01_mark_ditwah_articles.py

# 2. Generate claims
python3 scripts/ditwah_claims/02_generate_claims.py --version-id <version-id>
        """)
    st.stop()

st.success(f"Found {len(claims)} claims")

# Claim selector
claim_options = {
    f"{c['claim_text'][:100]}{'...' if len(c['claim_text']) > 100 else ''} "
    f"({c['article_count']} articles, {c['claim_category'].replace('_', ' ').title()})": c['id']
    for c in claims
}

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
st.subheader("Claim Details")
st.info(f"**{claim['claim_text']}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Category", claim['claim_category'].replace('_', ' ').title())
with col2:
    st.metric("Articles Mentioning", claim['article_count'] if claim['article_count'] else 0)
with col3:
    # Calculate unique sources
    sentiment_data = load_claim_sentiment_by_source(claim_id)
    sources_count = len(sentiment_data) if sentiment_data else 0
    st.metric("Sources Covering", sources_count)

st.markdown("---")

# Visualization 1: Sentiment Distribution (100% Stacked Bar)
st.subheader("📊 Sentiment Distribution: How do sources feel about this claim?")
st.caption("Shows what percentage of each source's articles fall into each sentiment category. Hover over bars to see exact counts.")

sentiment_breakdown = load_claim_sentiment_breakdown(claim_id)

if sentiment_breakdown:
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
else:
    st.warning("No sentiment data available for this claim")

st.markdown("---")

# Visualization 2: Stance Distribution (100% Stacked Bar)
st.subheader("⚖️ Stance Distribution: Do sources agree or disagree with this claim?")
st.caption("Shows what percentage of each source's articles agree, are neutral, or disagree with the claim. Hover over bars to see exact counts.")

stance_breakdown = load_claim_stance_breakdown(claim_id)

if stance_breakdown:
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
else:
    st.warning("No stance data available for this claim")

st.markdown("---")

# Sample Articles
st.subheader("📰 Sample Articles Mentioning This Claim")

articles = load_claim_articles(claim_id, limit=5)
if articles:
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
    st.info("No articles found with complete sentiment and stance data for this claim.")
