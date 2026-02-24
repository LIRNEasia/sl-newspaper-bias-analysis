"""Stance Analysis Page."""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style
from src.db import get_db

# Page config
st.set_page_config(
    page_title="Stance Analysis - Sri Lanka Media Bias Detector",
    page_icon="⚖️",
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
                    SELECT
                        dc.id,
                        dc.result_version_id,
                        dc.claim_text,
                        dc.claim_category as category,
                        dc.claim_order,
                        dc.article_count,
                        dc.individual_claims_count,
                        dc.representative_article_id,
                        dc.llm_provider,
                        dc.llm_model,
                        dc.created_at,
                        COUNT(DISTINCT cs.source_id) as source_count
                    FROM {schema}.ditwah_claims dc
                    LEFT JOIN {schema}.claim_stance cs ON cs.claim_id = dc.id
                    WHERE dc.result_version_id = %s
                      AND LOWER(dc.claim_text) LIKE %s
                    GROUP BY dc.id, dc.result_version_id, dc.claim_text, dc.claim_category,
                             dc.claim_order, dc.article_count, dc.individual_claims_count,
                             dc.representative_article_id, dc.llm_provider, dc.llm_model, dc.created_at
                    ORDER BY dc.claim_order, dc.article_count DESC
                    LIMIT 50
                """, (version_id, keyword_pattern))
            else:
                cur.execute(f"""
                    SELECT
                        dc.id,
                        dc.result_version_id,
                        dc.claim_text,
                        dc.claim_category as category,
                        dc.claim_order,
                        dc.article_count,
                        dc.individual_claims_count,
                        dc.representative_article_id,
                        dc.llm_provider,
                        dc.llm_model,
                        dc.created_at,
                        COUNT(DISTINCT cs.source_id) as source_count
                    FROM {schema}.ditwah_claims dc
                    LEFT JOIN {schema}.claim_stance cs ON cs.claim_id = dc.id
                    WHERE dc.result_version_id = %s
                    GROUP BY dc.id, dc.result_version_id, dc.claim_text, dc.claim_category,
                             dc.claim_order, dc.article_count, dc.individual_claims_count,
                             dc.representative_article_id, dc.llm_provider, dc.llm_model, dc.created_at
                    ORDER BY dc.claim_order, dc.article_count DESC
                """, (version_id,))
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
def load_claim_stance_breakdown(claim_id: str, stance_model: str = None):
    """Get stance distribution (agree/neutral/disagree percentages) by source.

    Optionally filter by stance_model (llm_model column).
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            model_clause = "AND llm_model = %s" if stance_model else ""
            params = (claim_id, stance_model) if stance_model else (claim_id,)
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


def load_stance_overview(version_id: str) -> dict:
    """Get high-level stance statistics."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Total claims with stance data
            cur.execute(f"""
                SELECT COUNT(DISTINCT claim_id) as total_claims
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
            """, (version_id,))
            total_claims = cur.fetchone()['total_claims']

            # Most controversial claim (highest stddev in stance_score)
            cur.execute(f"""
                SELECT
                    dc.id,
                    dc.claim_text,
                    STDDEV(cs.stance_score) as controversy
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id, dc.claim_text
                ORDER BY controversy DESC
                LIMIT 1
            """, (version_id,))
            most_controversial = cur.fetchone()

            # Strongest consensus claim (lowest stddev)
            cur.execute(f"""
                SELECT
                    dc.id,
                    dc.claim_text,
                    AVG(cs.stance_score) as avg_stance,
                    STDDEV(cs.stance_score) as controversy
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id, dc.claim_text
                HAVING COUNT(DISTINCT cs.source_id) >= 2
                ORDER BY controversy ASC
                LIMIT 1
            """, (version_id,))
            strongest_consensus = cur.fetchone()

            # Average confidence
            cur.execute(f"""
                SELECT AVG(cs.confidence) as avg_confidence
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
            """, (version_id,))
            avg_confidence = cur.fetchone()['avg_confidence']

            return {
                'total_claims': total_claims,
                'most_controversial': most_controversial,
                'strongest_consensus': strongest_consensus,
                'avg_confidence': avg_confidence
            }


@st.cache_data(ttl=300)
def load_stance_polarization_matrix(version_id: str, category_filter: Optional[str] = None) -> pd.DataFrame:
    """Get claim × source heatmap data."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            category_clause = "AND dc.claim_category = %s" if category_filter else ""
            params = [version_id, category_filter] if category_filter else [version_id]

            cur.execute(f"""
                WITH claim_controversy AS (
                    SELECT
                        cs_all.claim_id,
                        STDDEV(cs_all.stance_score) as controversy_index
                    FROM {schema}.claim_stance cs_all
                    JOIN {schema}.ditwah_claims dc_all ON cs_all.claim_id = dc_all.id
                    WHERE dc_all.result_version_id = %s {category_clause}
                    GROUP BY cs_all.claim_id
                )
                SELECT
                    dc.id as claim_id,
                    dc.claim_text,
                    dc.claim_category as category,
                    cs.source_id,
                    AVG(cs.stance_score) as avg_stance,
                    AVG(cs.confidence) as avg_confidence,
                    cc.controversy_index,
                    COUNT(cs.id) as article_count
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                JOIN claim_controversy cc ON cc.claim_id = dc.id
                WHERE dc.result_version_id = %s {category_clause}
                GROUP BY dc.id, dc.claim_text, dc.claim_category, cs.source_id, cc.controversy_index
                ORDER BY cc.controversy_index DESC, dc.claim_text, cs.source_id
            """, params + params)

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_source_alignment_matrix(version_id: str) -> pd.DataFrame:
    """Calculate source-to-source alignment scores."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                WITH source_stances AS (
                    SELECT
                        cs.claim_id,
                        cs.source_id,
                        CASE
                            WHEN cs.stance_score > 0.2 THEN 'agree'
                            WHEN cs.stance_score < -0.2 THEN 'disagree'
                            ELSE 'neutral'
                        END as stance_category
                    FROM {schema}.claim_stance cs
                    JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                    WHERE dc.result_version_id = %s
                )
                SELECT
                    s1.source_id as source1,
                    s2.source_id as source2,
                    COUNT(*) as total_claims,
                    SUM(CASE WHEN s1.stance_category = s2.stance_category THEN 1 ELSE 0 END) as agree_count,
                    SUM(CASE WHEN s1.stance_category != s2.stance_category THEN 1 ELSE 0 END) as disagree_count,
                    ROUND(SUM(CASE WHEN s1.stance_category = s2.stance_category THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as alignment_pct
                FROM source_stances s1
                JOIN source_stances s2 ON s1.claim_id = s2.claim_id AND s1.source_id < s2.source_id
                GROUP BY s1.source_id, s2.source_id
                ORDER BY alignment_pct DESC
            """, (version_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_confidence_weighted_stances(version_id: str) -> pd.DataFrame:
    """Get bubble chart data."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    dc.id as claim_id,
                    dc.claim_text,
                    dc.claim_category as category,
                    AVG(cs.stance_score) as avg_stance,
                    STDDEV(cs.stance_score) as stddev_stance,
                    AVG(cs.confidence) as avg_confidence,
                    COUNT(DISTINCT cs.article_id) as article_count,
                    COUNT(DISTINCT cs.source_id) as source_count
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id, dc.claim_text, dc.claim_category
                HAVING COUNT(DISTINCT cs.article_id) >= 2
                ORDER BY stddev_stance DESC
            """, (version_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_claim_source_comparison(claim_id: str) -> pd.DataFrame:
    """Get detailed comparison for a single claim across all sources."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    cs.source_id,
                    AVG(cs.stance_score) as avg_stance,
                    cs.stance_label,
                    AVG(cs.confidence) as avg_confidence,
                    COUNT(cs.article_id) as article_count,
                    (ARRAY_AGG(cs.supporting_quotes ORDER BY cs.processed_at DESC))[1] as sample_quotes
                FROM {schema}.claim_stance cs
                WHERE cs.claim_id = %s
                GROUP BY cs.source_id, cs.stance_label
                ORDER BY avg_stance DESC
            """, (claim_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_claim_quotes_by_stance(claim_id: str) -> pd.DataFrame:
    """Get supporting quotes grouped by stance."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    cs.stance_label,
                    cs.stance_score,
                    cs.supporting_quotes,
                    cs.source_id,
                    n.title as article_title,
                    n.id as article_id,
                    n.date_posted
                FROM {schema}.claim_stance cs
                JOIN {schema}.news_articles n ON cs.article_id = n.id
                WHERE cs.claim_id = %s
                  AND cs.supporting_quotes IS NOT NULL
                  AND jsonb_array_length(cs.supporting_quotes) > 0
                ORDER BY cs.stance_score DESC, n.date_posted DESC
            """, (claim_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


# ============================================================================
# Helper Render Functions
# ============================================================================

def render_stance_overview_section(overview: dict):
    """Render overview metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Claims",
            value=overview['total_claims'],
            help="Number of claims with stance analysis"
        )

    with col2:
        if overview['most_controversial']:
            controversy_score = overview['most_controversial']['controversy'] or 0
            st.metric(
                label="Most Controversial",
                value=f"{controversy_score:.2f}",
                help="Highest disagreement (stddev of stance scores)"
            )
            st.caption(f"_{overview['most_controversial']['claim_text'][:50]}..._")
        else:
            st.metric("Most Controversial", "N/A")

    with col3:
        if overview['strongest_consensus']:
            consensus_score = overview['strongest_consensus']['controversy'] or 0
            st.metric(
                label="Strongest Consensus",
                value=f"{consensus_score:.3f}",
                help="Lowest disagreement (stddev of stance scores)"
            )
            st.caption(f"_{overview['strongest_consensus']['claim_text'][:50]}..._")
        else:
            st.metric("Strongest Consensus", "N/A")

    with col4:
        confidence_pct = (overview['avg_confidence'] or 0) * 100
        st.metric(
            label="Avg Confidence",
            value=f"{confidence_pct:.1f}%",
            help="Average confidence across all stance predictions"
        )


def render_polarization_dashboard(version_id: str):
    """Render polarization heatmap and controversy analysis."""
    # Load data
    df = load_stance_polarization_matrix(version_id)

    if df.empty:
        st.info("No stance data available for polarization analysis.")
        return

    # Filter controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
        selected_category = st.selectbox("Filter by Category", categories, key="polar_category")

    with col2:
        min_articles = st.slider("Min Articles", 1, int(df['article_count'].max()), 2, key="polar_min_articles")

    with col3:
        sort_by = st.selectbox("Sort by", ["Controversy (High)", "Controversy (Low)", "Alphabetical"], key="polar_sort")

    with col4:
        show_count = st.slider("Show Top N Claims", 10, 50, 20, key="polar_count")

    # Apply filters
    filtered_df = df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    filtered_df = filtered_df[filtered_df['article_count'] >= min_articles]

    # Sort
    if sort_by == "Controversy (High)":
        filtered_df = filtered_df.sort_values('controversy_index', ascending=False)
    elif sort_by == "Controversy (Low)":
        filtered_df = filtered_df.sort_values('controversy_index', ascending=True)
    else:
        filtered_df = filtered_df.sort_values('claim_text')

    # Get top N claims
    top_claims = filtered_df['claim_id'].unique()[:show_count]
    plot_df = filtered_df[filtered_df['claim_id'].isin(top_claims)]

    if plot_df.empty:
        st.warning("No claims match the selected filters.")
        return

    # Create heatmap
    pivot_df = plot_df.pivot_table(
        index='claim_text',
        columns='source_id',
        values='avg_stance',
        aggfunc='mean'
    )

    # Truncate claim text for display
    pivot_df.index = [text[:60] + '...' if len(text) > 60 else text for text in pivot_df.index]

    fig = px.imshow(
        pivot_df,
        labels=dict(x="News Source", y="Claim", color="Stance Score"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        aspect="auto",
        title="Claim × Source Stance Heatmap"
    )

    fig.update_layout(
        height=max(400, len(pivot_df.index) * 25),
        xaxis_title="News Source",
        yaxis_title="",
        font=dict(size=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Controversy ranking
    st.markdown("#### 📈 Controversy Ranking")
    controversy_df = plot_df.groupby(['claim_id', 'claim_text', 'controversy_index']).first().reset_index()
    controversy_df = controversy_df.sort_values('controversy_index', ascending=False).head(10)

    for idx, row in controversy_df.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{row['claim_text']}**")
            st.caption(f"Category: {row['category']}")
        with col2:
            st.metric("Controversy", f"{row['controversy_index']:.2f}")


def render_source_alignment(version_id: str):
    """Render source-to-source alignment matrix."""
    df = load_source_alignment_matrix(version_id)

    if df.empty:
        st.info("Not enough data for source alignment analysis.")
        return

    # Create alignment matrix visualization
    sources = sorted(set(df['source1'].tolist() + df['source2'].tolist()))

    # Build full matrix (including diagonal)
    matrix_data = []
    for s1 in sources:
        row = []
        for s2 in sources:
            if s1 == s2:
                row.append(100.0)  # Perfect self-alignment
            else:
                # Find alignment percentage
                match = df[((df['source1'] == s1) & (df['source2'] == s2)) |
                          ((df['source1'] == s2) & (df['source2'] == s1))]
                if not match.empty:
                    row.append(match.iloc[0]['alignment_pct'])
                else:
                    row.append(None)
        matrix_data.append(row)

    # Create heatmap
    fig = px.imshow(
        matrix_data,
        labels=dict(x="Source", y="Source", color="Alignment %"),
        x=sources,
        y=sources,
        color_continuous_scale='Blues',
        title="Source-to-Source Alignment Matrix",
        text_auto='.1f'
    )

    fig.update_layout(
        height=500,
        xaxis_title="",
        yaxis_title=""
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed alignment table
    st.markdown("#### 📊 Alignment Details")
    display_df = df[['source1', 'source2', 'alignment_pct', 'agree_count', 'disagree_count', 'total_claims']].copy()
    display_df.columns = ['Source 1', 'Source 2', 'Alignment %', 'Agreements', 'Disagreements', 'Total Claims']
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_confidence_explorer(version_id: str):
    """Render confidence-weighted bubble chart."""
    df = load_confidence_weighted_stances(version_id)

    if df.empty:
        st.info("No data available for confidence analysis.")
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.5, 0.05, key="conf_min")
    with col2:
        categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
        selected_category = st.selectbox("Category", categories, key="conf_category")

    # Apply filters
    filtered_df = df[df['avg_confidence'] >= min_confidence].copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]

    if filtered_df.empty:
        st.warning("No claims match the selected filters.")
        return

    # Create bubble chart
    filtered_df['claim_short'] = filtered_df['claim_text'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

    fig = px.scatter(
        filtered_df,
        x='avg_stance',
        y='stddev_stance',
        size='article_count',
        color='avg_confidence',
        hover_data=['claim_short', 'category', 'source_count'],
        labels={
            'avg_stance': 'Average Stance Score',
            'stddev_stance': 'Controversy (Std Dev)',
            'avg_confidence': 'Confidence',
            'article_count': 'Article Count'
        },
        title="Stance Position vs. Controversy (sized by article count)",
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=600,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Quadrant analysis
    st.markdown("#### 📍 Quadrant Analysis")
    col1, col2, col3, col4 = st.columns(4)

    high_agree_high_controversy = filtered_df[(filtered_df['avg_stance'] > 0.3) & (filtered_df['stddev_stance'] > 0.3)]
    high_agree_low_controversy = filtered_df[(filtered_df['avg_stance'] > 0.3) & (filtered_df['stddev_stance'] <= 0.3)]
    high_disagree_high_controversy = filtered_df[(filtered_df['avg_stance'] < -0.3) & (filtered_df['stddev_stance'] > 0.3)]
    high_disagree_low_controversy = filtered_df[(filtered_df['avg_stance'] < -0.3) & (filtered_df['stddev_stance'] <= 0.3)]

    with col1:
        st.metric("High Agree + Controversial", len(high_agree_high_controversy))
    with col2:
        st.metric("High Agree + Consensus", len(high_agree_low_controversy))
    with col3:
        st.metric("High Disagree + Controversial", len(high_disagree_high_controversy))
    with col4:
        st.metric("High Disagree + Consensus", len(high_disagree_low_controversy))


def render_claim_deep_dive(version_id: str, stance_model: str = None):
    """Render detailed claim analysis with progressive disclosure."""
    # Load all claims for dropdown
    claims = load_ditwah_claims(version_id)

    if not claims:
        st.info("No claims available.")
        return

    # Create searchable dropdown
    claim_options = {f"{claim['claim_text'][:80]}... ({claim['category']})": claim['id']
                     for claim in claims}

    selected_display = st.selectbox(
        "Select a claim to analyze",
        options=list(claim_options.keys()),
        key="deep_dive_claim"
    )

    if not selected_display:
        return

    claim_id = claim_options[selected_display]
    selected_claim = next(c for c in claims if c['id'] == claim_id)

    # Claim header
    st.markdown(f"### {selected_claim['claim_text']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Category", selected_claim['category'])
    with col2:
        st.metric("Total Articles", selected_claim['article_count'])
    with col3:
        st.metric("Sources", selected_claim['source_count'])

    st.markdown("---")

    # Stance distribution (reuse existing function)
    st.markdown("#### 📊 Stance Distribution by Source")
    stance_breakdown = load_claim_stance_breakdown(claim_id, stance_model)

    if stance_breakdown:
        # Create stacked bar chart data
        sources = []
        agree_pcts = []
        neutral_pcts = []
        disagree_pcts = []

        for row in stance_breakdown:
            sources.append(row['source_id'])
            agree_pcts.append(row['agree_pct'])
            neutral_pcts.append(row['neutral_pct'])
            disagree_pcts.append(row['disagree_pct'])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Agree',
            x=sources,
            y=agree_pcts,
            marker_color='#2D6A4F'
        ))
        fig.add_trace(go.Bar(
            name='Neutral',
            x=sources,
            y=neutral_pcts,
            marker_color='#FFD93D'
        ))
        fig.add_trace(go.Bar(
            name='Disagree',
            x=sources,
            y=disagree_pcts,
            marker_color='#C9184A'
        ))

        fig.update_layout(
            barmode='stack',
            title='Stance Distribution (%)',
            yaxis_title='Percentage',
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed breakdown table
        breakdown_df = pd.DataFrame(stance_breakdown)
        breakdown_df = breakdown_df[['source_id', 'agree_count', 'neutral_count', 'disagree_count', 'total']]
        breakdown_df.columns = ['Source', 'Agree', 'Neutral', 'Disagree', 'Total']
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # Sub-tabs for detailed exploration
    tab1, tab2, tab3 = st.tabs(["📝 Supporting Quotes", "📰 Article List", "⚖️ Source Comparison"])

    with tab1:
        render_supporting_quotes_tab(claim_id)

    with tab2:
        render_article_list_tab(claim_id)

    with tab3:
        render_source_comparison_tab(claim_id)


def render_supporting_quotes_tab(claim_id: str):
    """Render supporting quotes grouped by stance."""
    quotes_df = load_claim_quotes_by_stance(claim_id)

    if quotes_df.empty:
        st.info("No supporting quotes available for this claim.")
        return

    # Group by stance
    for stance_label in ['strongly_agree', 'agree', 'neutral', 'disagree', 'strongly_disagree']:
        stance_quotes = quotes_df[quotes_df['stance_label'] == stance_label]

        if not stance_quotes.empty:
            # Set color based on stance
            if 'agree' in stance_label:
                emoji = "✅"
                color = "#2D6A4F"
            elif 'disagree' in stance_label:
                emoji = "❌"
                color = "#C9184A"
            else:
                emoji = "⚖️"
                color = "#FFD93D"

            st.markdown(f"### {emoji} {stance_label.replace('_', ' ').title()} ({len(stance_quotes)} articles)")

            for _, row in stance_quotes.iterrows():
                with st.expander(f"{row['source_id']} - {row['article_title'][:60]}..."):
                    quotes = row['supporting_quotes'] if isinstance(row['supporting_quotes'], list) else []
                    if quotes:
                        for quote in quotes:
                            st.markdown(f"> {quote}")
                    else:
                        st.info("No quotes extracted for this article.")

                    st.caption(f"Published: {row['date_posted']} | Stance Score: {row['stance_score']:.2f}")


def render_article_list_tab(claim_id: str):
    """Render full article list with stance scores."""
    articles = load_claim_articles(claim_id, limit=100)

    if not articles:
        st.info("No articles found for this claim.")
        return

    st.markdown(f"**Total articles:** {len(articles)}")

    for article in articles:
        stance_color = "#2D6A4F" if article['stance_score'] > 0.2 else ("#C9184A" if article['stance_score'] < -0.2 else "#FFD93D")

        with st.expander(f"{article['source_id']} - {article['title'][:80]}..."):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Title:** {article['title']}")
                st.markdown(f"**Published:** {article['date_posted']}")
                st.markdown(f"**Excerpt:** {article['content'][:300]}...")
                st.markdown(f"[Read full article]({article['url']})")

            with col2:
                st.metric("Stance Score", f"{article['stance_score']:.2f}")
                st.markdown(f"**Label:** {article['stance_label']}")
                st.metric("Sentiment", f"{article['sentiment_score']:.2f}")

            if article['supporting_quotes']:
                quotes = article['supporting_quotes'] if isinstance(article['supporting_quotes'], list) else []
                if quotes:
                    st.markdown("**Key Quotes:**")
                    for quote in quotes[:3]:
                        st.markdown(f"> {quote}")


def render_source_comparison_tab(claim_id: str):
    """Render side-by-side source comparison."""
    comparison_df = load_claim_source_comparison(claim_id)

    if comparison_df.empty:
        st.info("No comparison data available.")
        return

    st.markdown("#### 📊 Source-by-Source Breakdown")

    # Display as table
    display_df = comparison_df[['source_id', 'avg_stance', 'stance_label', 'avg_confidence', 'article_count']].copy()
    display_df.columns = ['Source', 'Avg Stance', 'Stance Label', 'Avg Confidence', 'Articles']
    display_df['Avg Stance'] = display_df['Avg Stance'].round(2)
    display_df['Avg Confidence'] = display_df['Avg Confidence'].round(2)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Visual comparison
    fig = go.Figure()

    for _, row in comparison_df.iterrows():
        color = SOURCE_COLORS.get(row['source_id'], '#888888')
        fig.add_trace(go.Bar(
            name=row['source_id'],
            x=[row['source_id']],
            y=[row['avg_stance']],
            marker_color=color,
            text=[f"{row['avg_stance']:.2f}"],
            textposition='outside'
        ))

    fig.update_layout(
        title="Average Stance Score by Source",
        yaxis_title="Stance Score",
        yaxis=dict(range=[-1, 1]),
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Main Page
# ============================================================================

st.title("⚖️ Stance Distribution Analysis")
st.markdown("Analyze how different news sources agree/disagree with claims about Cyclone Ditwah coverage.")

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
    st.info("👆 Select or create a ditwah_claims version to view stance analysis")
    st.stop()

# Check if stance data exists
try:
    overview = load_stance_overview(version_id)
    if overview['total_claims'] == 0:
        st.warning("⚠️ No stance data found for this version. Ensure the stance analysis pipeline has been run.")
        st.stop()
except Exception as e:
    st.error(f"Error loading stance data: {str(e)}")
    st.stop()

st.markdown("---")

# SECTION 1: Overview Metrics
render_stance_overview_section(overview)

st.divider()

# SECTION 2: Polarization Dashboard
st.subheader("📊 Claim Polarization Dashboard")
st.markdown("Visualize which claims generate agreement vs. disagreement across sources")
render_polarization_dashboard(version_id)

st.divider()

# SECTION 3: Source Alignment Analysis
st.subheader("Source Alignment Matrix")
st.markdown("See which news sources tend to agree or disagree with each other")
render_source_alignment(version_id)

st.divider()

# SECTION 4: Confidence Explorer
st.subheader("🎯 Confidence-Weighted Stance Explorer")
st.markdown("Explore claims by stance position, controversy level, and confidence")
render_confidence_explorer(version_id)

st.divider()

# SECTION 5: Claim Deep Dive
st.subheader("🔍 Claim Deep Dive")
st.markdown("Select a claim to see detailed stance breakdown, quotes, and article-level analysis")
render_claim_deep_dive(version_id, selected_stance_model)
