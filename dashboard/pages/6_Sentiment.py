"""Sentiment Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.loaders import (
    load_available_models,
    load_topic_list,
    load_sentiment_by_source_topic,
    load_sentiment_percentage_by_source_topic,
    load_sentiment_distribution,
    load_sentiment_timeline,
    load_topic_sentiment,
    load_multi_model_comparison,
)
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.charts import (
    render_multi_model_stacked_bars,
    render_source_model_comparison,
    render_model_agreement_heatmap,
)
from components.styling import apply_page_style

apply_page_style()

st.title("Sentiment Analysis - Multi-Model Comparison")

# Load available data
available_models = load_available_models()
topics = load_topic_list()

if not available_models:
    st.warning("No sentiment analysis data found. Run `python scripts/04_analyze_sentiment.py` first.")
    st.stop()

# Topic selector in main area
topic_options = ["All Topics"] + [t['name'] for t in topics]
selected_topic = st.selectbox(
    "Filter by Topic",
    options=topic_options,
    help="Filter all visualizations by topic"
)

# Show available models in an expander
model_list = [m['model_type'] for m in available_models]
with st.expander("Available Models"):
    for m in available_models:
        st.caption(f"[OK] {m['model_type']}: {m['article_count']:,} articles")

# View mode selector
view_mode = st.radio(
    "View Mode",
    options=["Single Model View", "Model Comparison View"],
    horizontal=True
)

st.divider()

# Model display names
MODEL_DISPLAY_NAMES = {
    'roberta': 'RoBERTa',
    'distilbert': 'DistilBERT',
    'finbert': 'FinBERT',
    'vader': 'VADER',
    'textblob': 'TextBlob',
    'local': 'Local (RoBERTa)'
}

# Model colors
MODEL_COLORS = {
    "roberta": "#1f77b4",
    "distilbert": "#ff7f0e",
    "finbert": "#2ca02c",
    "vader": "#d62728",
    "textblob": "#9467bd",
    "local": "#1f77b4"
}

if view_mode == "Single Model View":
    # Model selector
    selected_model = st.selectbox(
        "Select Model",
        options=model_list,
        format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x.upper())
    )

    st.markdown(f"### {MODEL_DISPLAY_NAMES.get(selected_model, selected_model.upper())} Analysis")
    if selected_topic != "All Topics":
        st.caption(f"Filtered by topic: **{selected_topic}**")

    # Load data with topic filter
    sentiment_data = load_sentiment_by_source_topic(selected_model, selected_topic)

    if not sentiment_data:
        st.warning(f"No data for {selected_model} with topic '{selected_topic}'")
        st.stop()

    # 1. Sentiment Distribution by Source (Stacked Bar Chart)
    st.markdown("#### Sentiment Distribution by Source")
    st.caption("Percentage of articles in each sentiment category")

    pct_data = load_sentiment_percentage_by_source_topic(selected_model, selected_topic)
    if pct_data:
        pct_df = pd.DataFrame(pct_data)
        pct_df['source_name'] = pct_df['source_id'].map(SOURCE_NAMES)

        # Calculate percentages
        pct_df['negative_pct'] = (pct_df['negative_count'] / pct_df['total_count'] * 100)
        pct_df['neutral_pct'] = (pct_df['neutral_count'] / pct_df['total_count'] * 100)
        pct_df['positive_pct'] = (pct_df['positive_count'] / pct_df['total_count'] * 100)

        # Create stacked bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Negative (< -0.5)',
            x=pct_df['source_name'],
            y=pct_df['negative_pct'],
            marker_color='#d62728',
            text=pct_df['negative_pct'].round(1).astype(str) + '%',
            textposition='inside',
            hovertemplate='%{x}<br>Negative: %{y:.1f}%<br>Count: ' + pct_df['negative_count'].astype(str) + '<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Neutral (-0.5 to 0.5)',
            x=pct_df['source_name'],
            y=pct_df['neutral_pct'],
            marker_color='#7f7f7f',
            text=pct_df['neutral_pct'].round(1).astype(str) + '%',
            textposition='inside',
            hovertemplate='%{x}<br>Neutral: %{y:.1f}%<br>Count: ' + pct_df['neutral_count'].astype(str) + '<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Positive (> 0.5)',
            x=pct_df['source_name'],
            y=pct_df['positive_pct'],
            marker_color='#2ca02c',
            text=pct_df['positive_pct'].round(1).astype(str) + '%',
            textposition='inside',
            hovertemplate='%{x}<br>Positive: %{y:.1f}%<br>Count: ' + pct_df['positive_count'].astype(str) + '<extra></extra>'
        ))

        fig.update_layout(
            barmode='stack',
            yaxis_title="Percentage (%)",
            xaxis_title="News Source",
            height=400,
            yaxis_range=[0, 100],
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, width='stretch')

    # 2. Average Sentiment by Source
    st.markdown("#### Average Sentiment by Source")
    source_df = pd.DataFrame(sentiment_data)
    source_df['source_name'] = source_df['source_id'].map(SOURCE_NAMES)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=source_df['source_name'],
        y=source_df['avg_sentiment'],
        error_y=dict(type='data', array=source_df['stddev_sentiment']),
        marker_color=[SOURCE_COLORS.get(name, '#999') for name in source_df['source_name']],
        text=source_df['avg_sentiment'].round(2),
        textposition='outside'
    ))
    fig.update_layout(
        yaxis_title="Average Sentiment Score (-5 to +5)",
        xaxis_title="News Source",
        height=400,
        yaxis_range=[-5, 5],
        hovermode='x unified'
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
    st.plotly_chart(fig, width='stretch')

    # 3. Sentiment Distribution (Box Plot)
    st.markdown("#### Sentiment Distribution")
    dist_data = load_sentiment_distribution(selected_model)
    if dist_data:
        dist_df = pd.DataFrame(dist_data)
        dist_df['source_name'] = dist_df['source_id'].map(SOURCE_NAMES)

        fig = go.Figure()
        for source in dist_df['source_name'].unique():
            source_data = dist_df[dist_df['source_name'] == source]
            fig.add_trace(go.Box(
                y=source_data['overall_sentiment'],
                name=source,
                marker_color=SOURCE_COLORS.get(source, '#999')
            ))
        fig.update_layout(
            yaxis_title="Sentiment Score (-5 to +5)",
            height=400,
            yaxis_range=[-5, 5],
            showlegend=True
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width='stretch')

    # 4. Sentiment Timeline
    st.markdown("#### Sentiment Over Time")
    timeline_data = load_sentiment_timeline(selected_model)
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        timeline_df['source_name'] = timeline_df['source_id'].map(SOURCE_NAMES)

        fig = px.line(
            timeline_df,
            x='date',
            y='avg_sentiment',
            color='source_name',
            color_discrete_map=SOURCE_COLORS,
            labels={'avg_sentiment': 'Avg Sentiment', 'date': 'Date', 'source_name': 'Source'}
        )
        fig.update_layout(height=400, yaxis_range=[-5, 5])
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width='stretch')

    # 5. Topic-Sentiment Heatmap
    st.markdown("#### Topic Sentiment by Source")
    topic_sentiment = load_topic_sentiment(selected_model)
    if topic_sentiment:
        ts_df = pd.DataFrame(topic_sentiment)
        ts_df['source_name'] = ts_df['source_id'].map(SOURCE_NAMES)

        # Pivot for heatmap
        pivot = ts_df.pivot_table(
            values='avg_sentiment',
            index='topic',
            columns='source_name',
            aggfunc='mean'
        )

        # Only show top 15 topics by total article count
        topic_counts = ts_df.groupby('topic')['article_count'].sum().sort_values(ascending=False)
        top_topics = topic_counts.head(15).index
        pivot = pivot.loc[pivot.index.isin(top_topics)]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            zmin=-3,
            zmax=3,
            colorbar=dict(title="Sentiment")
        ))
        fig.update_layout(
            height=600,
            xaxis_title="News Source",
            yaxis_title="Topic"
        )
        st.plotly_chart(fig, width='stretch')

else:
    # Model Comparison View
    st.markdown("### Model Comparison")
    if selected_topic != "All Topics":
        st.caption(f"Filtered by topic: **{selected_topic}**")

    # Load multi-model data
    comparison_data = load_multi_model_comparison(model_list, selected_topic)

    if not comparison_data:
        st.warning("No comparison data available")
        st.stop()

    df = pd.DataFrame(comparison_data)
    df['source_name'] = df['source_id'].map(SOURCE_NAMES)

    # 1. Multi-model stacked bar (grouped by source)
    st.markdown("#### Sentiment Distribution by Source & Model")
    st.caption("Percentage of negative/neutral/positive articles for each model, grouped by source")
    render_multi_model_stacked_bars(df, MODEL_COLORS)

    # 2. Average sentiment comparison (grouped bar chart)
    st.markdown("#### Average Sentiment: Source x Model Comparison")
    render_source_model_comparison(df, MODEL_COLORS)

    # 3. Model agreement analysis
    st.markdown("#### Model Agreement Matrix")
    st.caption("Correlation between models - higher values indicate models agree more")
    render_model_agreement_heatmap(df)
