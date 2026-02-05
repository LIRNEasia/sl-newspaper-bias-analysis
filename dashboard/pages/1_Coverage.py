"""Coverage Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.loaders import (
    load_overview_stats,
    load_coverage_timeline,
    load_ditwah_timeline,
    load_article_lengths,
    load_ditwah_article_lengths
)
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.styling import apply_page_style


def lighten_color(hex_color, factor=0.5):
    """Lighten a hex color by blending with white.

    Args:
        hex_color: Hex color string (e.g., '#FF5733')
        factor: Amount to lighten (0=original, 1=white). Default 0.5

    Returns:
        Lightened hex color string
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')

    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Blend with white (255, 255, 255)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)

    # Convert back to hex
    return f'#{r:02x}{g:02x}{b:02x}'


apply_page_style()

st.title("Coverage Analysis")

# Load stats
stats = load_overview_stats()

# Total Articles Section
st.header("Article Coverage by Source")

st.subheader("Articles by Source")

# Prepare data
source_df = pd.DataFrame(stats['by_source'])
source_df['source_name'] = source_df['source_id'].map(SOURCE_NAMES)

ditwah_df = pd.DataFrame(stats['ditwah_by_source'])

if not ditwah_df.empty:
    ditwah_df['source_name'] = ditwah_df['source_id'].map(SOURCE_NAMES)

    # Create figure with graph_objects for custom colors
    fig = go.Figure()

    # Get unique sources in order
    sources = source_df['source_id'].tolist()

    # Add bars for each source
    for source_id in sources:
        source_name = SOURCE_NAMES[source_id]
        source_color = SOURCE_COLORS[source_name]
        light_color = lighten_color(source_color, factor=0.5)

        # Total articles count
        total_count = source_df[source_df['source_id'] == source_id]['count'].values[0]

        # Ditwah count (0 if not found)
        ditwah_count = 0
        ditwah_row = ditwah_df[ditwah_df['source_id'] == source_id]
        if not ditwah_row.empty:
            ditwah_count = ditwah_row['count'].values[0]

        # Add total articles bar
        fig.add_trace(go.Bar(
            name=f'{source_name} (Total)',
            x=[source_name],
            y=[total_count],
            marker_color=source_color,
            legendgroup=source_name,
            showlegend=True,
            text=[total_count],
            textposition='auto',
        ))

        # Add Ditwah articles bar
        fig.add_trace(go.Bar(
            name=f'{source_name} (Ditwah)',
            x=[source_name],
            y=[ditwah_count],
            marker_color=light_color,
            legendgroup=source_name,
            showlegend=True,
            text=[ditwah_count],
            textposition='auto',
        ))

    fig.update_layout(
        barmode='group',
        height=450,
        xaxis_title='Source',
        yaxis_title='Number of Articles',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    # Fallback if no Ditwah data
    fig = px.bar(
        source_df,
        x='source_name',
        y='count',
        color='source_name',
        color_discrete_map=SOURCE_COLORS,
        labels={'count': 'Articles', 'source_name': 'Source'}
    )
    fig.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.info("No Ditwah cyclone articles found.")

st.divider()

# Timeline Section
st.header("Coverage Over Time")

st.subheader("Articles Timeline")
timeline_data = load_coverage_timeline()

if timeline_data:
    timeline_df = pd.DataFrame(timeline_data)
    timeline_df['source_name'] = timeline_df['source_id'].map(SOURCE_NAMES)

    fig = px.line(
        timeline_df,
        x='date',
        y='count',
        color='source_name',
        color_discrete_map=SOURCE_COLORS,
        labels={'count': 'Articles', 'date': 'Date', 'source_name': 'Source'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Ditwah Cyclone Timeline")
ditwah_timeline = load_ditwah_timeline()

if ditwah_timeline:
    ditwah_timeline_df = pd.DataFrame(ditwah_timeline)
    ditwah_timeline_df['source_name'] = ditwah_timeline_df['source_id'].map(SOURCE_NAMES)

    fig = px.line(
        ditwah_timeline_df,
        x='date',
        y='count',
        color='source_name',
        color_discrete_map=SOURCE_COLORS,
        labels={'count': 'Articles', 'date': 'Date', 'source_name': 'Source'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No Ditwah cyclone timeline data available.")

st.divider()

# Article Length Distribution Section
st.header("Article Length Distribution")

col1, col2 = st.columns(2)

with col1:
    st.subheader("All Articles Length Distribution")
    lengths_data = load_article_lengths()

    if lengths_data:
        lengths_df = pd.DataFrame(lengths_data)

        fig = px.histogram(
            lengths_df,
            x='article_length',
            labels={'article_length': 'Article Length (characters)'},
            nbins=50,
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Show statistics
        st.caption(f"**Mean length:** {lengths_df['article_length'].mean():.0f} characters")
        st.caption(f"**Median length:** {lengths_df['article_length'].median():.0f} characters")

with col2:
    st.subheader("Ditwah Articles Length Distribution")
    ditwah_lengths = load_ditwah_article_lengths()

    if ditwah_lengths:
        ditwah_lengths_df = pd.DataFrame(ditwah_lengths)

        fig = px.histogram(
            ditwah_lengths_df,
            x='article_length',
            labels={'article_length': 'Article Length (characters)'},
            nbins=50,
            color_discrete_sequence=['#EF553B']
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Show statistics
        st.caption(f"**Mean length:** {ditwah_lengths_df['article_length'].mean():.0f} characters")
        st.caption(f"**Median length:** {ditwah_lengths_df['article_length'].median():.0f} characters")
    else:
        st.info("No Ditwah cyclone articles available for length analysis.")
