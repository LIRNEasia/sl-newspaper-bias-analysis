"""Coverage Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from data.loaders import load_overview_stats, load_coverage_timeline
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.styling import apply_page_style

apply_page_style()

st.title("Coverage Analysis")

# Load stats
stats = load_overview_stats()

st.subheader("Article Coverage by Source")

# Articles by source bar chart
source_df = pd.DataFrame(stats['by_source'])
source_df['source_name'] = source_df['source_id'].map(SOURCE_NAMES)

fig = px.bar(
    source_df,
    x='source_name',
    y='count',
    color='source_name',
    color_discrete_map=SOURCE_COLORS,
    labels={'count': 'Articles', 'source_name': 'Source'}
)
fig.update_layout(showlegend=False, height=400)
st.plotly_chart(fig, width='stretch')

# Timeline
st.subheader("Coverage Over Time")
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
    st.plotly_chart(fig, width='stretch')
