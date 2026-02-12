"""Events Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from data.loaders import load_top_events, load_event_details
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style

apply_page_style()

st.title("Event Clustering Analysis")

# Create version button at the top
render_create_version_button('clustering')

# Version selector
version_id = render_version_selector('clustering')

if not version_id:
    st.stop()

st.markdown("---")

events = load_top_events(version_id, 30)
if not events:
    st.warning("No event clusters found for this version. Run clustering first.")
    st.code(f"python3 scripts/clustering/02_cluster_events.py --version-id {version_id}")
    st.info("Embeddings are auto-generated if needed, or run separately:\n"
            "`python3 scripts/embeddings/01_generate_embeddings.py --model <model>`")
    st.stop()

events_df = pd.DataFrame(events)

# Filter to multi-source events
multi_source_events = events_df[events_df['sources_count'] > 1]

# Event selector
event_options = {
    f"{e['cluster_name']}... ({e['article_count']} articles, {e['sources_count']} sources)": e['id']
    for _, e in multi_source_events.iterrows()
}

selected_event_label = st.selectbox(
    "Select an event to explore",
    options=list(event_options.keys())
)

if selected_event_label:
    event_id = event_options[selected_event_label]
    articles = load_event_details(event_id, version_id)

    if articles:
        articles_df = pd.DataFrame(articles)
        articles_df['source_name'] = articles_df['source_id'].map(SOURCE_NAMES)

        # Source breakdown
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Coverage by Source**")
            source_counts = articles_df['source_name'].value_counts()

            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                color=source_counts.index,
                color_discrete_map=SOURCE_COLORS
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown("**Articles in this Event**")
            display_df = articles_df[['title', 'source_name', 'date_posted']].copy()
            display_df.columns = ['Title', 'Source', 'Date']
            st.dataframe(display_df, width='stretch', height=300)
