"""Article Summaries Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.versions import get_version
from data.loaders import (
    load_summaries,
    load_summary_statistics,
    load_summaries_by_source
)
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style

apply_page_style()

st.title("Article Summaries")

# Create version button at the top
render_create_version_button('summarization')

# Version selector
version_id = render_version_selector('summarization')

if not version_id:
    st.info("Select or create a summarization version to view summaries")
    st.stop()

# Get version details
version = get_version(version_id)
if not version:
    st.error("Version not found")
    st.stop()

config = version['configuration']
summ_config = config.get('summarization', {})

# Load summary statistics
stats = load_summary_statistics(version_id)

if not stats or not stats.get('overall'):
    st.warning("No summaries found for this version. Run the pipeline first:")
    st.code(f"python3 scripts/summarization/01_generate_summaries.py --version-id {version_id}")
    st.stop()

overall = stats['overall']
by_source = stats['by_source']

# Overall statistics
st.subheader("Overall Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Summaries",
        f"{overall['total_summaries']:,}"
    )

with col2:
    compression = overall['avg_compression'] * 100 if overall['avg_compression'] else 0
    st.metric(
        "Avg Compression",
        f"{compression:.1f}%",
        help="Percentage of original article length retained in summary"
    )

with col3:
    st.metric(
        "Avg Word Count",
        f"{int(overall['avg_word_count'])}" if overall['avg_word_count'] else "0"
    )

with col4:
    st.metric(
        "Avg Processing Time",
        f"{int(overall['avg_time_ms'])}ms" if overall['avg_time_ms'] else "0ms"
    )


# Article browser with summaries
st.subheader("Article Summaries")

# Filters
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    source_filter = st.selectbox(
        "Filter by Source",
        options=["All"] + [SOURCE_NAMES[s['source_id']] for s in by_source],
        key="source_filter"
    )

with col2:
    search_query = st.text_input(
        "Search by title",
        key="search_query",
        placeholder="Enter keywords..."
    )

with col3:
    limit = st.number_input(
        "Articles to show",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        key="limit"
    )

# Load summaries
source_id = None
if source_filter != "All":
    # Find source_id from name
    for s in by_source:
        if SOURCE_NAMES[s['source_id']] == source_filter:
            source_id = s['source_id']
            break

summaries = load_summaries(version_id, source_id=source_id, limit=limit)

# Apply search filter
if search_query:
    summaries = [
        s for s in summaries
        if search_query.lower() in (s['title'] or '').lower()
    ]

if not summaries:
    st.info("No summaries found matching the filters")
    st.stop()

st.write(f"Showing {len(summaries)} article(s)")

# Display summaries
for summary in summaries:
    with st.expander(
        f"**{summary['title']}** - {SOURCE_NAMES.get(summary['source_id'], summary['source_id'])} | {summary['date_posted']}",
        expanded=False
    ):
        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            original_words = len((summary['content'] or '').split())
            st.metric("Original", f"{original_words} words")

        with col2:
            st.metric("Summary", f"{summary['word_count']} words")

        with col3:
            compression = summary['compression_ratio'] * 100 if summary['compression_ratio'] else 0
            st.metric("Compression", f"{compression:.1f}%")

        with col4:
            st.metric("Time", f"{summary['processing_time_ms']}ms")

        # Summary text
        st.markdown("**Summary:**")
        st.write(summary['summary_text'])

        # Link
        if summary['url']:
            st.markdown(f"[View original article]({summary['url']})")

        st.divider()
