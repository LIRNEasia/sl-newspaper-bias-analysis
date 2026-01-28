"""Word Frequency Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.versions import get_version
from data.loaders import load_word_frequencies
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style

apply_page_style()

st.title("Word Frequency Analysis")

# Version selector
version_id = render_version_selector('word_frequency')

# Create version button
render_create_version_button('word_frequency')

if not version_id:
    st.info("Select or create a word frequency version to view analysis")
    st.stop()

# Get version details
version = get_version(version_id)
if not version:
    st.error("Version not found")
    st.stop()

config = version['configuration']
wf_config = config.get('word_frequency', {})

# Load word frequencies
word_freqs = load_word_frequencies(version_id)

if not word_freqs:
    st.warning("No word frequencies found for this version. Run the pipeline first:")
    st.code(f"python3 scripts/word_frequency/01_compute_word_frequency.py --version-id {version_id}")
    st.stop()

# Get configuration for display
ranking_method = wf_config.get('ranking_method', 'frequency')

# Top words per source
st.subheader("Top Words by Source")

# Create 2x2 grid for sources
sources = list(word_freqs.keys())
num_sources = len(sources)

if num_sources == 0:
    st.warning("No sources found")
    st.stop()

# Create columns based on number of sources
if num_sources <= 2:
    cols = st.columns(num_sources)
else:
    # First row
    cols1 = st.columns(2)
    # Second row if needed
    if num_sources > 2:
        cols2 = st.columns(min(2, num_sources - 2))
        cols = list(cols1) + list(cols2)
    else:
        cols = cols1

for idx, (source_id, words) in enumerate(word_freqs.items()):
    if idx >= len(cols):
        break

    source_name = SOURCE_NAMES.get(source_id, source_id)

    with cols[idx]:
        st.markdown(f"**{source_name}**")

        # Prepare data
        df = pd.DataFrame(words)

        # Determine value column
        if ranking_method == 'frequency':
            value_col = 'frequency'
            label = 'Frequency'
        else:
            value_col = 'tfidf_score'
            label = 'TF-IDF Score'

        # Bar chart (top 20)
        fig = px.bar(
            df.head(20),
            x=value_col,
            y='word',
            orientation='h',
            labels={value_col: label, 'word': 'Word'},
            color_discrete_sequence=[SOURCE_COLORS.get(source_name, '#1f77b4')]
        )
        fig.update_layout(
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')
