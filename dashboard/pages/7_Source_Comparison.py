"""Source Comparison Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from components.styling import apply_page_style

# Page config
st.set_page_config(
    page_title="Source Comparison - Sri Lanka Media Bias Detector",
    page_icon="⚖️",
    layout="wide"
)

apply_page_style()

st.title("⚖️ Source Comparison")
st.markdown("Compare how different news sources cover the same events and topics")

st.info("Source comparison features are integrated into other tabs:")
st.markdown("""
- **Topics Tab**: View topic coverage distribution and selection bias across sources
- **Events Tab**: Compare multi-source coverage of the same events
- **Word Frequency Tab**: Compare distinctive vocabulary across sources
- **Sentiment Tab**: Compare sentiment patterns across sources
- **Ditwah Claims Tab**: Compare how sources cover specific claims about Cyclone Ditwah
- **Stance Tab**: Analyze stance alignment and disagreement patterns across sources
""")
