"""Semantic Search - Find articles by meaning using BGE embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from data.loaders import semantic_search_articles
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.styling import apply_page_style

# BGE-base requires this prefix for retrieval queries (documents were embedded without it)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

apply_page_style()

st.title("Semantic Search")
st.caption("Find articles by meaning using BGE embeddings and pgvector similarity search")

# --- Embedding Client (cached in session_state to avoid reloading ~420MB model) ---
if "search_embed_client" not in st.session_state:
    with st.spinner("Loading embedding model (first time only)..."):
        from src.llm import EmbeddingClient
        st.session_state.search_embed_client = EmbeddingClient(
            provider="local",
            model=BGE_MODEL_NAME
        )

embed_client = st.session_state.search_embed_client

# --- Search Controls ---
query = st.text_input(
    "Search query",
    placeholder="e.g., government response to cyclone damage in southern Sri Lanka",
)

col1, col2 = st.columns([1, 1])

with col1:
    selected_sources = st.multiselect(
        "Filter by source",
        options=list(SOURCE_NAMES.keys()),
        format_func=lambda x: SOURCE_NAMES.get(x, x),
        default=[],
    )

with col2:
    num_results = st.slider("Number of results", min_value=5, max_value=50, value=20, step=5)

# --- Search Execution ---
if not query or len(query.strip()) < 3:
    st.info("Enter a search query above to find semantically similar articles.")
    st.stop()

with st.spinner("Searching..."):
    prefixed_query = BGE_QUERY_PREFIX + query
    query_embedding = embed_client.embed_single(prefixed_query)

    results = semantic_search_articles(
        query_embedding=query_embedding,
        embedding_model=BGE_MODEL_NAME,
        limit=num_results,
        source_ids=selected_sources if selected_sources else None,
    )

# --- Display Results ---
if not results:
    st.warning("No results found. Try a different query or remove source filters.")
    st.stop()

st.divider()
st.subheader(f"{len(results)} results")

for i, result in enumerate(results):
    source_name = SOURCE_NAMES.get(result["source_id"], result["source_id"])
    source_color = SOURCE_COLORS.get(source_name, "#999")
    date_str = result["date_posted"].strftime("%Y-%m-%d") if result["date_posted"] else "Unknown"
    similarity = float(result["similarity_score"])

    with st.expander(
        f"**{i + 1}. {result['title']}** — {source_name} | {date_str} | similarity: {similarity:.3f}",
        expanded=(i < 3),
    ):
        meta1, meta2, meta3 = st.columns(3)
        with meta1:
            st.markdown(f"**Source:** <span style='color:{source_color}'>{source_name}</span>",
                        unsafe_allow_html=True)
        with meta2:
            st.markdown(f"**Date:** {date_str}")
        with meta3:
            st.metric("Similarity", f"{similarity:.3f}")

        content = result.get("content", "") or ""
        if content:
            snippet = content[:500] + ("..." if len(content) > 500 else "")
            st.markdown("**Content preview:**")
            st.text(snippet)

        if result.get("url"):
            st.markdown(f"[View original article]({result['url']})")
