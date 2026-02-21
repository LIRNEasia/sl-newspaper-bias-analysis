"""Entity Stance Detection Dashboard Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json as _json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.versions import get_version, list_versions
from data.loaders import (
    load_entity_stance_summary,
    load_entity_stance_summary_by_topic,
    load_entity_stance_examples,
    load_polarizing_entities,
    load_entity_stance_detail,
    load_entity_stance_overview,
    load_topics,
)
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style

apply_page_style()

st.title("Entity Stance Detection")
st.caption("How do different outlets portray the same entities? "
           "NLI-based stance scoring reveals positive/negative framing per entity per source.")

# Version selector
version_id = render_version_selector('entity_stance')

# Create version button
render_create_version_button('entity_stance')

if not version_id:
    st.info("Select or create an entity_stance version to view analysis")
    st.stop()

# Get version details
version = get_version(version_id)
if not version:
    st.error("Version not found")
    st.stop()

# Check if pipeline is complete
if not version.get('is_complete'):
    st.warning("Pipeline incomplete. Run the entity stance script:")
    st.code(f"python3 scripts/entity_stance/01_analyze_entity_stance.py --version-id {version_id}")
    st.stop()

st.divider()

# --- Section 1: Overview metrics ---
overview = load_entity_stance_overview(version_id)

if not overview or overview.get("total_stances", 0) == 0:
    st.info("No entity stances found. Run the pipeline first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Non-neutral Stances", f"{overview['total_stances']:,}")
with col2:
    st.metric("Unique Entities", f"{overview['unique_entities']:,}")
with col3:
    st.metric("Articles Processed", f"{overview['articles_processed']:,}")
with col4:
    avg_abs = overview.get('avg_abs_stance') or 0
    st.metric("Avg |Stance|", f"{avg_abs:.3f}")

st.divider()

# --- Topic filter (optional) ---
topic_versions = list_versions('topics')
topic_version_id_for_filter = None
selected_topic_bertopic_id = None

if topic_versions:
    col_tv, col_tf = st.columns(2)
    with col_tv:
        tv_options = {v['name']: v['id'] for v in topic_versions}
        tv_label = st.selectbox(
            "Topic version (optional)",
            ["(none)"] + list(tv_options.keys()),
            key="entity_topic_version",
            help="Select a topic version to enable topic filtering"
        )
        if tv_label != "(none)":
            topic_version_id_for_filter = tv_options[tv_label]
    with col_tf:
        if topic_version_id_for_filter:
            topics_list = load_topics(topic_version_id_for_filter)
            topic_options = {"All topics": None}
            for t in topics_list:
                display = t['name']
                try:
                    if t.get('description'):
                        d = _json.loads(t['description'])
                        if d.get('aspect'):
                            display = d['aspect']
                except (ValueError, TypeError):
                    pass
                topic_options[f"{display} ({t['article_count']} articles)"] = t['topic_id']
            selected_topic_label = st.selectbox(
                "Filter by topic", list(topic_options.keys()), key="entity_topic_filter"
            )
            selected_topic_bertopic_id = topic_options[selected_topic_label]

# Load entity summary (topic-filtered or full)
if topic_version_id_for_filter:
    df_deep_dive = load_entity_stance_summary_by_topic(
        version_id, topic_version_id_for_filter, selected_topic_bertopic_id
    )
    if not df_deep_dive.empty:
        df_deep_dive["source_name"] = (
            df_deep_dive["source_id"].map(SOURCE_NAMES).fillna(df_deep_dive["source_id"])
        )
else:
    df_summary = load_entity_stance_summary(version_id)
    if not df_summary.empty:
        df_summary["source_name"] = (
            df_summary["source_id"].map(SOURCE_NAMES).fillna(df_summary["source_id"])
        )
    df_deep_dive = df_summary.copy() if not df_summary.empty else pd.DataFrame()

# Multi-select entity
if not df_deep_dive.empty:
    entity_options = sorted(df_deep_dive["entity_text"].unique().tolist())

    # Preserve selection across topic version / topic filter changes.
    # Only initialise on the very first render; afterwards keep whatever the
    # user had picked, filtering out entities absent from the new option list.
    if "entity_multiselect" not in st.session_state:
        st.session_state["entity_multiselect"] = entity_options[:3] if entity_options else []
    else:
        st.session_state["entity_multiselect"] = [
            e for e in st.session_state["entity_multiselect"] if e in entity_options
        ]

    selected_entities = st.multiselect(
        "Select entities", entity_options, key="entity_multiselect"
    )

    if selected_entities:
        df_entity = df_deep_dive[df_deep_dive["entity_text"].isin(selected_entities)].copy()

        if not df_entity.empty:
            # Aggregate across all selected entities per source
            df_agg = (
                df_entity
                .groupby("source_name", as_index=False)
                .apply(lambda g: pd.Series({
                    "avg_stance": (g["avg_stance"] * g["stance_count"]).sum() / g["stance_count"].sum(),
                    "stance_count": g["stance_count"].sum(),
                    "avg_confidence": (g["avg_confidence"] * g["stance_count"]).sum() / g["stance_count"].sum(),
                }), include_groups=False)
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.bar(
                    df_agg,
                    x="source_name",
                    y="avg_stance",
                    color="source_name",
                    color_discrete_map=SOURCE_COLORS,
                    labels={"avg_stance": "Average Stance", "source_name": "Source"},
                    text="stance_count",
                )
                fig.update_layout(yaxis=dict(range=[-1, 1]), showlegend=False)
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_traces(
                    texttemplate="%{text} mentions", textposition="outside"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                for _, row in df_agg.iterrows():
                    score = row["avg_stance"]
                    icon = "+" if score > 0 else ""
                    st.markdown(
                        f"**{row['source_name']}**: {icon}{score:.3f} "
                        f"({int(row['stance_count'])} mentions, "
                        f"conf: {row['avg_confidence']:.2f})"
                    )

        # --- Example passages tabs ---
        st.markdown("**Example passages by outlet**")

        sort_order = st.radio(
            "Sort examples by",
            ["Most extreme (±)", "Most positive", "Most negative"],
            horizontal=True,
            key="examples_sort_order",
        )

        df_examples = load_entity_stance_examples(
            version_id,
            tuple(selected_entities),
            topic_version_id=topic_version_id_for_filter,
            topic_bertopic_id=selected_topic_bertopic_id,
        )

        # Build ordered list of (source_id, source_name) from SOURCE_NAMES
        all_sources = [(sid, name) for sid, name in SOURCE_NAMES.items()]
        tab_labels = [name for _, name in all_sources]
        tabs = st.tabs(tab_labels)

        for tab, (src_id, src_name) in zip(tabs, all_sources):
            with tab:
                if df_examples.empty:
                    st.info("No examples found.")
                    continue

                src_df = df_examples[df_examples["source_id"] == src_id]
                if src_df.empty:
                    st.info(f"No examples for {src_name}.")
                    continue

                # Re-sort src_df based on user selection before picking top 5 chunks
                if sort_order == "Most positive":
                    src_df = src_df.sort_values("stance_score", ascending=False)
                elif sort_order == "Most negative":
                    src_df = src_df.sort_values("stance_score", ascending=True)
                else:  # Most extreme (±) — current default
                    src_df = src_df.reindex(src_df["stance_score"].abs().sort_values(ascending=False).index)

                # Group by (article_id, chunk_index), keep top 5 chunks
                chunk_groups = {}
                for _, row in src_df.iterrows():
                    key = (str(row["article_id"]), int(row["chunk_index"]))
                    if key not in chunk_groups:
                        if len(chunk_groups) >= 5:
                            break
                        chunk_groups[key] = {
                            "title": row["title"],
                            "chunk_text": row["chunk_text"],
                            "entities": [],
                        }
                    chunk_groups[key]["entities"].append(row)

                for (article_id, chunk_idx), chunk in chunk_groups.items():
                    st.markdown(f"**{chunk['title']}**")
                    st.markdown(
                        f"<blockquote style='border-left: 3px solid #ccc; "
                        f"padding-left: 1em; color: #555;'>"
                        f"{chunk['chunk_text']}</blockquote>",
                        unsafe_allow_html=True,
                    )
                    for row in chunk["entities"]:
                        score = row["stance_score"]
                        color = "green" if score > 0 else "red"
                        icon = "+" if score > 0 else ""
                        st.markdown(
                            f"&nbsp;&nbsp;:{color}[**{row['entity_text']}**] "
                            f"{icon}{score:.3f} ({row['stance_label']}, "
                            f"conf: {row['confidence']:.2f})"
                        )
                    st.markdown("---")
