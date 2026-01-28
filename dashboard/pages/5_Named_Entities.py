"""Named Entity Recognition Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import html
import streamlit as st
import pandas as pd
import plotly.express as px

from src.db import get_db
from src.versions import get_version
from data.loaders import load_entity_statistics
from components.source_mapping import SOURCE_NAMES
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style

apply_page_style()

st.title("Named Entity Recognition")

# Version selector
version_id = render_version_selector('ner')

# Create version button
render_create_version_button('ner')

if not version_id:
    st.info("Select or create an NER version to view analysis")
    st.stop()

# Get version details
version = get_version(version_id)
if not version:
    st.error("Version not found")
    st.stop()

# Check if pipeline is complete
if not version.get('is_complete'):
    st.warning("Pipeline incomplete. Run the extraction script:")
    st.code(f"python3 scripts/ner/01_extract_entities.py --version-id {version_id}")
    st.stop()

st.divider()

# Load entity type distribution
with get_db() as db:
    schema = db.config["schema"]
    with db.cursor() as cur:
        cur.execute(f"""
            SELECT entity_type, COUNT(*) as count
            FROM {schema}.named_entities
            WHERE result_version_id = %s
            GROUP BY entity_type
            ORDER BY count DESC
        """, (version_id,))
        entity_type_stats = cur.fetchall()

if not entity_type_stats:
    st.info("No entities found. Run the extraction pipeline.")
    st.stop()

# Entity type distribution chart
st.subheader("Entity Distribution by Type")
df_types = pd.DataFrame(entity_type_stats)
fig = px.bar(
    df_types,
    x='entity_type',
    y='count',
    labels={'entity_type': 'Entity Type', 'count': 'Count'},
    color='entity_type'
)
fig.update_layout(showlegend=False, height=400)
st.plotly_chart(fig, width='stretch')

# Filter by entity type
st.subheader("Top Entities by Source")

entity_type_filter = st.selectbox(
    "Filter by Entity Type",
    options=["All"] + [row['entity_type'] for row in entity_type_stats],
    key="ner_entity_type_filter"
)

# Load entity statistics
entity_filter = None if entity_type_filter == "All" else entity_type_filter
entity_stats = load_entity_statistics(version_id, entity_type=entity_filter, limit=100)

if not entity_stats:
    st.info(f"No entities found for type: {entity_type_filter}")
    st.stop()

# Create dataframe
df_entities = pd.DataFrame(entity_stats)
df_entities['source_name'] = df_entities['source_id'].map(SOURCE_NAMES)

# Pivot for heatmap
pivot = df_entities.pivot_table(
    index='entity_text',
    columns='source_name',
    values='mention_count',
    fill_value=0
)

# Show top 20 entities
top_entities = pivot.sum(axis=1).sort_values(ascending=False).head(20)
pivot_top = pivot.loc[top_entities.index]

# Heatmap
fig = px.imshow(
    pivot_top,
    labels=dict(x="Source", y="Entity", color="Mentions"),
    title=f"Top 20 {entity_type_filter} Entities by Source",
    aspect="auto",
    color_continuous_scale="Blues"
)
fig.update_layout(height=600)
st.plotly_chart(fig, width='stretch')

# Show detailed table
st.subheader("Detailed Entity Statistics")

# Format dataframe for display
display_df = df_entities[['entity_text', 'entity_type', 'source_name', 'mention_count', 'article_count']].copy()
display_df.columns = ['Entity', 'Type', 'Source', 'Mentions', 'Articles']
display_df = display_df.sort_values('Mentions', ascending=False)

st.dataframe(
    display_df.head(100),
    width='stretch',
    hide_index=True
)

# Article Entity Viewer Section
st.divider()
st.subheader("Article Entity Viewer")
st.markdown("View named entities in context for any article from the corpus")


def render_article_with_entities(content: str, entities: list) -> str:
    """Generate HTML with inline entity highlighting."""
    # Entity type color mapping
    entity_colors = {
        'PERSON': '#E3F2FD',
        'ORG': '#F3E5F5',
        'ORGANIZATION': '#F3E5F5',
        'LOC': '#E8F5E9',
        'LOCATION': '#E8F5E9',
        'GPE': '#C8E6C9',
        'DATE': '#FFF3E0',
        'TIME': '#FFE0B2',
        'EVENT': '#FCE4EC',
        'FAC': '#E1BEE7',
        'PRODUCT': '#FFECB3',
        'PERCENT': '#B2DFDB',
        'NORP': '#D1C4E9',
        'MONEY': '#C5E1A5',
        'LAW': '#FFCCBC',
    }
    default_color = '#F5F5F5'

    if not entities:
        return f'<div style="line-height: 1.8; font-size: 16px; white-space: pre-wrap;">{html.escape(content)}</div>'

    html_parts = []
    last_end = 0
    highlighted_ranges = []

    for entity in entities:
        start = entity['start_char']
        end = entity['end_char']

        is_overlap = any(
            (start < prev_end and end > prev_start)
            for prev_start, prev_end in highlighted_ranges
        )
        if is_overlap:
            continue

        if start > last_end:
            html_parts.append(html.escape(content[last_end:start]))

        entity_text = content[start:end]
        entity_type = entity['entity_type']
        confidence = entity.get('confidence', 0.0)
        color = entity_colors.get(entity_type, default_color)

        entity_html = (
            f'<span style="background-color: {color}; padding: 2px 4px; '
            f'border-radius: 3px; cursor: help; border: 1px solid #ccc;" '
            f'title="{entity_type} (confidence: {confidence:.2f})">'
            f'{html.escape(entity_text)}'
            f'</span>'
        )
        html_parts.append(entity_html)

        highlighted_ranges.append((start, end))
        last_end = end

    if last_end < len(content):
        html_parts.append(html.escape(content[last_end:]))

    html_content = ''.join(html_parts)
    return f'<div style="line-height: 1.8; font-size: 16px; white-space: pre-wrap;">{html_content}</div>'


# URL input
article_url = st.text_input(
    "Enter Article URL",
    placeholder="Enter article URL from the corpus",
    key="ner_article_url_input"
)

if article_url:
    with get_db() as db:
        # Fetch article by URL
        article = db.get_article_by_url(article_url)

        if not article:
            st.warning("Article not found. Please ensure the URL is exactly as stored in the database.")
        else:
            # Fetch entities for this article
            entities = db.get_entities_for_article(article['id'], version_id)

            # Display article metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Title:** {article['title']}")
            with col2:
                source_name = SOURCE_NAMES.get(article['source_id'], article['source_id'])
                st.markdown(f"**Source:** {source_name}")
            with col3:
                st.markdown(f"**Date:** {article['date_posted']}")

            if not entities:
                st.info("No entities were extracted from this article.")
            else:
                # NER extraction uses title + "\n\n" + content
                full_text = f"{article['title']}\n\n{article['content']}"

                # Entity summary
                entity_type_counts = {}
                for entity in entities:
                    entity_type = entity['entity_type']
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

                entity_summary = ", ".join([f"{count} {etype}" for etype, count in entity_type_counts.items()])
                st.markdown(f"**Found {len(entities)} entities:** {entity_summary}")

                # Entity legend
                st.markdown("**Entity Type Legend:**")
                legend_html = """
                <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; font-size: 14px;">
                    <span style="background-color: #E3F2FD; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">PERSON</span>
                    <span style="background-color: #F3E5F5; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">ORG</span>
                    <span style="background-color: #E8F5E9; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">LOC</span>
                    <span style="background-color: #C8E6C9; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">GPE</span>
                    <span style="background-color: #FFF3E0; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">DATE</span>
                    <span style="background-color: #FFE0B2; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">TIME</span>
                    <span style="background-color: #FCE4EC; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">EVENT</span>
                    <span style="background-color: #E1BEE7; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">FAC</span>
                    <span style="background-color: #FFECB3; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">PRODUCT</span>
                    <span style="background-color: #B2DFDB; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">PERCENT</span>
                    <span style="background-color: #D1C4E9; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">NORP</span>
                    <span style="background-color: #C5E1A5; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">MONEY</span>
                    <span style="background-color: #FFCCBC; padding: 4px 8px; border-radius: 3px; border: 1px solid #ccc;">LAW</span>
                </div>
                """
                st.markdown(legend_html, unsafe_allow_html=True)

                # Render article with highlighted entities
                st.markdown("**Article with Highlighted Entities:**")
                st.markdown("*(Hover over highlighted text to see entity type and confidence)*")

                # Truncate long articles for display
                is_truncated = len(full_text) > 5000
                display_text = full_text[:5000] if is_truncated else full_text

                # Filter entities to only those within the display range
                display_entities = [e for e in entities if e['start_char'] < 5000]

                html_content = render_article_with_entities(display_text, display_entities)
                st.markdown(html_content, unsafe_allow_html=True)

                if is_truncated:
                    st.info("Article truncated for display (showing first 5000 characters)")
