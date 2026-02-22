"""Named Entity Recognition Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import html
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.db import get_db
from src.versions import get_version
from data.loaders import (
    load_entity_statistics,
    load_entities_grouped_by_type,
    load_articles_for_entity,
    load_entity_sentiment_by_source,
    load_available_models
)
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
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

# ========== ENTITY EXPLORER SECTION ==========
st.divider()
st.subheader("Entity Explorer")
st.markdown("Analyze sentiment patterns for specific entities (people, organizations, locations, etc.)")

# Load entities grouped by type
grouped_entities = load_entities_grouped_by_type(version_id, limit_per_type=20)

if not grouped_entities:
    st.info("No entities found. Run the extraction pipeline.")
else:
    # Group entities by type
    entities_by_type = {}
    for entity in grouped_entities:
        etype = entity['entity_type']
        if etype not in entities_by_type:
            entities_by_type[etype] = []
        entities_by_type[etype].append(entity)

    # Filters row
    col1, col2 = st.columns([1, 1])

    with col1:
        # Entity type filter
        entity_type_options = ["All Types"] + sorted(entities_by_type.keys())
        selected_entity_type = st.selectbox(
            "Filter by Entity Type",
            options=entity_type_options,
            key="ner_type_filter"
        )

    with col2:
        # Sentiment model selector
        available_models = load_available_models()
        if available_models:
            MODEL_DISPLAY_NAMES = {
                'roberta': 'RoBERTa',
                'distilbert': 'DistilBERT',
                'finbert': 'FinBERT',
                'vader': 'VADER',
                'textblob': 'TextBlob',
                'local': 'Local (RoBERTa)'
            }
            model_options = [m['model_type'] for m in available_models]
            selected_sentiment_model = st.selectbox(
                "Sentiment Model",
                options=model_options,
                format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x.upper()),
                key="ner_sentiment_model"
            )
        else:
            st.warning("No sentiment data available. Sentiment analysis will be skipped.")
            selected_sentiment_model = None

    # Filter entities based on type selection
    if selected_entity_type == "All Types":
        filtered_entities = grouped_entities
    else:
        filtered_entities = entities_by_type[selected_entity_type]

    # Create entity selection options
    # Format: "PERSON: Ranil Wickremesinghe (245 mentions)"
    entity_options = []
    entity_map = {}  # Maps display string to entity data

    for entity in filtered_entities:
        display_label = f"{entity['entity_type']}: {entity['entity_text']} ({entity['total_mentions']} mentions, {entity['total_articles']} articles)"
        entity_options.append(display_label)
        entity_map[display_label] = entity

    # Add placeholder at the beginning
    entity_options.insert(0, "— Select an entity to analyze —")

    # Entity selector
    selected_entity_display = st.selectbox(
        "Select Entity",
        options=entity_options,
        key="ner_entity_selector",
        help="Choose an entity to view sentiment analysis and related articles"
    )

    # Show details if an entity is selected
    if selected_entity_display != "— Select an entity to analyze —" and selected_sentiment_model:
        selected = entity_map[selected_entity_display]

        # Entity details header
        st.divider()
        st.markdown(f"## {selected['entity_text']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entity Type", selected['entity_type'])
        with col2:
            st.metric("Total Mentions", f"{selected['total_mentions']:,}")
        with col3:
            st.metric("Articles", f"{selected['total_articles']:,}")

        # Load sentiment by source
        sentiment_data = load_entity_sentiment_by_source(
            version_id,
            selected['entity_text'],
            selected['entity_type'],
            selected_sentiment_model
        )

        if sentiment_data:
            st.markdown("### Average Sentiment by Outlet")
            st.caption(f"Using {MODEL_DISPLAY_NAMES.get(selected_sentiment_model, selected_sentiment_model)} model")

            df_sentiment = pd.DataFrame(sentiment_data)
            df_sentiment['source_name'] = df_sentiment['source_id'].map(SOURCE_NAMES)

            # Horizontal bar chart with error bars
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_sentiment['source_name'],
                x=df_sentiment['avg_sentiment'],
                orientation='h',
                error_x=dict(
                    type='data',
                    array=df_sentiment['stddev_sentiment'].fillna(0)
                ),
                marker_color=[SOURCE_COLORS.get(name, '#999') for name in df_sentiment['source_name']],
                text=df_sentiment['avg_sentiment'].round(2),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Avg Sentiment: %{x:.2f}<br>Articles: %{customdata}<extra></extra>',
                customdata=df_sentiment['article_count']
            ))

            fig.update_layout(
                xaxis_title="Average Sentiment Score (-5 to +5)",
                yaxis_title="",
                height=max(300, len(df_sentiment) * 60),
                xaxis_range=[-5, 5],
                margin=dict(l=150, r=50, t=30, b=50),
                showlegend=False
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No sentiment data available for **{selected['entity_text']}** using {MODEL_DISPLAY_NAMES.get(selected_sentiment_model, selected_sentiment_model)} model")

        # Load articles for this entity
        articles = load_articles_for_entity(
            version_id,
            selected['entity_text'],
            selected['entity_type'],
            selected_sentiment_model
        )

        if articles:
            st.markdown("### Articles Mentioning This Entity")
            st.caption(f"Showing {len(articles)} articles")

            df_articles = pd.DataFrame(articles)
            df_articles['source_name'] = df_articles['source_id'].map(SOURCE_NAMES)

            # Format for display
            display_df = df_articles[[
                'title', 'source_name', 'date_posted', 'overall_sentiment', 'url'
            ]].copy()

            display_df.columns = ['Title', 'Source', 'Date', 'Sentiment', 'URL']

            # Display dataframe
            st.dataframe(
                display_df,
                column_config={
                    'URL': st.column_config.LinkColumn('URL', display_text='View'),
                    'Sentiment': st.column_config.NumberColumn('Sentiment', format='%.2f'),
                    'Date': st.column_config.DateColumn('Date', format='YYYY-MM-DD')
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
        else:
            st.warning(f"No articles found mentioning **{selected['entity_text']}**")

# ========== TOP ENTITIES BY SOURCE (EXISTING) ==========
st.divider()
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
        from src.db import ditwah_filters
        article = db.get_article_by_url(article_url, filters=ditwah_filters())

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
