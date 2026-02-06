"""Article Insights Page - Comprehensive analysis view for individual articles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from streamlit_searchbox import st_searchbox
import plotly.express as px

from src.versions import list_versions
from data.loaders import (
    search_articles_by_title,
    load_article_by_id,
    load_article_sentiment,
    load_article_topic,
    load_article_summary,
    load_article_entities,
    load_article_cluster,
    load_event_details,
    get_available_sentiment_models,
    load_topic_coverage_by_source
)
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.styling import apply_page_style

apply_page_style()

st.title("Article Insights")

st.subheader("Search & Select Article")

if 'article_mapping' not in st.session_state:
    st.session_state.article_mapping = {}


def search_articles(search_term: str) -> list:
    """Search function for autocomplete that returns article titles with metadata."""
    if not search_term or len(search_term) < 2:
        return []

    results = search_articles_by_title(search_term, limit=50)

    if not results:
        return []

    suggestions = []
    for article in results:
        source_name = SOURCE_NAMES.get(article['source_id'], article['source_id'])
        date_str = article['date_posted'].strftime('%Y-%m-%d') if article['date_posted'] else 'Unknown'
        # Format: "Title - Source (Date)"
        label = f"{article['title']} - {source_name} ({date_str})"
        suggestions.append(label)
        st.session_state.article_mapping[label] = article['id']

    return suggestions


selected_label = st_searchbox(
    search_articles,
    key="article_searchbox",
    placeholder="Start typing to search articles by title...",
    label="Search by title",
    clear_on_submit=False,
    rerun_on_update=True
)

if not selected_label:
    st.info("👆 Start typing in the search box to find articles")
    st.stop()

article_id = st.session_state.article_mapping.get(selected_label)

if not article_id:
    st.info("Start typing in the search box to find articles")
    st.stop()

article = load_article_by_id(article_id)

if not article:
    st.error("Article not found")
    st.stop()

st.divider()
st.subheader("Article Metadata")

st.markdown(f"**Title:** {article['title']}")
source_name = SOURCE_NAMES.get(article['source_id'], article['source_id'])
st.markdown(f"**Source:** {source_name}")
if article['date_posted']:
    st.markdown(f"**Published:** {article['date_posted'].strftime('%Y-%m-%d')}")
if article['url']:
    st.markdown(f"[View original article]({article['url']})")

st.divider()

st.markdown("### Topic Assignment")

topic_versions = list_versions(analysis_type='topics')

if topic_versions:
    topic_version_options = {
        f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v['id']
        for v in topic_versions
    }

    selected_topic_version_label = st.selectbox(
        "Topic Version",
        options=list(topic_version_options.keys()),
        key="topic_version_selector"
    )

    topic_version_id = topic_version_options[selected_topic_version_label]
    topic_data = load_article_topic(article_id, topic_version_id)

    if topic_data:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Primary Topic:** {topic_data['topic_name']}")

        # with col2:
        #     if topic_data.get('topic_confidence'):
        #         st.metric(
        #             "Confidence",
        #             f"{topic_data['topic_confidence']:.2%}"
        #         )

        # st.markdown("#### Coverage of This Topic Across Outlets")
        st.caption("Shows how much each outlet covers this topic relative to their total coverage")

        topic_coverage = load_topic_coverage_by_source(topic_data['topic_name'], topic_version_id)

        if topic_coverage:
            # Prepare data for visualization
            coverage_data = []
            for row in topic_coverage:
                source_name = SOURCE_NAMES.get(row['source_id'], row['source_id'])
                coverage_data.append({
                    'Source': source_name,
                    'Article Count': row['article_count'],
                    'Percentage': row['percentage']
                })

            coverage_df = pd.DataFrame(coverage_data)
            # Sort by article count descending (most coverage first)
            coverage_df = coverage_df.sort_values('Article Count', ascending=False)

            # Horizontal bar chart
            fig = px.bar(
                coverage_df,
                x='Percentage',
                y='Source',
                orientation='h',
                color='Source',
                color_discrete_map=SOURCE_COLORS,
                labels={'Percentage': '% of Source\'s Coverage'},
                text=coverage_df.apply(
                    lambda row: f"{row['Article Count']} articles ({row['Percentage']:.1f}%)",
                    axis=1
                )
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=300,
                showlegend=False,
                xaxis_title='% of Outlet\'s Total Coverage',
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No coverage data available for this topic")
    else:
        st.info(f"Article does not belong to any topic (outlier).")
else:
    st.info("No topic versions found. Create and run a topic analysis version first.")

st.markdown("### Sentiment Analysis")

available_models = get_available_sentiment_models()

if available_models:
    sentiment_model = st.selectbox(
        "Sentiment Model",
        options=available_models,
        index=available_models.index('roberta') if 'roberta' in available_models else 0,
        key="sentiment_model_selector"
    )

    sentiment = load_article_sentiment(article_id, sentiment_model)

    if sentiment:
        col1, col2, col3 = st.columns(3)

        with col1:
            overall = sentiment['overall_sentiment']
            st.metric(
                "Overall Sentiment",
                f"{overall:.2f}",
                help="Range: -5 (very negative) to +5 (very positive)"
            )

        with col2:
            headline = sentiment['headline_sentiment']
            st.metric(
                "Headline Sentiment",
                f"{headline:.2f}",
                help="Range: -5 (very negative) to +5 (very positive)"
            )

        with col3:
            if sentiment.get('overall_confidence'):
                st.metric(
                    "Confidence",
                    f"{sentiment['overall_confidence']:.2%}"
                )

        if sentiment.get('sentiment_reasoning'):
            st.markdown("**Reasoning:**")
            st.write(sentiment['sentiment_reasoning'])
    else:
        st.info(f"No sentiment analysis found for model: {sentiment_model}")
else:
    st.warning("No sentiment models have analyzed articles yet. Run sentiment analysis pipeline first.")


st.markdown("### Summary")

summarization_versions = list_versions(analysis_type='summarization')

if summarization_versions:
    summary_version_options = {
        f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v['id']
        for v in summarization_versions
    }

    selected_summary_version_label = st.selectbox(
        "Summarization Version",
        options=list(summary_version_options.keys()),
        key="summary_version_selector"
    )

    summary_version_id = summary_version_options[selected_summary_version_label]
    summary = load_article_summary(article_id, summary_version_id)

    if summary:
        st.write(summary['summary_text'])
    else:
        st.info("Article not summarized in this version")
else:
    st.info("No summarization versions found. Create and run a summarization version first.")

# Named Entities Section
st.markdown("### Actors Involved")

ner_versions = list_versions(analysis_type='ner')

if ner_versions:
    ner_version_options = {
        f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v['id']
        for v in ner_versions
    }

    selected_ner_version_label = st.selectbox(
        "NER Version",
        options=list(ner_version_options.keys()),
        key="ner_version_selector"
    )

    ner_version_id = ner_version_options[selected_ner_version_label]
    entities = load_article_entities(article_id, ner_version_id)

    if entities:
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity['entity_type']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        # Display entities grouped by type (exclude Date, Money, Time)
        excluded_types = {'DATE', 'MONEY', 'TIME', 'PERCENT'}
        for entity_type in sorted(entities_by_type.keys()):
            if entity_type in excluded_types:
                continue
            entity_list = entities_by_type[entity_type]
            st.markdown(f"**{entity_type}** ({len(entity_list)})")

            # Create tags for entities (deduplicated)
            unique_entities = set()
            entity_tags = []
            for entity in entity_list:
                entity_text = entity['entity_text']
                if entity_text not in unique_entities:
                    unique_entities.add(entity_text)
                    tag = f"- `{entity_text}`"
                    entity_tags.append(tag)

            st.markdown("\n".join(entity_tags))
    else:
        st.info("No entities extracted in this version")
else:
    st.info("No NER versions found. Create and run an NER version first.")

# Event Clustering Section
st.markdown("### Event Clustering")

clustering_versions = list_versions(analysis_type='clustering')

if clustering_versions:
    clustering_version_options = {
        f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v['id']
        for v in clustering_versions
    }

    selected_clustering_version_label = st.selectbox(
        "Clustering Version",
        options=list(clustering_version_options.keys()),
        key="clustering_version_selector"
    )

    clustering_version_id = clustering_version_options[selected_clustering_version_label]
    cluster = load_article_cluster(article_id, clustering_version_id)

    if cluster:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Event Cluster:** {cluster['cluster_name']}")

        with col2:
            if cluster.get('similarity_score'):
                st.metric(
                    "Similarity",
                    f"{cluster['similarity_score']:.2%}"
                )

        # Display cluster details
        st.markdown(f"**Cluster Size:** {cluster['article_count']} articles from {cluster['sources_count']} sources")

        # Display other sources covering this event
        if cluster.get('other_sources'):
            other_sources = [s for s in cluster['other_sources'] if s]
            if other_sources:
                source_names = [SOURCE_NAMES.get(s, s) for s in other_sources]
                st.markdown(f"**Other sources:** {', '.join(source_names)}")

        # Display date range
        if cluster.get('date_start') and cluster.get('date_end'):
            date_range = f"{cluster['date_start'].strftime('%Y-%m-%d')} to {cluster['date_end'].strftime('%Y-%m-%d')}"
            st.markdown(f"**Event Period:** {date_range}")

        # Load and display articles in this cluster
        cluster_articles = load_event_details(cluster['cluster_id'], clustering_version_id)

        if cluster_articles:
            st.markdown("**Articles in this cluster:**")

            # Create a dataframe for better display
            articles_data = []
            for art in cluster_articles:
                source_name = SOURCE_NAMES.get(art['source_id'], art['source_id'])
                date_str = art['date_posted'].strftime('%Y-%m-%d') if art['date_posted'] else 'Unknown'
                articles_data.append({
                    'Title': art['title'],
                    'Source': source_name,
                    'Date': date_str,
                    'URL': art['url'] if art['url'] else ''
                })

            # Display as dataframe
            df = pd.DataFrame(articles_data)

            # Make clickable links in the dataframe
            st.dataframe(
                df,
                column_config={
                    "URL": st.column_config.LinkColumn("URL", display_text="View"),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("Article is not part of any event cluster (outlier)")
else:
    st.info("No clustering versions found. Create and run a clustering version first.")

# Footer
st.divider()
