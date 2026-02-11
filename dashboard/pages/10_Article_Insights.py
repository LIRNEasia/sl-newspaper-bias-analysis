"""Article Insights Page - Comprehensive analysis view for individual articles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from streamlit_searchbox import st_searchbox
import plotly.express as px
import plotly.graph_objects as go

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
    load_topic_coverage_by_source,
    load_sentiment_by_source_topic,
    load_article_claims,
    load_multi_doc_summary_for_topic,
    load_multi_doc_summary_for_cluster
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

# Initialize variables for multi-document summaries
topic_data = None
topic_version_id = None
cluster = None
clustering_version_id = None

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

# Topic Sentiment Comparison - only show if article has topic and sentiment
if topic_data and sentiment:
    st.caption(f"How different outlets cover '{topic_data['topic_name']}' (using {sentiment_model} model)")

    # Load sentiment data for this topic across all outlets
    topic_sentiment_data = load_sentiment_by_source_topic(sentiment_model, topic_data['topic_name'])

    if topic_sentiment_data and len(topic_sentiment_data) > 0:
        # Transform data
        sentiment_df = pd.DataFrame(topic_sentiment_data)
        sentiment_df['source_name'] = sentiment_df['source_id'].map(SOURCE_NAMES)

        # Sort by a fixed order (using SOURCE_COLORS keys to maintain consistency)
        source_order = list(SOURCE_COLORS.keys())
        sentiment_df['source_order'] = sentiment_df['source_name'].apply(
            lambda x: source_order.index(x) if x in source_order else 999
        )
        sentiment_df = sentiment_df.sort_values('source_order')

        # Create horizontal bar chart
        fig = go.Figure()

        # Add bars with error bars (no text labels on bars)
        fig.add_trace(go.Bar(
            y=sentiment_df['source_name'],
            x=sentiment_df['avg_sentiment'],
            orientation='h',
            marker_color=[SOURCE_COLORS.get(name, '#999') for name in sentiment_df['source_name']],
            error_x=dict(
                type='data',
                symmetric=True,
                array=sentiment_df['stddev_sentiment'],
                arrayminus=sentiment_df['stddev_sentiment'],
                visible=True
            ),
            hovertemplate='<b>%{y}</b><br>' +
                         'Avg Sentiment: %{x:.2f}<br>' +
                         'Articles: %{customdata[0]}<br>' +
                         'Std Dev: %{customdata[1]:.2f}<extra></extra>',
            customdata=sentiment_df[['article_count', 'stddev_sentiment']].values
        ))

        # Add neutral reference line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")

        # Layout configuration
        fig.update_layout(
            xaxis_title="Average Sentiment Score",
            yaxis_title="Outlet",
            height=300,  # Increased height for better spacing
            xaxis_range=[-5, 5],
            showlegend=False,
            yaxis={'categoryorder': 'array', 'categoryarray': sentiment_df['source_name'].tolist()}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display summary table with sentiment details
        st.caption("Sentiment Details:")
        display_df = sentiment_df[['source_name', 'avg_sentiment', 'stddev_sentiment', 'article_count']].copy()
        display_df.columns = ['Outlet', 'Avg Sentiment', 'Std Dev', 'Articles']
        display_df['Avg Sentiment'] = display_df['Avg Sentiment'].map('{:.2f}'.format)
        display_df['Std Dev'] = display_df['Std Dev'].map('{:.2f}'.format)
        display_df['Articles'] = display_df['Articles'].astype(int)

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=180
        )

        # Add contextual note comparing article sentiment to outlet average
        current_source = SOURCE_NAMES.get(article['source_id'], article['source_id'])
        current_outlet_data = sentiment_df[sentiment_df['source_name'] == current_source]

        if not current_outlet_data.empty:
            current_sentiment = current_outlet_data.iloc[0]['avg_sentiment']
            article_sentiment = sentiment['overall_sentiment']

            # Only show if significantly different (>1.0 points)
            if abs(article_sentiment - current_sentiment) > 1.0:
                st.info(f"📊 This article's sentiment ({article_sentiment:.2f}) differs significantly from {current_source}'s average for this topic ({current_sentiment:.2f})")
    else:
        st.info("Not enough sentiment data available for cross-outlet comparison on this topic")

st.markdown("### Claims Mentioned")

if article.get('is_ditwah_cyclone'):
    claims = load_article_claims(article_id)

    if claims:

        for claim in claims:
            claim_preview = claim['claim_text']

            with st.expander(f"**Claim:** {claim_preview}"):
                st.markdown(f"**{claim['claim_text']}**")

                if claim.get('claim_category'):
                    st.caption(f"Category: {claim['claim_category'].replace('_', ' ').title()}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if claim.get('stance_score') is not None:
                        stance_score = claim['stance_score']
                        stance_label = claim.get('stance_label', 'Unknown')

                        if stance_score > 0.2:
                            stance_color = "green"
                        elif stance_score < -0.2:
                            stance_color = "red"
                        else:
                            stance_color = "orange"

                        st.metric(
                            "Stance",
                            f"{stance_score:.2f}",
                            help="Agreement score: -1 (disagree) to +1 (agree)"
                        )
                        st.markdown(f":{stance_color}[{stance_label}]")

                with col2:
                    if claim.get('sentiment_score') is not None:
                        st.metric(
                            "Sentiment",
                            f"{claim['sentiment_score']:.2f}",
                            help="Sentiment score: -5 (very negative) to +5 (very positive)"
                        )

                with col3:
                    if claim.get('confidence') is not None:
                        st.metric(
                            "Confidence",
                            f"{claim['confidence']:.0%}"
                        )

                if claim.get('supporting_quotes'):
                    st.markdown("**Supporting Quotes:**")
                    quotes = claim['supporting_quotes'] if isinstance(claim['supporting_quotes'], list) else []
                    for quote in quotes[:3]:
                        st.markdown(f"> {quote}")
    else:
        st.info("This article does not mention any tracked claims")
else:
    st.info("Claims analysis is only available for Cyclone Ditwah articles")

st.divider()
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


# Multi-document summary for topic (if article has a topic)
if topic_data:
    st.markdown("#### What Other Articles in This Topic Covered")

    # Multi-doc summarization version selector
    multi_doc_versions = list_versions(analysis_type='multi_doc_summarization')

    if multi_doc_versions:
        multi_doc_version_options = {
            f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v['id']
            for v in multi_doc_versions
        }

        selected_multi_doc_version_label = st.selectbox(
            "Multi-Doc Summarization Version",
            options=list(multi_doc_version_options.keys()),
            key="topic_multi_doc_version_selector"
        )

        multi_doc_version_id = multi_doc_version_options[selected_multi_doc_version_label]

        # Show version config
        selected_version = [v for v in multi_doc_versions if v['id'] == multi_doc_version_id][0]
        mds_config = selected_version['configuration'].get('multi_doc_summarization', {})
        method = mds_config.get('method', 'gemini')
        model = mds_config.get('llm_model', 'N/A')
        st.caption(f"Method: {method.upper()} | Model: {model}")

        with st.spinner(f"Generating multi-document summary with {method.upper()}..."):
            topic_summary = load_multi_doc_summary_for_topic(
                article_id=article_id,
                topic_version_id=topic_version_id,
                multi_doc_version_id=multi_doc_version_id
            )

        if topic_summary and 'error' not in topic_summary:
            # Show sampling info if articles were sampled
            if topic_summary.get('sampled'):
                st.caption(f"ℹ️ Summarizing {topic_summary['article_count']} of {topic_summary['sampled_from']} articles (most recent from each source)")

            st.write(topic_summary['summary_text'])

            # Show metadata
            # col1, col2, col3, col4 = st.columns(4)
            # with col1:
            #     st.metric("Articles", topic_summary['article_count'])
            # with col2:
            #     st.metric("Sources", topic_summary['source_count'])
            # with col3:
            #     st.metric("Words", topic_summary['word_count'])
            # with col4:
            #     time_sec = topic_summary['processing_time_ms'] / 1000
            #     st.metric("Time", f"{time_sec:.1f}s")
        elif topic_summary and topic_summary.get('error') == 'not_enough_articles':
            article_count = topic_summary.get('article_count', 0)
            st.warning(f"Not enough articles in this topic for multi-document summarization (need at least 2, found {article_count}).")
        elif topic_summary and topic_summary.get('error') == 'generation_failed':
            # Show detailed error message
            error_message = topic_summary.get('error_message', 'Unknown error occurred')
            article_count = topic_summary.get('article_count', 0)

            # Show sampling info if available
            if topic_summary.get('sampled'):
                st.warning(f"ℹ️ Attempted to summarize {article_count} of {topic_summary.get('sampled_from')} articles")

            st.error(f"**Failed to generate summary**\n\n{error_message}")

            # Show error type for debugging if available
            if topic_summary.get('error_type'):
                st.caption(f"Error type: {topic_summary['error_type']}")
        else:
            st.warning("Could not generate multi-document summary. This may occur if the article has no topic assignment in this version.")
    else:
        st.info("No multi-doc summarization versions found. Create one using the version management system.")
else:
    st.info("Article does not have a topic assignment. Multi-document topic summary not available.")

# Multi-document summary for event cluster (if article is in a cluster)
if cluster:
    st.markdown("#### 📰 What Other Sources Reported on This Event")

    # Multi-doc summarization version selector
    multi_doc_versions_cluster = list_versions(analysis_type='multi_doc_summarization')

    if multi_doc_versions_cluster:
        multi_doc_version_options_cluster = {
            f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v['id']
            for v in multi_doc_versions_cluster
        }

        selected_multi_doc_version_label_cluster = st.selectbox(
            "Multi-Doc Summarization Version",
            options=list(multi_doc_version_options_cluster.keys()),
            key="cluster_multi_doc_version_selector"
        )

        multi_doc_version_id_cluster = multi_doc_version_options_cluster[selected_multi_doc_version_label_cluster]

        # Show version config
        selected_version_cluster = [v for v in multi_doc_versions_cluster if v['id'] == multi_doc_version_id_cluster][0]
        mds_config_cluster = selected_version_cluster['configuration'].get('multi_doc_summarization', {})
        method_cluster = mds_config_cluster.get('method', 'gemini')
        model_cluster = mds_config_cluster.get('llm_model', 'N/A')
        st.caption(f"Method: {method_cluster.upper()} | Model: {model_cluster}")

        with st.spinner(f"Generating multi-document summary with {method_cluster.upper()}..."):
            cluster_summary = load_multi_doc_summary_for_cluster(
                article_id=article_id,
                cluster_version_id=clustering_version_id,
                multi_doc_version_id=multi_doc_version_id_cluster
            )

        if cluster_summary and 'error' not in cluster_summary:
            # Show sampling info if articles were sampled
            if cluster_summary.get('sampled'):
                st.caption(f"ℹ️ Summarizing {cluster_summary['article_count']} of {cluster_summary['sampled_from']} articles (most recent from each source)")

            st.write(cluster_summary['summary_text'])

            # Show metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Articles", cluster_summary['article_count'])
            with col2:
                st.metric("Sources", cluster_summary['source_count'])
            with col3:
                st.metric("Words", cluster_summary['word_count'])
            with col4:
                time_sec = cluster_summary['processing_time_ms'] / 1000
                st.metric("Time", f"{time_sec:.1f}s")
        elif cluster_summary and cluster_summary.get('error') == 'not_enough_articles':
            article_count = cluster_summary.get('article_count', 0)
            st.warning(f"Not enough articles in this event cluster for multi-document summarization (need at least 2, found {article_count}).")
        elif cluster_summary and cluster_summary.get('error') == 'generation_failed':
            # Show detailed error message
            error_message = cluster_summary.get('error_message', 'Unknown error occurred')
            article_count = cluster_summary.get('article_count', 0)

            # Show sampling info if available
            if cluster_summary.get('sampled'):
                st.warning(f"ℹ️ Attempted to summarize {cluster_summary['article_count']} of {cluster_summary.get('sampled_from')} articles")

            st.error(f"**Failed to generate summary**\n\n{error_message}")

            # Show error type for debugging if available
            if cluster_summary.get('error_type'):
                st.caption(f"Error type: {cluster_summary['error_type']}")
        else:
            st.warning("Could not generate multi-document summary. This may occur if the article is not part of any cluster in this version.")
    else:
        st.info("No multi-doc summarization versions found. Create one using the version management system.")
# else:
#     st.info("Article is not part of an event cluster. Multi-document cluster summary not available.")

st.divider()

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
        st.markdown(f"**Event Cluster:** {cluster['cluster_name']}")
        st.markdown(f"**Cluster Size:** {cluster['article_count']} articles from {cluster['sources_count']} sources")

        if cluster.get('other_sources'):
            other_sources = [s for s in cluster['other_sources'] if s]
            if other_sources:
                source_names = [SOURCE_NAMES.get(s, s) for s in other_sources]
                st.markdown(f"**Other sources:** {', '.join(source_names)}")

        if cluster.get('date_start') and cluster.get('date_end'):
            date_range = f"{cluster['date_start'].strftime('%Y-%m-%d')} to {cluster['date_end'].strftime('%Y-%m-%d')}"
            st.markdown(f"**Event Period:** {date_range}")

        cluster_articles = load_event_details(cluster['cluster_id'], clustering_version_id)

        if cluster_articles:
            st.markdown("**Articles in this cluster:**")

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

            df = pd.DataFrame(articles_data)

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
