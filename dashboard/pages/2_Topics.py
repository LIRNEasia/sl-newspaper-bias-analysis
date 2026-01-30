"""Topics Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from data.loaders import load_topics, load_topic_by_source, load_bertopic_model
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector, render_create_version_button
from components.styling import apply_page_style

apply_page_style()

st.title("Topic Analysis")

# Create version button at the top
render_create_version_button('topics')

# Version selector
version_id = render_version_selector('topics')

if not version_id:
    st.stop()

topics = load_topics(version_id)
if not topics:
    st.warning("No topics found for this version. Run topic discovery first.")
    st.code(f"""python3 scripts/topics/01_generate_embeddings.py --version-id {version_id}
python3 scripts/topics/02_discover_topics.py --version-id {version_id}""")
    st.stop()

topics_df = pd.DataFrame(topics)

# Top 20 topics bar chart
top_topics = topics_df.head(20)

fig = px.bar(
    top_topics,
    x='article_count',
    y='name',
    orientation='h',
    labels={'article_count': 'Articles', 'name': 'Topic'}
)
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig, width='stretch')

# Topic by source heatmap
st.subheader("Topic Coverage by Source")

topic_source_data = load_topic_by_source(version_id)
if topic_source_data:
    ts_df = pd.DataFrame(topic_source_data)
    ts_df['source_name'] = ts_df['source_id'].map(SOURCE_NAMES)

    # Get top 15 topics for heatmap
    top_topic_names = topics_df.head(15)['name'].tolist()
    ts_filtered = ts_df[ts_df['topic'].isin(top_topic_names)]

    # Pivot for heatmap
    pivot_df = ts_filtered.pivot(index='topic', columns='source_name', values='count').fillna(0)

    fig = px.imshow(
        pivot_df,
        labels=dict(x="Source", y="Topic", color="Articles"),
        color_continuous_scale='Blues',
        aspect='auto'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')

# Source comparison section
st.divider()

# Topic coverage comparison
st.markdown("### Topic Focus by Source")
st.markdown("What percentage of each source's coverage goes to each topic?")

if topic_source_data:
    # Initialize session state for topic pagination
    if 'topic_focus_page' not in st.session_state:
        st.session_state.topic_focus_page = 0

    # Calculate percentages per source
    source_totals = ts_df.groupby('source_name')['count'].sum()

    # Get all topics (we'll paginate through them)
    topics_per_page = 10
    total_topics = len(topics)
    max_page = (total_topics - 1) // topics_per_page

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("Previous", disabled=st.session_state.topic_focus_page == 0, key="prev_topics"):
            st.session_state.topic_focus_page = max(0, st.session_state.topic_focus_page - 1)
            st.rerun()
    with col2:
        st.caption(f"Showing topics {st.session_state.topic_focus_page * topics_per_page + 1}-{min((st.session_state.topic_focus_page + 1) * topics_per_page, total_topics)} of {total_topics}")
    with col3:
        if st.button("Next", disabled=st.session_state.topic_focus_page >= max_page, key="next_topics"):
            st.session_state.topic_focus_page = min(max_page, st.session_state.topic_focus_page + 1)
            st.rerun()

    # Get topics for current page
    start_idx = st.session_state.topic_focus_page * topics_per_page
    end_idx = start_idx + topics_per_page
    top_topic_names_comparison = [t['name'] for t in topics[start_idx:end_idx]]

    # Helper function to truncate topic names to first 3 n-grams
    def truncate_topic_name(topic_name, max_ngrams=3):
        """Truncate topic name to first N n-grams."""
        parts = topic_name.split()
        return ' '.join(parts[:max_ngrams])

    comparison_data = []
    for source in SOURCE_NAMES.values():
        source_data = ts_df[ts_df['source_name'] == source]
        total = source_totals.get(source, 1)

        for topic in top_topic_names_comparison:
            topic_count = source_data[source_data['topic'] == topic]['count'].sum()
            comparison_data.append({
                'Source': source,
                'Topic': truncate_topic_name(topic),  # Use truncated name for display
                'Percentage': (topic_count / total) * 100
            })

    comp_df = pd.DataFrame(comparison_data)

    fig = px.bar(
        comp_df,
        x='Topic',
        y='Percentage',
        color='Source',
        barmode='group',
        color_discrete_map=SOURCE_COLORS,
        labels={'Percentage': '% of Coverage'}
    )
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        xaxis=dict(tickfont=dict(size=14))
    )
    st.plotly_chart(fig, width='stretch')

# BERTopic Visualizations
st.divider()
st.subheader("Topic Model Visualizations")

topic_model = load_bertopic_model(version_id)
if topic_model:
    viz_option = st.selectbox(
        "Select visualization",
        [
            "Topic Similarity Map (2D)",
            "Topic Bar Charts",
            "Topic Similarity Heatmap",
            "Hierarchical Topic Clustering"
        ]
    )

    try:
        if viz_option == "Topic Similarity Map (2D)":
            st.markdown("**Interactive 2D visualization of topic relationships**")
            st.caption("Topics closer together are more semantically similar")
            fig = topic_model.visualize_topics()
            st.plotly_chart(fig, width='stretch')

        elif viz_option == "Topic Bar Charts":
            st.markdown("**Top words per topic**")
            # Show top 20 topics
            top_topics_ids = [t['topic_id'] for t in topics[:20]]
            fig = topic_model.visualize_barchart(top_n_topics=20, topics=top_topics_ids)
            st.plotly_chart(fig, width='stretch')

        elif viz_option == "Topic Similarity Heatmap":
            st.markdown("**Similarity matrix between topics**")
            st.caption("Darker colors indicate higher similarity")
            # Limit to top 20 topics for readability
            top_topics_ids = [t['topic_id'] for t in topics[:20]]
            fig = topic_model.visualize_heatmap(topics=top_topics_ids)
            st.plotly_chart(fig, width='stretch')

        elif viz_option == "Hierarchical Topic Clustering":
            st.markdown("**Hierarchical clustering of topics**")
            st.caption("Shows how topics group into broader categories")
            fig = topic_model.visualize_hierarchy()
            st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"Error generating visualization: {e}")
else:
    st.info("BERTopic model not found. Save the model during topic discovery.")

# Browse Articles by Topic
st.divider()
st.subheader("Browse Articles by Topic")

# Create dropdown with all topics (including topic_id)
topic_options = [f"[{t['topic_id']}] {t['name']}" for t in topics]
topic_name_map = {f"[{t['topic_id']}] {t['name']}": t['name'] for t in topics}

selected_topic_display = st.selectbox(
    "Select a topic to view articles",
    options=topic_options,
    key="topic_selector"
)

if selected_topic_display:
    from data.loaders import load_articles_by_topic

    # Get the actual topic name from the display string
    selected_topic = topic_name_map[selected_topic_display]
    articles = load_articles_by_topic(version_id, selected_topic)

    if articles:
        st.markdown(f"**{len(articles)} articles in this topic:**")

        # Display articles
        for article in articles:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Make title a clickable link
                st.markdown(f"[{article['title']}]({article['url']})")
            with col2:
                # Show source and date
                st.caption(f"{SOURCE_NAMES.get(article['source_id'], article['source_id'])} • {article['date_posted'].strftime('%Y-%m-%d')}")
    else:
        st.info("No articles found for this topic.")
