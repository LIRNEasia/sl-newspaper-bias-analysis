"""Topics Analysis Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from data.loaders import (
    load_topics, load_topic_by_source, load_bertopic_model,
    load_topics_with_keywords, load_outlet_totals, generate_topic_aspects,
    generate_selection_bias_analysis, load_bias_narrative
)
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
    st.code(f"python3 scripts/topics/02_discover_topics.py --version-id {version_id}")
    st.info("Embeddings are auto-generated if needed, or run separately:\n"
            "`python3 scripts/embeddings/01_generate_embeddings.py --model <model>`")
    st.stop()

# Load topics with keywords for aspect label generation
topics_with_kw = load_topics_with_keywords(version_id, limit=15)

# Aspect label generation button at the top
if topics_with_kw:
    import json as _json_btn

    # Check if any topics have aspect labels
    has_aspects = False
    for t in topics_with_kw:
        try:
            if t.get('description'):
                desc_data = _json_btn.loads(t['description'])
                if desc_data.get('aspect'):
                    has_aspects = True
                    break
        except (ValueError, TypeError):
            pass

    btn_label = "Regenerate Topic Labels" if has_aspects else "Generate Topic Labels"
    btn_help = "Uses the configured LLM to generate short aspect phrases for each topic (using random article samples)"

    if st.button(btn_label, help=btn_help):
        with st.spinner("Generating aspect labels via LLM... This may take a minute."):
            count = generate_topic_aspects(version_id, topics_with_kw, force=has_aspects)
            st.success(f"Generated aspect labels for {count} topics.")
            st.rerun()

    if not has_aspects:
        st.info("💡 Topic labels are keyword-based. Generate LLM aspect labels for clearer, human-readable descriptions.")

st.divider()

topics_df = pd.DataFrame(topics)

# Extract aspect labels from description JSON when available
import json as _json_top

def get_display_name(row):
    """Return LLM aspect label if available, otherwise keyword name."""
    try:
        if row.get('description'):
            desc_data = _json_top.loads(row['description'])
            if desc_data.get('aspect'):
                return desc_data['aspect']
    except (ValueError, TypeError):
        pass
    return row['name']

topics_df['display_name'] = topics_df.apply(get_display_name, axis=1)

# Top 20 topics bar chart
top_topics = topics_df.head(20)

fig = px.bar(
    top_topics,
    x='article_count',
    y='display_name',
    orientation='h',
    labels={'article_count': 'Articles', 'display_name': 'Topic'}
)
fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig, width='stretch')

topic_source_data = load_topic_by_source(version_id)
if topic_source_data:
    ts_df = pd.DataFrame(topic_source_data)
    ts_df['source_name'] = ts_df['source_id'].map(SOURCE_NAMES)
    

st.subheader("Selection Bias Analysis")
st.markdown(
    "Which topics show the largest coverage differences between outlets? "
)
st.markdown(
    "**Spread** = max proportion - min proportion across outlets."
)

outlet_totals = load_outlet_totals()

if topics_with_kw and topic_source_data and outlet_totals:
    import json as _json
    import numpy as np

    ts_df_bias = pd.DataFrame(topic_source_data)

    topic_info = {t['name']: t for t in topics_with_kw}
    top_names = [t['name'] for t in topics_with_kw]
    ts_bias_filtered = ts_df_bias[ts_df_bias['topic'].isin(top_names)]

    # Only include outlets that have articles in the dataset
    all_outlets = [sid for sid in SOURCE_NAMES.keys() if sid in outlet_totals]

    bias_rows = []
    for t in topics_with_kw:
        tname = t['name']
        proportions = {}
        for sid in all_outlets:
            count = ts_bias_filtered[
                (ts_bias_filtered['topic'] == tname) & (ts_bias_filtered['source_id'] == sid)
            ]['count'].sum()
            total = outlet_totals.get(sid, 1)
            proportions[sid] = (count / total) * 100

        vals = list(proportions.values())
        spread = max(vals) - min(vals)
        max_outlet = SOURCE_NAMES[max(proportions, key=proportions.get)]
        min_outlet = SOURCE_NAMES[min(proportions, key=proportions.get)]

        aspect = tname
        try:
            if t.get('description'):
                desc_data = _json.loads(t['description'])
                if desc_data.get('aspect'):
                    aspect = desc_data['aspect']
        except (ValueError, TypeError):
            pass

        row = {'Aspect': aspect, 'Topic': tname}
        for sid in all_outlets:
            row[SOURCE_NAMES[sid]] = round(proportions[sid], 1)
        row['Spread'] = round(spread, 1)
        row['Most'] = max_outlet
        row['Least'] = min_outlet
        bias_rows.append(row)

    bias_df = pd.DataFrame(bias_rows)

    display_cols = ['Aspect'] + [SOURCE_NAMES[s] for s in all_outlets] + ['Spread', 'Most', 'Least']
    st.dataframe(
        bias_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            'Spread': st.column_config.NumberColumn(format="%.1f pp"),
            **{SOURCE_NAMES[s]: st.column_config.NumberColumn(format="%.1f%%") for s in all_outlets}
        }
    )

    # Selection bias narrative generation
    st.markdown("")  # Add spacing

    # Check if narrative already exists
    narrative = load_bias_narrative(version_id)
    has_narrative = narrative is not None

    # Check if any topics have bias insights
    has_insights = False
    for t in topics_with_kw:
        try:
            if t.get('description'):
                desc_data = _json.loads(t['description'])
                if desc_data.get('bias_insight'):
                    has_insights = True
                    break
        except (ValueError, TypeError):
            pass

    btn_label = "Regenerate Description" if (has_narrative or has_insights) else "Generate Description"
    btn_help = "Uses LLM to analyze selection bias patterns (outlet specialization and coverage gaps)"

    if st.button(btn_label, help=btn_help, key="generate_bias_analysis"):
        with st.spinner("Analyzing selection bias patterns... This may take 1-2 minutes."):
            results = generate_selection_bias_analysis(
                version_id,
                bias_df,
                force=(has_narrative or has_insights)
            )

            msg_parts = []
            if results['overall_success']:
                msg_parts.append("overall narrative")
            if results['topic_count'] > 0:
                msg_parts.append(f"{results['topic_count']} topic insights")

            if msg_parts:
                st.success(f"Generated {' and '.join(msg_parts)}.")
                st.rerun()
            else:
                st.error("Failed to generate bias analysis. Check logs.")

    # Display overall narrative if exists
    if narrative:
        st.markdown(narrative)

        # Optionally show per-topic insights
        with st.expander("Per-Topic Bias Insights"):
            for t in topics_with_kw:
                try:
                    if t.get('description'):
                        desc_data = _json.loads(t['description'])
                        insight = desc_data.get('bias_insight')
                        aspect = desc_data.get('aspect', t['name'])

                        if insight:
                            st.markdown(f"**{aspect}:** {insight}")
                except (ValueError, TypeError):
                    pass
    elif not has_insights:
        st.info("💡 Generate bias analysis to see LLM-powered insights about coverage patterns.")

else:
    st.info("Run topic discovery to see selection bias analysis.")

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

    # Create a mapping from topic name to display name (aspect if available)
    topic_display_map = dict(zip(topics_df['name'], topics_df['display_name']))

    comparison_data = []
    for source in SOURCE_NAMES.values():
        source_data = ts_df[ts_df['source_name'] == source]
        total = source_totals.get(source, 1)

        for topic in top_topic_names_comparison:
            topic_count = source_data[source_data['topic'] == topic]['count'].sum()

            # Use aspect name if available, otherwise truncate keywords
            display_name = topic_display_map.get(topic, topic)
            if display_name == topic:  # No aspect generated, use truncated keywords
                display_name = truncate_topic_name(topic)

            comparison_data.append({
                'Source': source,
                'Topic': display_name,  # Use aspect name or truncated keywords
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
