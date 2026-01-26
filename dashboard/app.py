"""Sri Lanka Media Bias Dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_db
from src.versions import (
    list_versions,
    get_version,
    create_version,
    find_version_by_config,
    get_default_topic_config,
    get_default_clustering_config,
    get_default_word_frequency_config
)
from bertopic import BERTopic

# Page config
st.set_page_config(
    page_title="Sri Lanka Media Bias Detector",
    page_icon="📰",
    layout="wide"
)

# Source name mapping
SOURCE_NAMES = {
    "dailynews_en": "Daily News",
    "themorning_en": "The Morning",
    "ft_en": "Daily FT",
    "island_en": "The Island"
}

SOURCE_COLORS = {
    "Daily News": "#1f77b4",
    "The Morning": "#ff7f0e",
    "Daily FT": "#2ca02c",
    "The Island": "#d62728"
}


@st.cache_data(ttl=300)
def load_overview_stats(version_id=None):
    """Load overview statistics for a specific version."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Total articles
            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.news_articles")
            total_articles = cur.fetchone()["count"]

            # Articles by source
            cur.execute(f"""
                SELECT source_id, COUNT(*) as count
                FROM {schema}.news_articles
                GROUP BY source_id
                ORDER BY count DESC
            """)
            by_source = cur.fetchall()

            if version_id:
                # Total topics for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.topics WHERE topic_id != -1 AND result_version_id = %s",
                    (version_id,)
                )
                total_topics = cur.fetchone()["count"]

                # Total clusters for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE result_version_id = %s",
                    (version_id,)
                )
                total_clusters = cur.fetchone()["count"]

                # Multi-source clusters for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE sources_count > 1 AND result_version_id = %s",
                    (version_id,)
                )
                multi_source = cur.fetchone()["count"]
            else:
                # Fallback for no version selected
                total_topics = 0
                total_clusters = 0
                multi_source = 0

            # Date range
            cur.execute(f"""
                SELECT MIN(date_posted)::date as min_date, MAX(date_posted)::date as max_date
                FROM {schema}.news_articles
            """)
            date_range = cur.fetchone()

    return {
        "total_articles": total_articles,
        "by_source": by_source,
        "total_topics": total_topics,
        "total_clusters": total_clusters,
        "multi_source_clusters": multi_source,
        "date_range": date_range
    }


@st.cache_data(ttl=300)
def load_topics(version_id=None):
    """Load topic data for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT topic_id, name, description, article_count
                FROM {schema}.topics
                WHERE topic_id != -1 AND result_version_id = %s
                ORDER BY article_count DESC
            """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_topic_by_source(version_id=None):
    """Load topic distribution by source for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT t.name as topic, n.source_id, COUNT(*) as count
                FROM {schema}.article_analysis aa
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                JOIN {schema}.news_articles n ON aa.article_id = n.id
                WHERE t.topic_id != -1
                  AND aa.result_version_id = %s
                  AND t.result_version_id = %s
                GROUP BY t.name, n.source_id
                ORDER BY count DESC
            """, (version_id, version_id))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_top_events(version_id=None, limit=20):
    """Load top event clusters for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT ec.id, ec.cluster_name, ec.article_count, ec.sources_count,
                       ec.date_start, ec.date_end
                FROM {schema}.event_clusters ec
                WHERE ec.result_version_id = %s
                ORDER BY ec.article_count DESC
                LIMIT {limit}
            """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_event_details(event_id, version_id=None):
    """Load details for a specific event cluster."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Get articles in cluster
            cur.execute(f"""
                SELECT n.title, n.source_id, n.date_posted, n.url
                FROM {schema}.article_clusters ac
                JOIN {schema}.news_articles n ON ac.article_id = n.id
                WHERE ac.cluster_id = %s
                ORDER BY n.date_posted
            """, (event_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_coverage_timeline():
    """Load daily article counts by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT date_posted::date as date, source_id, COUNT(*) as count
                FROM {schema}.news_articles
                WHERE date_posted IS NOT NULL
                GROUP BY date_posted::date, source_id
                ORDER BY date
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_word_frequencies(version_id=None, limit=50):
    """Load word frequencies for a specific version."""
    if not version_id:
        return {}

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT source_id, word, frequency, tfidf_score, rank
                FROM {schema}.word_frequencies
                WHERE result_version_id = %s
                  AND rank <= %s
                ORDER BY source_id, rank
            """, (version_id, limit))
            rows = cur.fetchall()

            # Group by source
            result = {}
            for row in rows:
                source = row['source_id']
                if source not in result:
                    result[source] = []
                result[source].append(row)
            return result


@st.cache_resource
def load_bertopic_model(version_id=None):
    """Load the saved BERTopic model for a specific version."""
    if not version_id:
        return None

    # Try version-specific model first
    model_path = Path(__file__).parent.parent / "models" / f"bertopic_model_{version_id[:8]}"
    if not model_path.exists():
        # Fall back to default model
        model_path = Path(__file__).parent.parent / "models" / "bertopic_model"

    if model_path.exists():
        try:
            return BERTopic.load(str(model_path))
        except Exception as e:
            st.warning(f"Could not load BERTopic model: {e}")
            return None
    return None


def render_version_selector(analysis_type):
    """Render version selector for a specific analysis type.

    Args:
        analysis_type: 'topics', 'clustering', or 'word_frequency'

    Returns:
        version_id of selected version or None
    """
    # Load versions for this analysis type
    versions = list_versions(analysis_type=analysis_type)

    if not versions:
        st.warning(f"No {analysis_type} versions found!")
        st.info(f"Create a {analysis_type} version using the button below to get started")
        return None

    # Version selector
    version_options = {
        f"{v['name']} ({v['created_at'].strftime('%Y-%m-%d')})": v['id']
        for v in versions
    }

    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    selected_label = st.selectbox(
        f"Select {display_name} Version",
        options=list(version_options.keys()),
        index=0,
        key=f"{analysis_type}_version_selector"
    )

    version_id = version_options[selected_label]
    version = get_version(version_id)

    # Display version info in an expander
    with st.expander("ℹ️ Version Details"):
        st.markdown(f"**Name:** {version['name']}")
        if version['description']:
            st.markdown(f"**Description:** {version['description']}")
        st.markdown(f"**Created:** {version['created_at'].strftime('%Y-%m-%d %H:%M')}")

        # Pipeline status
        status = version['pipeline_status']
        st.markdown("**Pipeline Status:**")

        if analysis_type == 'word_frequency':
            # Word frequency only has one pipeline step
            st.caption(f"{'✅' if status.get('word_frequency') else '⭕'} Word Frequency")
        else:
            # Topics and clustering have embeddings + analysis
            cols = st.columns(2)
            with cols[0]:
                st.caption(f"{'✅' if status.get('embeddings') else '⭕'} Embeddings")
            with cols[1]:
                if analysis_type == 'topics':
                    st.caption(f"{'✅' if status.get('topics') else '⭕'} Topics")
                else:
                    st.caption(f"{'✅' if status.get('clustering') else '⭕'} Clustering")

        # Configuration preview
        config = version['configuration']
        st.markdown("**Configuration:**")

        if analysis_type == 'word_frequency':
            # Word frequency-specific settings
            wf_config = config.get('word_frequency', {})
            st.caption(f"Random Seed: {config.get('random_seed', 42)}")
            st.caption(f"Ranking Method: {wf_config.get('ranking_method', 'N/A')}")
            if wf_config.get('ranking_method') == 'tfidf':
                st.caption(f"TF-IDF Scope: {wf_config.get('tfidf_scope', 'N/A')}")
            st.caption(f"Top N Words: {wf_config.get('top_n_words', 'N/A')}")
            st.caption(f"Min Word Length: {wf_config.get('min_word_length', 'N/A')}")

            # Custom stopwords
            stopwords = wf_config.get('custom_stopwords', [])
            if stopwords:
                st.caption(f"Custom Stopwords: {', '.join(stopwords[:5])}{'...' if len(stopwords) > 5 else ''}")

        elif analysis_type == 'topics':
            # General settings
            st.caption(f"Random Seed: {config.get('random_seed', 42)}")
            st.caption(f"Embedding Model: {config.get('embeddings', {}).get('model', 'N/A')}")

            # Topic-specific settings
            topics_config = config.get('topics', {})
            st.caption(f"Min Topic Size: {topics_config.get('min_topic_size', 'N/A')}")
            st.caption(f"Diversity: {topics_config.get('diversity', 'N/A')}")

            # Stopwords
            stopwords = topics_config.get('stop_words', [])
            if stopwords:
                st.caption(f"Stop Words: {', '.join(stopwords)}")

            # Vectorizer parameters
            vectorizer_config = topics_config.get('vectorizer', {})
            if vectorizer_config:
                ngram_range = vectorizer_config.get('ngram_range', 'N/A')
                st.caption(f"N-gram Range: {ngram_range}")
                st.caption(f"Min DF: {vectorizer_config.get('min_df', 'N/A')}")

            # UMAP parameters
            umap_config = topics_config.get('umap', {})
            if umap_config:
                st.caption(f"UMAP n_neighbors: {umap_config.get('n_neighbors', 'N/A')}")
                st.caption(f"UMAP n_components: {umap_config.get('n_components', 'N/A')}")
                st.caption(f"UMAP min_dist: {umap_config.get('min_dist', 'N/A')}")
                st.caption(f"UMAP metric: {umap_config.get('metric', 'N/A')}")

            # HDBSCAN parameters
            hdbscan_config = topics_config.get('hdbscan', {})
            if hdbscan_config:
                st.caption(f"HDBSCAN min_cluster_size: {hdbscan_config.get('min_cluster_size', 'N/A')}")
                st.caption(f"HDBSCAN metric: {hdbscan_config.get('metric', 'N/A')}")
                st.caption(f"HDBSCAN cluster_selection_method: {hdbscan_config.get('cluster_selection_method', 'N/A')}")

        else:  # clustering
            # General settings
            st.caption(f"Random Seed: {config.get('random_seed', 42)}")
            st.caption(f"Embedding Model: {config.get('embeddings', {}).get('model', 'N/A')}")

            # Clustering-specific settings
            clustering_config = config.get('clustering', {})
            st.caption(f"Similarity Threshold: {clustering_config.get('similarity_threshold', 'N/A')}")
            st.caption(f"Time Window: {clustering_config.get('time_window_days', 'N/A')} days")
            st.caption(f"Min Cluster Size: {clustering_config.get('min_cluster_size', 'N/A')}")

    return version_id


def render_create_version_button(analysis_type):
    """Render button to create a new version for a specific analysis type.

    Args:
        analysis_type: 'topics', 'clustering', or 'word_frequency'
    """
    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    if st.button(f"➕ Create New {display_name} Version", key=f"create_{analysis_type}_btn"):
        st.session_state[f'show_create_{analysis_type}'] = True

    # Show create dialog if requested
    if st.session_state.get(f'show_create_{analysis_type}', False):
        render_create_version_form(analysis_type)


def render_create_version_form(analysis_type):
    """Render form for creating a new version.

    Args:
        analysis_type: 'topics', 'clustering', or 'word_frequency'
    """
    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    st.markdown("---")
    st.subheader(f"Create New {display_name} Version")

    with st.form(f"create_{analysis_type}_form"):
        name = st.text_input("Version Name", placeholder=f"e.g., baseline-{analysis_type}")
        description = st.text_area("Description (optional)", placeholder="What makes this version unique?")

        # Configuration editor
        st.markdown("**Configuration (JSON)**")
        if analysis_type == 'topics':
            default_config = get_default_topic_config()
        elif analysis_type == 'clustering':
            default_config = get_default_clustering_config()
        elif analysis_type == 'word_frequency':
            default_config = get_default_word_frequency_config()
        else:
            default_config = {}

        config_str = st.text_area(
            "Edit configuration",
            value=json.dumps(default_config, indent=2),
            height=300,
            key=f"{analysis_type}_config_editor"
        )

        col1, col2 = st.columns(2)

        with col1:
            submit = st.form_submit_button("Create Version")
        with col2:
            cancel = st.form_submit_button("Cancel")

        if cancel:
            st.session_state[f'show_create_{analysis_type}'] = False
            st.rerun()

        if submit:
            if not name:
                st.error("Version name is required")
            else:
                try:
                    # Parse configuration
                    config = json.loads(config_str)

                    # Check if config already exists for this analysis type
                    existing = find_version_by_config(config, analysis_type=analysis_type)
                    if existing:
                        st.warning(f"A {analysis_type} version with this configuration already exists: **{existing['name']}**")
                        st.info(f"Version ID: {existing['id']}")
                    else:
                        # Create version
                        version_id = create_version(name, description, config, analysis_type=analysis_type)
                        st.success(f"✅ Created {analysis_type} version: {name}")
                        st.info(f"Version ID: {version_id}")

                        # Show pipeline instructions
                        st.markdown("**Next steps:** Run the pipeline")
                        if analysis_type == 'word_frequency':
                            st.code(f"""# Compute word frequencies
python3 scripts/word_frequency/01_compute_word_frequency.py --version-id {version_id}""")
                        else:
                            st.code(f"""# Generate embeddings
python3 scripts/{analysis_type}/01_generate_embeddings.py --version-id {version_id}

# Run analysis
python3 scripts/{analysis_type}/02_{'discover_topics' if analysis_type == 'topics' else 'cluster_events'}.py --version-id {version_id}""")

                        # Hide dialog
                        st.session_state[f'show_create_{analysis_type}'] = False

                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON configuration: {e}")
                except Exception as e:
                    st.error(f"Error creating version: {e}")


def main():
    st.title("🇱🇰 Sri Lanka Media Bias Detector")
    st.markdown("Analyzing coverage patterns across Sri Lankan English newspapers")

    # Initialize session state for tabs
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Load overview stats (no version required for coverage)
    stats = load_overview_stats()

    # Overview metrics
    st.header("Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Articles", f"{stats['total_articles']:,}")
    with col2:
        if stats['date_range']['min_date']:
            st.caption(f"**Date range:** {stats['date_range']['min_date']} to {stats['date_range']['max_date']}")

    st.divider()

    # Tabs for different views
    tab_names = ["📊 Coverage", "🏷️ Topics", "📰 Events", "📝 Word Frequency"]

    # Create buttons to switch tabs
    cols = st.columns(4)
    for idx, (col, tab_name) in enumerate(zip(cols, tab_names)):
        with col:
            if st.button(tab_name, key=f"tab_{idx}",
                        type="primary" if st.session_state.active_tab == idx else "secondary"):
                st.session_state.active_tab = idx

    st.divider()

    # Render the active tab
    if st.session_state.active_tab == 0:
        render_coverage_tab(stats)
    elif st.session_state.active_tab == 1:
        render_topics_tab()
    elif st.session_state.active_tab == 2:
        render_events_tab()
    elif st.session_state.active_tab == 3:
        render_word_frequency_tab()


def render_coverage_tab(stats):
    """Render coverage analysis tab."""
    st.subheader("Article Coverage by Source")

    # Articles by source bar chart
    source_df = pd.DataFrame(stats['by_source'])
    source_df['source_name'] = source_df['source_id'].map(SOURCE_NAMES)

    fig = px.bar(
        source_df,
        x='source_name',
        y='count',
        color='source_name',
        color_discrete_map=SOURCE_COLORS,
        labels={'count': 'Articles', 'source_name': 'Source'}
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Timeline
    st.subheader("Coverage Over Time")
    timeline_data = load_coverage_timeline()

    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        timeline_df['source_name'] = timeline_df['source_id'].map(SOURCE_NAMES)

        fig = px.line(
            timeline_df,
            x='date',
            y='count',
            color='source_name',
            color_discrete_map=SOURCE_COLORS,
            labels={'count': 'Articles', 'date': 'Date', 'source_name': 'Source'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_topics_tab():
    """Render topics analysis and source comparison tab."""
    st.subheader("📊 Topic Analysis")

    # Version selector at the top
    version_id = render_version_selector('topics')

    # Create version button
    render_create_version_button('topics')

    if not version_id:
        return

    st.markdown("---")

    topics = load_topics(version_id)
    if not topics:
        st.warning("No topics found for this version. Run topic discovery first.")
        st.code(f"""python3 scripts/topics/01_generate_embeddings.py --version-id {version_id}
python3 scripts/topics/02_discover_topics.py --version-id {version_id}""")
        return

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
    st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
            if st.button("← Previous", disabled=st.session_state.topic_focus_page == 0, key="prev_topics"):
                st.session_state.topic_focus_page = max(0, st.session_state.topic_focus_page - 1)
                st.rerun()
        with col2:
            st.caption(f"Showing topics {st.session_state.topic_focus_page * topics_per_page + 1}-{min((st.session_state.topic_focus_page + 1) * topics_per_page, total_topics)} of {total_topics}")
        with col3:
            if st.button("Next →", disabled=st.session_state.topic_focus_page >= max_page, key="next_topics"):
                st.session_state.topic_focus_page = min(max_page, st.session_state.topic_focus_page + 1)
                st.rerun()

        # Get topics for current page
        start_idx = st.session_state.topic_focus_page * topics_per_page
        end_idx = start_idx + topics_per_page
        top_topic_names_comparison = [t['name'] for t in topics[start_idx:end_idx]]

        comparison_data = []
        for source in SOURCE_NAMES.values():
            source_data = ts_df[ts_df['source_name'] == source]
            total = source_totals.get(source, 1)

            for topic in top_topic_names_comparison:
                topic_count = source_data[source_data['topic'] == topic]['count'].sum()
                comparison_data.append({
                    'Source': source,
                    'Topic': topic,
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
            xaxis=dict(tickfont=dict(size=14))  # Increased font size from default (~12) to 14
        )
        st.plotly_chart(fig, use_container_width=True)

        # # Selection bias indicator
        # st.markdown("### Selection Bias Indicators")
        # st.markdown("Topics where sources significantly differ in coverage")

        # Calculate variance in coverage percentage across sources
        variance_data = []
        for topic in top_topic_names_comparison:
            topic_data = comp_df[comp_df['Topic'] == topic]
            if len(topic_data) > 1:
                variance_data.append({
                    'Topic': topic,
                    'Coverage Variance': topic_data['Percentage'].var(),
                    'Max Coverage': topic_data['Percentage'].max(),
                    'Min Coverage': topic_data['Percentage'].min(),
                    'Range': topic_data['Percentage'].max() - topic_data['Percentage'].min()
                })

        var_df = pd.DataFrame(variance_data).sort_values('Range', ascending=False)

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.markdown("**Highest Variation (potential selection bias)**")
        #     st.dataframe(
        #         var_df.head(5)[['Topic', 'Range']].rename(columns={'Range': 'Coverage Gap (%)'}),
        #         use_container_width=True
        #     )
        # with col2:
        #     st.markdown("**Most Consistent Coverage**")
        #     st.dataframe(
        #         var_df.tail(5)[['Topic', 'Range']].rename(columns={'Range': 'Coverage Gap (%)'}),
        #         use_container_width=True
        #     )

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
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Topic Bar Charts":
                st.markdown("**Top words per topic**")
                # Show top 20 topics
                top_topics_ids = [t['topic_id'] for t in topics[:20]]
                fig = topic_model.visualize_barchart(top_n_topics=20, topics=top_topics_ids)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Topic Similarity Heatmap":
                st.markdown("**Similarity matrix between topics**")
                st.caption("Darker colors indicate higher similarity")
                # Limit to top 20 topics for readability
                top_topics_ids = [t['topic_id'] for t in topics[:20]]
                fig = topic_model.visualize_heatmap(topics=top_topics_ids)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Hierarchical Topic Clustering":
                st.markdown("**Hierarchical clustering of topics**")
                st.caption("Shows how topics group into broader categories")
                fig = topic_model.visualize_hierarchy()
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating visualization: {e}")
    else:
        st.info("BERTopic model not found. Save the model during topic discovery.")


def render_events_tab():
    """Render events analysis tab."""
    st.subheader("📰 Event Clustering Analysis")

    # Version selector at the top
    version_id = render_version_selector('clustering')

    # Create version button
    render_create_version_button('clustering')

    if not version_id:
        return

    st.markdown("---")

    events = load_top_events(version_id, 30)
    if not events:
        st.warning("No event clusters found for this version. Run clustering first.")
        st.code(f"""python3 scripts/clustering/01_generate_embeddings.py --version-id {version_id}
python3 scripts/clustering/02_cluster_events.py --version-id {version_id}""")
        return

    events_df = pd.DataFrame(events)

    # Filter to multi-source events
    multi_source_events = events_df[events_df['sources_count'] > 1]

    # Event selector
    event_options = {
        f"{e['cluster_name']}... ({e['article_count']} articles, {e['sources_count']} sources)": e['id']
        for _, e in multi_source_events.iterrows()
    }

    selected_event_label = st.selectbox(
        "Select an event to explore",
        options=list(event_options.keys())
    )

    if selected_event_label:
        event_id = event_options[selected_event_label]
        articles = load_event_details(event_id, version_id)

        if articles:
            articles_df = pd.DataFrame(articles)
            articles_df['source_name'] = articles_df['source_id'].map(SOURCE_NAMES)

            # Source breakdown
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Coverage by Source**")
                source_counts = articles_df['source_name'].value_counts()

                fig = px.pie(
                    values=source_counts.values,
                    names=source_counts.index,
                    color=source_counts.index,
                    color_discrete_map=SOURCE_COLORS
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Articles in this Event**")
                display_df = articles_df[['title', 'source_name', 'date_posted']].copy()
                display_df.columns = ['Title', 'Source', 'Date']
                st.dataframe(display_df, use_container_width=True, height=300)



def render_word_frequency_tab():
    """Render word frequency analysis tab."""
    st.subheader("📝 Word Frequency Analysis")

    # Version selector
    version_id = render_version_selector('word_frequency')

    # Create version button
    render_create_version_button('word_frequency')

    if not version_id:
        st.info("👆 Select or create a word frequency version to view analysis")
        return

    # st.markdown("---")

    # Get version details
    version = get_version(version_id)
    if not version:
        st.error("Version not found")
        return
    
    config = version['configuration']
    wf_config = config.get('word_frequency', {})

    # # Show version info
    # with st.expander("ℹ️ Version Configuration", expanded=False):
        
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.metric("Ranking Method", wf_config.get('ranking_method', 'N/A').upper())
    #     with col2:
    #         if wf_config.get('ranking_method') == 'tfidf':
    #             st.metric("TF-IDF Scope", wf_config.get('tfidf_scope', 'N/A'))
    #         else:
    #             st.metric("Top Words", wf_config.get('top_n_words', 50))
    #     with col3:
    #         st.metric("Min Word Length", wf_config.get('min_word_length', 3))

    # Load word frequencies
    word_freqs = load_word_frequencies(version_id)

    if not word_freqs:
        st.warning("⚠️ No word frequencies found for this version. Run the pipeline first:")
        st.code(f"python3 scripts/word_frequency/01_compute_word_frequency.py --version-id {version_id}")
        return

    # Get configuration for display
    ranking_method = wf_config.get('ranking_method', 'frequency')

    # Top words per source
    st.subheader("Top Words by Source")

    # Create 2x2 grid for sources
    sources = list(word_freqs.keys())
    num_sources = len(sources)

    if num_sources == 0:
        st.warning("No sources found")
        return

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
            st.plotly_chart(fig, use_container_width=True)

    # # Cross-source comparison
    # st.divider()
    # st.subheader("Word Comparison Across Sources")

    # # Find common words across sources
    # word_sets = {source_id: set([w['word'] for w in words[:50]]) for source_id, words in word_freqs.items()}

    # # Calculate overlaps
    # if len(word_sets) >= 2:
    #     source_ids = list(word_sets.keys())

    #     # Create comparison matrix
    #     st.markdown("**Top Word Overlap Between Sources**")

    #     overlap_data = []
    #     for i, source1 in enumerate(source_ids):
    #         for source2 in source_ids[i+1:]:
    #             common = word_sets[source1] & word_sets[source2]
    #             overlap_pct = len(common) / 50 * 100
    #             overlap_data.append({
    #                 'Source 1': SOURCE_NAMES.get(source1, source1),
    #                 'Source 2': SOURCE_NAMES.get(source2, source2),
    #                 'Common Words': len(common),
    #                 'Overlap %': f"{overlap_pct:.1f}%"
    #             })

    #     if overlap_data:
    #         overlap_df = pd.DataFrame(overlap_data)
    #         st.dataframe(overlap_df, use_container_width=True, hide_index=True)

    # # Show unique words per source
    # st.markdown("**Distinctive Words per Source** (words appearing in top 50 of only one source)")

    # # Find words unique to each source
    # all_words = set()
    # for words_set in word_sets.values():
    #     all_words.update(words_set)

    # unique_words = {}
    # for source_id, words_set in word_sets.items():
    #     # Words that appear in this source but not in any other source's top 50
    #     other_words = set()
    #     for other_id, other_set in word_sets.items():
    #         if other_id != source_id:
    #             other_words.update(other_set)

    #     unique = words_set - other_words
    #     if unique:
    #         unique_words[source_id] = unique

    # # Display unique words
    # if unique_words:
    #     unique_cols = st.columns(len(unique_words))
    #     for idx, (source_id, words) in enumerate(unique_words.items()):
    #         source_name = SOURCE_NAMES.get(source_id, source_id)
    #         with unique_cols[idx]:
    #             st.markdown(f"**{source_name}**")
    #             if words:
    #                 st.write(", ".join(sorted(list(words)[:15])))
    #             else:
    #                 st.write("(none)")
    # else:
    #     st.info("No distinctive words found - all sources share similar vocabulary in their top 50 words")


if __name__ == "__main__":
    main()
