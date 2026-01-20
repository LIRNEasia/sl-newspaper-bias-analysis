"""Sri Lanka Media Bias Dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_db
from src.versions import (
    list_versions,
    get_version,
    create_version,
    find_version_by_config,
    get_default_config
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


def render_version_sidebar():
    """Render version selection and management in sidebar."""
    st.sidebar.header("📊 Result Version")

    # Load available versions
    versions = list_versions()

    if not versions:
        st.sidebar.warning("No result versions found!")
        st.sidebar.markdown("Run the pipeline to create a version:")
        st.sidebar.code("python3 scripts/run_pipeline.py --name baseline")
        return None

    # Version selector
    version_options = {
        f"{v['name']}": v['id']
        for v in versions
    }

    selected_name = st.sidebar.selectbox(
        "Select Version",
        options=list(version_options.keys()),
        index=0
    )

    version_id = version_options[selected_name]
    version = get_version(version_id)

    # Display version info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version Info**")
    st.sidebar.caption(f"**Name:** {version['name']}")
    if version['description']:
        st.sidebar.caption(f"**Description:** {version['description']}")
    st.sidebar.caption(f"**Created:** {version['created_at'].strftime('%Y-%m-%d %H:%M')}")

    # Pipeline status
    status = version['pipeline_status']
    st.sidebar.markdown("**Pipeline Status:**")
    st.sidebar.caption(f"{'✅' if status.get('embeddings') else '⭕'} Embeddings")
    st.sidebar.caption(f"{'✅' if status.get('topics') else '⭕'} Topics")
    st.sidebar.caption(f"{'✅' if status.get('clustering') else '⭕'} Clustering")

    if version['is_complete']:
        st.sidebar.success("Complete")
    else:
        st.sidebar.warning("Incomplete")

    # Configuration preview
    with st.sidebar.expander("⚙️ Configuration"):
        config = version['configuration']
        st.caption(f"**Random Seed:** {config.get('random_seed', 42)}")
        st.caption(f"**Embedding Model:** {config.get('embeddings', {}).get('model', 'N/A')}")
        st.caption(f"**Min Topic Size:** {config.get('topics', {}).get('min_topic_size', 'N/A')}")
        st.caption(f"**Similarity Threshold:** {config.get('clustering', {}).get('similarity_threshold', 'N/A')}")

        if st.button("View Full Config"):
            st.json(config)

    # Create new version button
    st.sidebar.markdown("---")
    if st.sidebar.button("➕ Create New Version"):
        st.session_state.show_create_version = True

    return version_id


def render_create_version_dialog():
    """Render dialog for creating a new version."""
    st.header("Create New Result Version")

    with st.form("create_version_form"):
        name = st.text_input("Version Name", placeholder="e.g., no-location-stopwords")
        description = st.text_area("Description (optional)", placeholder="What makes this version different?")

        # Configuration editor
        st.markdown("**Configuration (JSON)**")
        default_config = get_default_config()
        config_str = st.text_area(
            "Edit configuration",
            value=json.dumps(default_config, indent=2),
            height=400
        )

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            submit = st.form_submit_button("Create Version")
        with col2:
            cancel = st.form_submit_button("Cancel")

        if cancel:
            st.session_state.show_create_version = False
            st.rerun()

        if submit:
            if not name:
                st.error("Version name is required")
            else:
                try:
                    # Parse configuration
                    config = json.loads(config_str)

                    # Check if config already exists
                    existing = find_version_by_config(config)
                    if existing:
                        st.warning(f"A version with this configuration already exists: **{existing['name']}**")
                        st.info(f"Version ID: {existing['id']}")
                    else:
                        # Create version
                        version_id = create_version(name, description, config)
                        st.success(f"✅ Created version: {name}")
                        st.info(f"Version ID: {version_id}")
                        st.markdown("**Next step:** Run the pipeline")
                        st.code(f"python3 scripts/run_pipeline.py --version-id {version_id}")

                        # Hide dialog
                        st.session_state.show_create_version = False

                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON configuration: {e}")
                except Exception as e:
                    st.error(f"Error creating version: {e}")


def main():
    st.title("🇱🇰 Sri Lanka Media Bias Detector")
    st.markdown("Analyzing coverage patterns across Sri Lankan English newspapers")

    # Initialize session state
    if 'show_create_version' not in st.session_state:
        st.session_state.show_create_version = False

    # Render version selector in sidebar
    version_id = render_version_sidebar()

    # Show create version dialog if requested
    if st.session_state.get('show_create_version', False):
        render_create_version_dialog()
        return

    # Check if version is selected
    if not version_id:
        st.info("Please select or create a result version to view dashboard")
        return

    # Load data for selected version
    stats = load_overview_stats(version_id)

    # Overview metrics
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Articles", f"{stats['total_articles']:,}")
    with col2:
        st.metric("Topics Discovered", stats['total_topics'])
    with col3:
        st.metric("Event Clusters", f"{stats['total_clusters']:,}")
    with col4:
        st.metric("Multi-Source Events", f"{stats['multi_source_clusters']:,}")

    if stats['date_range']['min_date']:
        st.caption(f"Date range: {stats['date_range']['min_date']} to {stats['date_range']['max_date']}")

    st.divider()

    # Initialize session state for active tab
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Tabs for different views
    tab_names = ["📊 Coverage", "🏷️ Topics", "📰 Events"]

    # Create buttons to switch tabs
    cols = st.columns(3)
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
        render_topics_tab(version_id)
    elif st.session_state.active_tab == 2:
        render_events_tab(version_id)


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


def render_topics_tab(version_id):
    """Render topics analysis and source comparison tab."""
    st.subheader("Discovered Topics")

    topics = load_topics(version_id)
    if not topics:
        st.warning("No topics found for this version. Run topic discovery first.")
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
    st.subheader("Source Comparison")
    st.markdown("Compare how different sources prioritize topics")

    # Topic coverage comparison
    st.markdown("### Topic Focus by Source")
    st.markdown("What percentage of each source's coverage goes to each topic?")

    if topic_source_data:
        # Calculate percentages per source
        source_totals = ts_df.groupby('source_name')['count'].sum()

        # Get top 10 topics
        top_topic_names_comparison = [t['name'] for t in topics[:10]]

        comparison_data = []
        for source in SOURCE_NAMES.values():
            source_data = ts_df[ts_df['source_name'] == source]
            total = source_totals.get(source, 1)

            for topic in top_topic_names_comparison:
                topic_count = source_data[source_data['topic'] == topic]['count'].sum()
                comparison_data.append({
                    'Source': source,
                    'Topic': topic[:30],
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
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Selection bias indicator
        st.markdown("### Selection Bias Indicators")
        st.markdown("Topics where sources significantly differ in coverage")

        # Calculate variance in coverage percentage across sources
        variance_data = []
        for topic in top_topic_names_comparison:
            topic_data = comp_df[comp_df['Topic'] == topic[:30]]
            if len(topic_data) > 1:
                variance_data.append({
                    'Topic': topic[:30],
                    'Coverage Variance': topic_data['Percentage'].var(),
                    'Max Coverage': topic_data['Percentage'].max(),
                    'Min Coverage': topic_data['Percentage'].min(),
                    'Range': topic_data['Percentage'].max() - topic_data['Percentage'].min()
                })

        var_df = pd.DataFrame(variance_data).sort_values('Range', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Highest Variation (potential selection bias)**")
            st.dataframe(
                var_df.head(5)[['Topic', 'Range']].rename(columns={'Range': 'Coverage Gap (%)'}),
                use_container_width=True
            )
        with col2:
            st.markdown("**Most Consistent Coverage**")
            st.dataframe(
                var_df.tail(5)[['Topic', 'Range']].rename(columns={'Range': 'Coverage Gap (%)'}),
                use_container_width=True
            )

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
        st.info("BERTopic model not found. Run `python3 scripts/02_discover_topics.py` to generate the model.")


def render_events_tab(version_id):
    """Render events analysis tab."""
    st.subheader("Top Event Clusters")
    st.markdown("Events covered by multiple sources - useful for comparing coverage")

    events = load_top_events(version_id, 30)
    if not events:
        st.warning("No event clusters found for this version. Run clustering first.")
        return

    events_df = pd.DataFrame(events)

    # Filter to multi-source events
    multi_source_events = events_df[events_df['sources_count'] > 1]

    # Event selector
    event_options = {
        f"{e['cluster_name'][:60]}... ({e['article_count']} articles, {e['sources_count']} sources)": e['id']
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


if __name__ == "__main__":
    main()
