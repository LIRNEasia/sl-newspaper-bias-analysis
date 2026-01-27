"""Version selector components for the dashboard."""

import json
import streamlit as st

from src.versions import (
    list_versions,
    get_version,
    create_version,
    find_version_by_config,
    get_default_topic_config,
    get_default_clustering_config,
    get_default_word_frequency_config,
    get_default_ner_config
)


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
    with st.expander("Version Details"):
        st.markdown(f"**Name:** {version['name']}")
        if version['description']:
            st.markdown(f"**Description:** {version['description']}")
        st.markdown(f"**Created:** {version['created_at'].strftime('%Y-%m-%d %H:%M')}")

        # Pipeline status
        status = version['pipeline_status']
        st.markdown("**Pipeline Status:**")

        if analysis_type == 'word_frequency':
            # Word frequency only has one pipeline step
            st.caption(f"{'[OK]' if status.get('word_frequency') else '[ ]'} Word Frequency")
        else:
            # Topics and clustering have embeddings + analysis
            cols = st.columns(2)
            with cols[0]:
                st.caption(f"{'[OK]' if status.get('embeddings') else '[ ]'} Embeddings")
            with cols[1]:
                if analysis_type == 'topics':
                    st.caption(f"{'[OK]' if status.get('topics') else '[ ]'} Topics")
                else:
                    st.caption(f"{'[OK]' if status.get('clustering') else '[ ]'} Clustering")

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
        analysis_type: 'topics', 'clustering', 'word_frequency', or 'ner'
    """
    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    if st.button(f"Create New {display_name} Version", key=f"create_{analysis_type}_btn"):
        st.session_state[f'show_create_{analysis_type}'] = True

    # Show create dialog if requested
    if st.session_state.get(f'show_create_{analysis_type}', False):
        render_create_version_form(analysis_type)


def render_create_version_form(analysis_type):
    """Render form for creating a new version.

    Args:
        analysis_type: 'topics', 'clustering', 'word_frequency', or 'ner'
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
        elif analysis_type == 'ner':
            default_config = get_default_ner_config()
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
                        st.success(f"Created {analysis_type} version: {name}")
                        st.info(f"Version ID: {version_id}")

                        # Show pipeline instructions
                        st.markdown("**Next steps:** Run the pipeline")
                        if analysis_type == 'word_frequency':
                            st.code(f"""# Compute word frequencies
python3 scripts/word_frequency/01_compute_word_frequency.py --version-id {version_id}""")
                        elif analysis_type == 'ner':
                            st.code(f"""# Extract named entities
python3 scripts/ner/01_extract_entities.py --version-id {version_id}""")
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
