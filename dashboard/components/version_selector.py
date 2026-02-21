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
    get_default_ner_config,
    get_default_summarization_config,
    get_default_ditwah_claims_config,
    get_default_entity_stance_config
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
        # Basic info in columns
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**{version['name']}**")
            if version['description']:
                st.caption(version['description'])
            st.caption(f"Version ID: `{version_id}`")
        with col2:
            st.caption(f"Created: {version['created_at'].strftime('%Y-%m-%d %H:%M')}")

        # Pipeline status (compact)
        status = version['pipeline_status']
        status_items = []

        if analysis_type == 'word_frequency':
            status_items.append(f"{'✓' if status.get('word_frequency') else '○'} Word Frequency")
        elif analysis_type == 'summarization':
            status_items.append(f"{'✓' if status.get('summarization') else '○'} Summarization")
        elif analysis_type == 'ditwah_claims':
            status_items.append(f"{'✓' if status.get('ditwah_claims') else '○'} Ditwah Claims")
        elif analysis_type == 'entity_stance':
            status_items.append(f"{'✓' if status.get('entity_stance') else '○'} Entity Stance")
        elif analysis_type == 'topics':
            status_items.append(f"{'✓' if status.get('topics') else '○'} Topics")
        elif analysis_type == 'clustering':
            status_items.append(f"{'✓' if status.get('clustering') else '○'} Clustering")

        st.caption("Pipeline: " + " • ".join(status_items))

        # Configuration as JSON
        config = version['configuration']
        st.markdown("**Configuration:**")
        st.code(json.dumps(config, indent=2), language='json')

    return version_id


def render_create_version_button(analysis_type):
    """Render button to create a new version for a specific analysis type.

    Args:
        analysis_type: 'topics', 'clustering', 'word_frequency', 'ner', or 'summarization'
    """
    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    # Check if we should open the dialog
    dialog_key = f"create_{analysis_type}_dialog"
    should_open_dialog = st.button(f"➕ Create New {display_name} Version", key=f"create_{analysis_type}_btn")

    # Also open if version was just created (to show success message)
    if dialog_key in st.session_state and st.session_state[dialog_key].get("created"):
        should_open_dialog = True

    if should_open_dialog:
        render_create_version_dialog(analysis_type)


@st.dialog("Create New Version")
def render_create_version_dialog(analysis_type):
    """Render modal dialog for creating a new version.

    Args:
        analysis_type: 'topics', 'clustering', 'word_frequency', 'ner', or 'summarization'
    """
    # Initialize session state for this dialog
    dialog_key = f"create_{analysis_type}_dialog"
    if dialog_key not in st.session_state:
        st.session_state[dialog_key] = {"created": False, "version_id": None, "version_name": None}

    dialog_state = st.session_state[dialog_key]

    # Format analysis type for display
    display_name = analysis_type.replace('_', ' ').title()

    # If version was just created, show success message
    if dialog_state["created"]:
        st.success(f"✅ Successfully created {analysis_type} version: **{dialog_state['version_name']}**")

        st.markdown("---")

        # Version ID in a nice info box
        st.markdown("**Version ID:**")
        st.code(dialog_state['version_id'], language=None)

        st.markdown("---")

        # Pipeline instructions
        st.markdown("### Next Steps: Run the Pipeline")

        version_id = dialog_state['version_id']

        if analysis_type == 'word_frequency':
            st.markdown("**Step 1: Compute word frequencies**")
            st.code(f"python3 scripts/word_frequency/01_compute_word_frequency.py --version-id {version_id}", language="bash")

        elif analysis_type == 'ner':
            st.markdown("**Step 1: Extract named entities**")
            st.code(f"python3 scripts/ner/01_extract_entities.py --version-id {version_id}", language="bash")

        elif analysis_type == 'summarization':
            st.markdown("**Step 1: Generate summaries**")
            st.code(f"python3 scripts/summarization/01_generate_summaries.py --version-id {version_id}", language="bash")

        elif analysis_type == 'topics':
            st.markdown("**Run topic discovery:**")
            st.code(f"python3 scripts/topics/02_discover_topics.py --version-id {version_id}", language="bash")
            st.info("Embeddings are auto-generated if needed, or run separately:\n"
                    "`python3 scripts/embeddings/01_generate_embeddings.py --model <model>`")

        elif analysis_type == 'clustering':
            st.markdown("**Run event clustering:**")
            st.code(f"python3 scripts/clustering/02_cluster_events.py --version-id {version_id}", language="bash")
            st.info("Embeddings are auto-generated if needed, or run separately:\n"
                    "`python3 scripts/embeddings/01_generate_embeddings.py --model <model>`")

        elif analysis_type == 'entity_stance':
            st.markdown("**Step 1: Analyze entity stance**")
            st.code(f"python3 scripts/entity_stance/01_analyze_entity_stance.py --version-id {version_id}", language="bash")
            st.info("Requires a completed NER version. Set `ner_version_id` in the configuration.")

        st.markdown("---")
        st.info("💡 Close this dialog and the page will automatically refresh to show your new version")

        # Close button
        if st.button("Close", type="primary", use_container_width=True):
            # Reset dialog state
            st.session_state[dialog_key] = {"created": False, "version_id": None, "version_name": None}
            st.rerun()

        return

    # Otherwise, show the creation form
    st.markdown(f"**Analysis Type:** {display_name}")

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
    elif analysis_type == 'summarization':
        default_config = get_default_summarization_config()
    elif analysis_type == 'ditwah_claims':
        default_config = get_default_ditwah_claims_config()
    elif analysis_type == 'entity_stance':
        default_config = get_default_entity_stance_config()
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
        if st.button("Create Version", type="primary", use_container_width=True):
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
                        st.info(f"Version ID: `{existing['id']}`")
                    else:
                        # Create version
                        version_id = create_version(name, description, config, analysis_type=analysis_type)

                        # Update dialog state
                        st.session_state[dialog_key] = {
                            "created": True,
                            "version_id": version_id,
                            "version_name": name
                        }
                        st.rerun()

                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON configuration: {e}")
                except Exception as e:
                    st.error(f"❌ Error creating version: {e}")

    with col2:
        if st.button("Cancel", use_container_width=True):
            # Reset dialog state
            st.session_state[dialog_key] = {"created": False, "version_id": None, "version_name": None}
            st.rerun()
