"""Version management for result configurations."""

import json
import tarfile
import tempfile
import io
from pathlib import Path
from typing import Dict, List, Optional, Any
from src.db import Database, load_config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration from config.yaml.

    Returns:
        Dictionary with default configuration for embeddings, topics, and clustering.
    """
    config = load_config()

    return {
        "embeddings": {
            "provider": config["embeddings"]["provider"],
            "model": config["embeddings"]["model"]
        },
        "topics": {
            "min_topic_size": config["topics"]["min_topic_size"],
            "diversity": config["topics"].get("diversity", 0.5),
            "nr_topics": None,
            "stop_words": ["sri", "lanka", "lankan"],
            "umap": {
                "n_neighbors": 15,
                "n_components": 5,
                "min_dist": 0.0,
                "metric": "cosine"
            },
            "hdbscan": {
                "min_cluster_size": config["topics"]["min_topic_size"],
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "core_dist_n_jobs": 1
            },
            "vectorizer": {
                "ngram_range": [1, 3],
                "min_df": 5
            }
        },
        "clustering": {
            "similarity_threshold": config["clustering"]["similarity_threshold"],
            "time_window_days": config["clustering"]["time_window_days"],
            "min_cluster_size": config["clustering"]["min_cluster_size"]
        }
    }


def get_default_topic_config() -> Dict[str, Any]:
    """
    Get default configuration for topic analysis.

    Returns:
        Dictionary with configuration for embeddings and topics only.
    """
    config = load_config()

    return {
        "embeddings": {
            "provider": config["embeddings"]["provider"],
            "model": config["embeddings"]["model"]
        },
        "topics": {
            "min_topic_size": config["topics"]["min_topic_size"],
            "diversity": config["topics"].get("diversity", 0.5),
            "nr_topics": None,
            "stop_words": ["sri", "lanka", "lankan"],
            "filter_ner_entities": config["topics"].get("filter_ner_entities", False),
            "ner_version_id": None,
            "ner_entity_types": config["topics"].get("ner_entity_types"),
            "umap": {
                "n_neighbors": 15,
                "n_components": 5,
                "min_dist": 0.0,
                "metric": "cosine"
            },
            "hdbscan": {
                "min_cluster_size": config["topics"]["min_topic_size"],
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "core_dist_n_jobs": 1
            },
            "vectorizer": {
                "ngram_range": [1, 3],
                "min_df": 5
            }
        }
    }


def get_default_clustering_config() -> Dict[str, Any]:
    """
    Get default configuration for clustering analysis.

    Returns:
        Dictionary with configuration for embeddings and clustering only.
    """
    config = load_config()

    return {
        "embeddings": {
            "provider": config["embeddings"]["provider"],
            "model": config["embeddings"]["model"]
        },
        "clustering": {
            "similarity_threshold": config["clustering"]["similarity_threshold"],
            "time_window_days": config["clustering"]["time_window_days"],
            "min_cluster_size": config["clustering"]["min_cluster_size"]
        }
    }


def get_default_word_frequency_config() -> Dict[str, Any]:
    """
    Get default configuration for word frequency analysis.

    Returns:
        Dictionary with configuration for word frequency only.
    """
    config = load_config()

    return {
        "word_frequency": {
            "ranking_method": config["word_frequency"]["ranking_method"],
            "tfidf_scope": config["word_frequency"]["tfidf_scope"],
            "top_n_words": config["word_frequency"]["top_n_words"],
            "min_word_length": config["word_frequency"].get("min_word_length", 3),
            "custom_stopwords": config["word_frequency"].get("custom_stopwords", [])
        }
    }


def get_default_ner_config() -> Dict[str, Any]:
    """
    Get default configuration for NER analysis.

    Returns:
        Dictionary with configuration for NER only.
    """
    config = load_config()

    return {
        "ner": {
            "provider": config["ner"]["provider"],
            "model": config["ner"]["model"],
            "batch_size": config["ner"].get("batch_size", 32),
            "confidence_threshold": config["ner"].get("confidence_threshold", 0.5),
            "entity_types": config["ner"].get("entity_types", []),
            "custom_entity_types": config["ner"].get("custom_entity_types", [])
        }
    }


def get_default_summarization_config() -> Dict[str, Any]:
    """
    Get default configuration for summarization analysis.

    Returns:
        Dictionary with configuration for summarization only.
    """
    config = load_config()

    return {
        "summarization": {
            "method": config["summarization"]["method"],
            "summary_length": config["summarization"]["summary_length"],
            "short_sentences": config["summarization"].get("short_sentences", 3),
            "short_words": config["summarization"].get("short_words", 50),
            "medium_sentences": config["summarization"].get("medium_sentences", 5),
            "medium_words": config["summarization"].get("medium_words", 100),
            "long_sentences": config["summarization"].get("long_sentences", 8),
            "long_words": config["summarization"].get("long_words", 150),
            "max_input_length": config["summarization"].get("max_input_length", 1024),
            "chunk_long_articles": config["summarization"].get("chunk_long_articles", True),
            "llm_model": config["summarization"].get("llm_model", "claude-sonnet-4-20250514"),
            "llm_temperature": config["summarization"].get("llm_temperature", 0.0)
        }
    }


def get_default_ditwah_config() -> Dict[str, Any]:
    """
    Get default configuration for Ditwah analysis.

    Returns:
        Dictionary with configuration for Ditwah hypothesis stance analysis.
    """
    config = load_config()

    return {
        "random_seed": 42,
        "ditwah": {
            "hypotheses": [
                {
                    "key": "h1",
                    "statement": "The government's disaster response to Cyclone Ditwah was adequate and timely",
                    "category": "government_response"
                },
                {
                    "key": "h2",
                    "statement": "International aid was crucial in addressing the cyclone's aftermath",
                    "category": "international_aid"
                },
                {
                    "key": "h3",
                    "statement": "The economic impact of Cyclone Ditwah will have long-term consequences for Sri Lanka",
                    "category": "economic_impact"
                },
                {
                    "key": "h4",
                    "statement": "The cyclone's impact was exacerbated by inadequate infrastructure and preparedness",
                    "category": "preparedness"
                },
                {
                    "key": "h5",
                    "statement": "Climate change played a significant role in the intensity of Cyclone Ditwah",
                    "category": "climate_change"
                }
            ],
            "llm": {
                "provider": config["llm"].get("provider", "local"),
                "model": config["llm"].get("model", "llama3.1:70b"),
                "base_url": "http://localhost:11434",
                "temperature": 0.0,
                "max_tokens": 1000
            },
            "batch_size": 5
        }
    }


def get_default_ditwah_claims_config() -> Dict[str, Any]:
    """
    Get default configuration for Ditwah claims analysis.

    Returns:
        Dictionary with configuration for automatic claim generation + sentiment + stance.
    """
    config = load_config()

    # Use ditwah_claims config if available, otherwise fall back to defaults
    ditwah_config = config.get("ditwah_claims", {})

    return {
        "random_seed": 42,
        "generation": ditwah_config.get("generation", {
            "min_articles": 5,  # Minimum articles per claim
            "categories": [
                "government_response",
                "humanitarian_aid",
                "infrastructure_damage",
                "economic_impact",
                "international_response",
                "casualties_and_displacement"
            ]
        }),
        "llm": ditwah_config.get("llm", {
            "provider": config["llm"].get("provider", "mistral"),
            "model": config["llm"].get("model", "mistral-large-latest"),
            "temperature": 0.3,
            "max_tokens": 4000
        }),
        "sentiment": ditwah_config.get("sentiment", {
            "primary_model": "roberta"
        }),
        "stance": ditwah_config.get("stance", {
            "batch_size": 5,
            "temperature": 0.0
        })
    }


def get_default_multi_doc_summarization_config() -> Dict[str, Any]:
    """
    Get default configuration for multi-document summarization analysis.

    Returns:
        Dictionary with configuration for multi-doc summarization only.
    """
    config = load_config()

    # Default to Gemini for multi-doc summarization
    default_method = "gemini"
    default_model = "gemini-2.0-flash"

    # Check if user has specific preferences in config
    if "multi_doc_summarization" in config:
        mds_config = config["multi_doc_summarization"]
        default_method = mds_config.get("method", default_method)
        default_model = mds_config.get("llm_model", default_model)
    else:
        # Fallback to checking provider-specific configs
        if "gemini" in config and config["gemini"].get("model"):
            default_model = config["gemini"]["model"]
        elif "openai" in config and config["openai"].get("model"):
            default_method = "openai"
            default_model = config["openai"].get("model", "gpt-4o")

    return {
        "multi_doc_summarization": {
            "method": default_method,  # 'gemini' or 'openai'
            "llm_model": default_model,
            "temperature": 0.0,
            "summary_length": "medium",
            "max_articles": 10 if default_method == "openai" else 50,  # Token limit-aware sampling
            "short_sentences": config["summarization"].get("short_sentences", 5),
            "short_words": config["summarization"].get("short_words", 80),
            "medium_sentences": config["summarization"].get("medium_sentences", 8),
            "medium_words": config["summarization"].get("medium_words", 150),
            "long_sentences": config["summarization"].get("long_sentences", 12),
            "long_words": config["summarization"].get("long_words", 200)
        }
    }


def get_default_entity_stance_config() -> Dict[str, Any]:
    """Get default configuration for entity stance analysis.

    Returns:
        Dictionary with entity stance configuration.
    """
    return {
        "entity_stance": {
            "chunk_size": 5,
            "neutral_threshold": 0.2,
            "min_confidence": 0.3,
            "entity_types": ["PERSON", "ORG", "GPE", "NORP", "EVENT", "LAW"],
            "model": "cross-encoder/nli-deberta-v3-base"
        },
        "ner_version_id": None  # Must be set to an existing NER version
    }


def create_version(
    name: str,
    description: str = "",
    configuration: Optional[Dict[str, Any]] = None,
    analysis_type: str = 'combined'
) -> str:
    """
    Create a new result version.

    Args:
        name: Unique name for this version (can be same across different analysis types)
        description: Optional description of this version
        configuration: Configuration dictionary (uses default if not provided)
        analysis_type: Type of analysis ('topics', 'clustering', or 'combined')

    Returns:
        UUID of the created version

    Raises:
        ValueError: If version name already exists for the same analysis type
    """
    valid_types = ['topics', 'clustering', 'word_frequency', 'ner', 'summarization', 'multi_doc_summarization', 'ditwah', 'ditwah_claims', 'entity_stance', 'combined']
    if analysis_type not in valid_types:
        raise ValueError(f"Invalid analysis_type: {analysis_type}. Must be one of {valid_types}")

    if configuration is None:
        configuration = get_default_config()

    with Database() as db:
        schema = db.config["schema"]

        # Check if name already exists for this analysis type
        with db.cursor() as cur:
            cur.execute(
                f"SELECT id FROM {schema}.result_versions WHERE name = %s AND analysis_type = %s",
                (name, analysis_type)
            )
            if cur.fetchone():
                raise ValueError(f"Version with name '{name}' and analysis_type '{analysis_type}' already exists")

        # Insert new version
        with db.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {schema}.result_versions
                (name, description, configuration, analysis_type)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, description, json.dumps(configuration), analysis_type)
            )
            result = cur.fetchone()
            return str(result["id"])


def get_version(version_id: str) -> Optional[Dict[str, Any]]:
    """
    Get version metadata by ID.

    Args:
        version_id: UUID of the version

    Returns:
        Dictionary with version metadata or None if not found
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, name, description, configuration, analysis_type, is_complete,
                       pipeline_status, created_at, updated_at
                FROM {schema}.result_versions
                WHERE id = %s
                """,
                (version_id,)
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "configuration": row["configuration"],
                    "analysis_type": row["analysis_type"],
                    "is_complete": row["is_complete"],
                    "pipeline_status": row["pipeline_status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None


def get_version_by_name(name: str, analysis_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get version metadata by name.

    Args:
        name: Name of the version
        analysis_type: Optional filter by analysis type ('topics', 'clustering', 'combined')

    Returns:
        Dictionary with version metadata or None if not found
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if analysis_type:
                cur.execute(
                    f"""
                    SELECT id, name, description, configuration, analysis_type, is_complete,
                           pipeline_status, created_at, updated_at
                    FROM {schema}.result_versions
                    WHERE name = %s AND analysis_type = %s
                    """,
                    (name, analysis_type)
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, name, description, configuration, analysis_type, is_complete,
                           pipeline_status, created_at, updated_at
                    FROM {schema}.result_versions
                    WHERE name = %s
                    """,
                    (name,)
                )
            row = cur.fetchone()
            if row:
                return {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "configuration": row["configuration"],
                    "analysis_type": row["analysis_type"],
                    "is_complete": row["is_complete"],
                    "pipeline_status": row["pipeline_status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None


def list_versions(analysis_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all versions, optionally filtered by analysis type.

    Args:
        analysis_type: Optional filter by analysis type ('topics', 'clustering', 'combined')

    Returns:
        List of dictionaries with version metadata
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if analysis_type:
                cur.execute(
                    f"""
                    SELECT id, name, description, configuration, analysis_type, is_complete,
                           pipeline_status, created_at, updated_at
                    FROM {schema}.result_versions
                    WHERE analysis_type = %s
                    ORDER BY created_at DESC
                    """,
                    (analysis_type,)
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, name, description, configuration, analysis_type, is_complete,
                           pipeline_status, created_at, updated_at
                    FROM {schema}.result_versions
                    ORDER BY created_at DESC
                    """
                )
            rows = cur.fetchall()
            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "configuration": row["configuration"],
                    "analysis_type": row["analysis_type"],
                    "is_complete": row["is_complete"],
                    "pipeline_status": row["pipeline_status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                for row in rows
            ]


def find_version_by_config(configuration: Dict[str, Any], analysis_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Find a version with matching configuration.

    Args:
        configuration: Configuration dictionary to match
        analysis_type: Optional filter by analysis type ('topics', 'clustering', 'combined')

    Returns:
        Dictionary with version metadata or None if no match found
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if analysis_type:
                cur.execute(
                    f"""
                    SELECT id, name, description, configuration, analysis_type, is_complete,
                           pipeline_status, created_at, updated_at
                    FROM {schema}.result_versions
                    WHERE configuration = %s::jsonb AND analysis_type = %s
                    """,
                    (json.dumps(configuration), analysis_type)
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, name, description, configuration, analysis_type, is_complete,
                           pipeline_status, created_at, updated_at
                    FROM {schema}.result_versions
                    WHERE configuration = %s::jsonb
                    """,
                    (json.dumps(configuration),)
                )
            row = cur.fetchone()
            if row:
                return {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "configuration": row["configuration"],
                    "analysis_type": row["analysis_type"],
                    "is_complete": row["is_complete"],
                    "pipeline_status": row["pipeline_status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
            return None


def update_pipeline_status(
    version_id: str,
    step: str,
    complete: bool
) -> None:
    """
    Update pipeline completion status for a specific step.

    Args:
        version_id: UUID of the version
        step: Pipeline step name ('embeddings', 'topics', 'clustering', 'word_frequency', 'ner', 'summarization', 'ditwah', or 'ditwah_claims')
        complete: Whether the step is complete
    """
    valid_steps = ['topics', 'clustering', 'word_frequency', 'ner', 'summarization', 'ditwah', 'ditwah_claims', 'entity_stance']
    if step not in valid_steps:
        raise ValueError(f"Invalid step: {step}. Must be one of {valid_steps}")

    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Update the specific step status
            cur.execute(
                f"""
                UPDATE {schema}.result_versions
                SET pipeline_status = jsonb_set(
                    pipeline_status,
                    %s,
                    %s
                ),
                updated_at = NOW()
                WHERE id = %s
                """,
                (f'{{{step}}}', json.dumps(complete), version_id)
            )

            cur.execute(
                f"""
                UPDATE {schema}.result_versions
                SET is_complete = (
                    CASE analysis_type
                        WHEN 'topics' THEN
                            (pipeline_status->>'topics')::boolean
                        WHEN 'clustering' THEN
                            (pipeline_status->>'clustering')::boolean
                        WHEN 'word_frequency' THEN
                            (pipeline_status->>'word_frequency')::boolean
                        WHEN 'ner' THEN
                            (pipeline_status->>'ner')::boolean
                        WHEN 'summarization' THEN
                            (pipeline_status->>'summarization')::boolean
                        WHEN 'ditwah' THEN
                            (pipeline_status->>'ditwah')::boolean
                        WHEN 'ditwah_claims' THEN
                            (pipeline_status->>'ditwah_claims')::boolean
                        WHEN 'entity_stance' THEN
                            (pipeline_status->>'entity_stance')::boolean
                        WHEN 'combined' THEN
                            (pipeline_status->>'embeddings')::boolean AND
                            (pipeline_status->>'topics')::boolean AND
                            (pipeline_status->>'clustering')::boolean
                        ELSE FALSE
                    END
                )
                WHERE id = %s
                """,
                (version_id,)
            )


def get_version_config(version_id: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific version.

    Args:
        version_id: UUID of the version

    Returns:
        Configuration dictionary or None if version not found
    """
    version = get_version(version_id)
    return version["configuration"] if version else None


def delete_version(version_id: str) -> bool:
    """
    Delete a version and all its associated results.

    Args:
        version_id: UUID of the version to delete

    Returns:
        True if deleted, False if version not found

    Note:
        This will cascade delete all associated embeddings, topics,
        article_analysis, event_clusters, and article_clusters.
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"DELETE FROM {schema}.result_versions WHERE id = %s",
                (version_id,)
            )
            return cur.rowcount > 0


def get_version_statistics(version_id: str) -> Dict[str, int]:
    """
    Get statistics for a version (counts of embeddings, topics, clusters, etc.).

    Args:
        version_id: UUID of the version

    Returns:
        Dictionary with counts for various entities
    """
    with Database() as db:
        schema = db.config["schema"]
        stats = {}

        # Count embeddings (shared across versions, count by model from version config)
        version = get_version(version_id)
        if version and version.get("configuration", {}).get("embeddings", {}).get("model"):
            embedding_model = version["configuration"]["embeddings"]["model"]
            stats["embeddings"] = db.get_embedding_count(embedding_model=embedding_model)
        else:
            stats["embeddings"] = 0

        # Count topics
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.topics WHERE result_version_id = %s",
                (version_id,)
            )
            stats["topics"] = cur.fetchone()["count"]

        # Count article analyses
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.article_analysis WHERE result_version_id = %s",
                (version_id,)
            )
            stats["article_analysis"] = cur.fetchone()["count"]

        # Count event clusters
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE result_version_id = %s",
                (version_id,)
            )
            stats["event_clusters"] = cur.fetchone()["count"]

        # Count article-cluster mappings
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.article_clusters WHERE result_version_id = %s",
                (version_id,)
            )
            stats["article_clusters"] = cur.fetchone()["count"]

        # Count article summaries
        with db.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as count FROM {schema}.article_summaries WHERE result_version_id = %s",
                (version_id,)
            )
            stats["article_summaries"] = cur.fetchone()["count"]

        return stats


def save_model_to_version(version_id: str, model_directory: str) -> None:
    """
    Save a BERTopic model directory to the database as a compressed archive.

    This enables team collaboration by storing models in the shared database
    rather than local filesystems.

    Args:
        version_id: UUID of the version
        model_directory: Path to the BERTopic model directory

    Raises:
        FileNotFoundError: If model_directory doesn't exist
        Exception: If database operation fails
    """
    model_path = Path(model_directory)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_directory}")

    # Create tar.gz archive in memory
    tar_buffer = io.BytesIO()

    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        # Add all files in the model directory
        for item in model_path.iterdir():
            tar.add(item, arcname=item.name)

    # Get the compressed bytes
    model_bytes = tar_buffer.getvalue()

    # Store in database
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {schema}.result_versions
                SET model_data = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (model_bytes, version_id)
            )

    # Report size
    size_mb = len(model_bytes) / (1024 * 1024)
    print(f"  Model compressed to {size_mb:.2f} MB and saved to database")


def get_model_from_version(version_id: str, extract_to: str) -> Optional[str]:
    """
    Retrieve a BERTopic model from the database and extract to filesystem.

    Args:
        version_id: UUID of the version
        extract_to: Directory to extract model to

    Returns:
        Path to extracted model directory, or None if no model stored in database

    Raises:
        Exception: If extraction or database operation fails
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"SELECT model_data FROM {schema}.result_versions WHERE id = %s",
                (version_id,)
            )
            row = cur.fetchone()

            if not row or not row["model_data"]:
                return None

            model_bytes = row["model_data"]

    # Extract tar.gz archive
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)

    tar_buffer = io.BytesIO(model_bytes)

    with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
        tar.extractall(path=extract_path)

    return str(extract_path)


def delete_version_interactive(version_id: str) -> bool:
    """
    Interactively delete a version with confirmation prompt.

    Shows version details and statistics before asking for confirmation.

    Args:
        version_id: UUID of the version to delete

    Returns:
        True if deleted, False if cancelled or version not found
    """
    # Get version info
    version = get_version(version_id)
    if not version:
        print(f"❌ Version not found: {version_id}")
        return False

    # Get statistics
    stats = get_version_statistics(version_id)

    # Display version info
    print("\n" + "="*60)
    print("VERSION DELETION PREVIEW")
    print("="*60)
    print(f"\nVersion ID: {version['id']}")
    print(f"Name: {version['name']}")
    print(f"Analysis Type: {version['analysis_type']}")
    print(f"Description: {version['description'] or '(none)'}")
    print(f"Complete: {'Yes' if version['is_complete'] else 'No'}")
    print(f"Created: {version['created_at']}")

    # Display what will be deleted
    print(f"\n{'='*60}")
    print("DATA TO BE DELETED:")
    print("="*60)
    print(f"  Embeddings: {stats['embeddings']:,}")
    print(f"  Topics: {stats['topics']:,}")
    print(f"  Article Analyses: {stats['article_analysis']:,}")
    print(f"  Event Clusters: {stats['event_clusters']:,}")
    print(f"  Article-Cluster Mappings: {stats['article_clusters']:,}")

    total_records = sum(stats.values())
    print(f"\n  TOTAL RECORDS: {total_records:,}")
    print("="*60)

    # Warning message
    print("\n⚠️  WARNING: This action cannot be undone!")
    print("⚠️  All analysis results for this version will be permanently deleted.")
    print("⚠️  Original articles in news_articles table will NOT be affected.")

    # Confirmation prompt
    print(f"\n{'='*60}")
    confirmation = input(f"Type the version name '{version['name']}' to confirm deletion: ")

    if confirmation != version['name']:
        print("\n❌ Deletion cancelled - confirmation text did not match.")
        return False

    # Final confirmation
    final = input("\nAre you absolutely sure? Type 'DELETE' to proceed: ")

    if final != 'DELETE':
        print("\n❌ Deletion cancelled.")
        return False

    # Perform deletion
    print(f"\n🗑️  Deleting version '{version['name']}'...")
    success = delete_version(version_id)

    if success:
        print(f"✅ Version deleted successfully!")
        print(f"   Removed {total_records:,} records from the database.")
        return True
    else:
        print("❌ Deletion failed - version not found.")
        return False
