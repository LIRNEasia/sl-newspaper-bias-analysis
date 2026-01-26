"""Event clustering to group related articles."""

import random
import numpy as np
from typing import Dict
import uuid

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .db import get_db, load_config


def cluster_articles(
    result_version_id: str,
    similarity_threshold: float = 0.8,
    time_window_days: int = 7,
    min_cluster_size: int = 2,
    random_seed: int = 42
) -> Dict:
    """
    Cluster articles into events based on embedding similarity for a specific version.

    Args:
        result_version_id: UUID of the result version
        similarity_threshold: Minimum cosine similarity to cluster together
        time_window_days: Only cluster articles within this time window
        min_cluster_size: Minimum articles per cluster
        random_seed: Random seed for reproducibility

    Returns:
        Summary of clustering results
    """
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Loading articles with embeddings for version {result_version_id}...")
    with get_db() as db:
        data = db.get_all_embeddings(result_version_id=result_version_id)

    print(f"Loaded {len(data)} articles")

    # Prepare data
    article_ids = [str(d['article_id']) for d in data]
    embeddings = np.array([d['embedding'] for d in data])
    dates = [d['date_posted'] for d in data]
    sources = [d['source_id'] for d in data]
    titles = [d['title'] for d in data]

    print("Computing similarity matrix...")
    # Compute cosine similarity (this can be memory intensive for large datasets)
    # For 8k articles, this creates an 8k x 8k matrix (~500MB)
    similarity_matrix = cosine_similarity(embeddings)

    print("Finding clusters...")
    # Find clusters using a simple approach:
    # 1. For each article, find all similar articles within time window
    # 2. Merge overlapping groups

    clusters = []
    clustered = set()

    for i in tqdm(range(len(article_ids)), desc="Clustering"):
        if i in clustered:
            continue

        # Find all similar articles
        similar_indices = []
        for j in range(len(article_ids)):
            if i == j:
                continue

            # Check similarity threshold
            if similarity_matrix[i, j] < similarity_threshold:
                continue

            # Check time window
            if dates[i] and dates[j]:
                time_diff = abs((dates[i] - dates[j]).days)
                if time_diff > time_window_days:
                    continue

            similar_indices.append(j)

        # Create cluster if we have enough similar articles
        if len(similar_indices) >= min_cluster_size - 1:  # -1 because we include article i
            cluster_indices = [i] + similar_indices

            # Skip if all articles are already clustered
            new_articles = [idx for idx in cluster_indices if idx not in clustered]
            if len(new_articles) < min_cluster_size:
                continue

            # Mark as clustered
            for idx in cluster_indices:
                clustered.add(idx)

            # Calculate cluster info
            cluster_embeddings = embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)

            cluster_dates = [dates[idx] for idx in cluster_indices if dates[idx]]
            cluster_sources = list(set([sources[idx] for idx in cluster_indices]))

            # Find representative article (closest to centroid)
            distances = [np.linalg.norm(embeddings[idx] - centroid) for idx in cluster_indices]
            rep_idx = cluster_indices[np.argmin(distances)]

            clusters.append({
                "id": str(uuid.uuid4()),
                "article_indices": cluster_indices,
                "article_ids": [article_ids[idx] for idx in cluster_indices],
                "representative_article_id": article_ids[rep_idx],
                "representative_title": titles[rep_idx],
                "centroid": centroid.tolist(),
                "article_count": len(cluster_indices),
                "sources": cluster_sources,
                "sources_count": len(cluster_sources),
                "date_start": min(cluster_dates) if cluster_dates else None,
                "date_end": max(cluster_dates) if cluster_dates else None,
            })

    print(f"Found {len(clusters)} event clusters")

    # Store clusters in database
    print("Saving clusters to database...")
    with get_db() as db:
        for cluster in tqdm(clusters, desc="Saving"):
            db.store_event_clusters([{
                "id": cluster["id"],
                "name": cluster["representative_title"][:200],  # Use title as name
                "description": f"Event cluster with {cluster['article_count']} articles from {cluster['sources_count']} sources",
                "representative_article_id": cluster["representative_article_id"],
                "article_count": cluster["article_count"],
                "sources_count": cluster["sources_count"],
                "date_start": cluster["date_start"],
                "date_end": cluster["date_end"],
                "centroid": cluster["centroid"],
                "articles": [
                    {"article_id": aid, "similarity": 1.0}
                    for aid in cluster["article_ids"]
                ]
            }], result_version_id)

    # Summary
    total_clustered = len(clustered)
    multi_source_clusters = sum(1 for c in clusters if c["sources_count"] > 1)

    summary = {
        "total_articles": len(article_ids),
        "articles_clustered": total_clustered,
        "articles_unclustered": len(article_ids) - total_clustered,
        "total_clusters": len(clusters),
        "multi_source_clusters": multi_source_clusters,
        "avg_cluster_size": np.mean([c["article_count"] for c in clusters]) if clusters else 0,
    }

    print("\nClustering Complete:")
    print(f"  Total articles: {summary['total_articles']}")
    print(f"  Articles in clusters: {summary['articles_clustered']}")
    print(f"  Total clusters: {summary['total_clusters']}")
    print(f"  Multi-source clusters: {summary['multi_source_clusters']}")
    print(f"  Avg cluster size: {summary['avg_cluster_size']:.1f}")

    return summary


def get_cluster_stats() -> Dict:
    """Get statistics about event clusters."""
    with get_db() as db:
        with db.cursor() as cur:
            schema = db.config["schema"]

            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.event_clusters")
            total_clusters = cur.fetchone()["count"]

            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.article_clusters")
            total_mappings = cur.fetchone()["count"]

            cur.execute(f"""
                SELECT ec.cluster_name, ec.article_count, ec.sources_count,
                       ec.date_start, ec.date_end
                FROM {schema}.event_clusters ec
                ORDER BY ec.article_count DESC
                LIMIT 10
            """)
            top_clusters = cur.fetchall()

    return {
        "total_clusters": total_clusters,
        "total_article_mappings": total_mappings,
        "top_clusters": top_clusters
    }


if __name__ == "__main__":
    print("Please use scripts/03_cluster_events.py instead.")
    print("Usage: python3 scripts/03_cluster_events.py --version-id <uuid>")
