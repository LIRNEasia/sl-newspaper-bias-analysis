-- Media Bias Analysis Schema Extensions
-- Run with: psql -h localhost -U username -d database -f schema.sql
-- Schema: media_bias

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Result versions for configuration-based analysis
CREATE TABLE IF NOT EXISTS media_bias.result_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    configuration JSONB NOT NULL,
    is_complete BOOLEAN DEFAULT false,
    pipeline_status JSONB DEFAULT '{"embeddings": false, "topics": false, "clustering": false}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Topics discovered via BERTopic
CREATE TABLE IF NOT EXISTS media_bias.topics (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER NOT NULL,
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    parent_topic_id INTEGER REFERENCES media_bias.topics(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    keywords TEXT[],
    article_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(topic_id, result_version_id)
);

-- Article embeddings
CREATE TABLE IF NOT EXISTS media_bias.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    embedding VECTOR,
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(article_id, result_version_id)
);

-- Article-level analysis results
CREATE TABLE IF NOT EXISTS media_bias.article_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    primary_topic_id INTEGER REFERENCES media_bias.topics(id),
    topic_confidence FLOAT,
    article_type VARCHAR(50),
    article_type_confidence FLOAT,
    overall_tone FLOAT,
    headline_tone FLOAT,
    tone_reasoning TEXT,
    llm_provider VARCHAR(50),
    llm_model VARCHAR(100),
    processed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(article_id, result_version_id)
);

-- Event clusters
CREATE TABLE IF NOT EXISTS media_bias.event_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    cluster_name VARCHAR(255),
    cluster_description TEXT,
    representative_article_id UUID REFERENCES media_bias.news_articles(id),
    article_count INTEGER DEFAULT 0,
    sources_count INTEGER DEFAULT 0,
    date_start DATE,
    date_end DATE,
    primary_topic_id INTEGER REFERENCES media_bias.topics(id),
    centroid_embedding VECTOR,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Article to cluster mapping
CREATE TABLE IF NOT EXISTS media_bias.article_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    cluster_id UUID NOT NULL REFERENCES media_bias.event_clusters(id),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    similarity_score FLOAT,
    UNIQUE(article_id, cluster_id, result_version_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_article ON media_bias.embeddings(article_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_version ON media_bias.embeddings(result_version_id);
CREATE INDEX IF NOT EXISTS idx_topics_version ON media_bias.topics(result_version_id);
CREATE INDEX IF NOT EXISTS idx_article_analysis_article ON media_bias.article_analysis(article_id);
CREATE INDEX IF NOT EXISTS idx_article_analysis_version ON media_bias.article_analysis(result_version_id);
CREATE INDEX IF NOT EXISTS idx_article_analysis_topic ON media_bias.article_analysis(primary_topic_id);
CREATE INDEX IF NOT EXISTS idx_article_clusters_article ON media_bias.article_clusters(article_id);
CREATE INDEX IF NOT EXISTS idx_article_clusters_cluster ON media_bias.article_clusters(cluster_id);
CREATE INDEX IF NOT EXISTS idx_article_clusters_version ON media_bias.article_clusters(result_version_id);
CREATE INDEX IF NOT EXISTS idx_event_clusters_version ON media_bias.event_clusters(result_version_id);

-- HNSW index for similarity search (if pgvector supports it)
-- CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON media_bias.embeddings
--     USING hnsw (embedding vector_cosine_ops);
