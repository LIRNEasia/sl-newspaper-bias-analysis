-- Media Bias Analysis Schema Extensions
-- Run with: psql -h localhost -U your_db_user -d your_database -f schema.sql
-- Note: Replace 'media_bias' below with your actual schema name

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Result versions for configuration-based analysis
CREATE TABLE IF NOT EXISTS media_bias.result_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    configuration JSONB NOT NULL,
    analysis_type VARCHAR(50) NOT NULL DEFAULT 'combined',
    is_complete BOOLEAN DEFAULT false,
    pipeline_status JSONB DEFAULT '{"embeddings": false, "topics": false, "clustering": false, "word_frequency": false, "ner": false, "summarization": false}'::jsonb,
    model_data BYTEA,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT result_versions_name_analysis_type_key UNIQUE (name, analysis_type)
);

COMMENT ON COLUMN media_bias.result_versions.analysis_type IS
  'Type of analysis: ''topics'' for topic discovery, ''clustering'' for event clustering, ''combined'' for legacy versions';

COMMENT ON COLUMN media_bias.result_versions.model_data IS
  'Compressed tar.gz archive of BERTopic model directory (for visualizations). NULL if model not stored in database.';

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

-- Word frequency analysis results
CREATE TABLE IF NOT EXISTS media_bias.word_frequencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    source_id VARCHAR(50) NOT NULL,
    word VARCHAR(255) NOT NULL,
    frequency INTEGER NOT NULL,
    tfidf_score FLOAT,
    rank INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(result_version_id, source_id, word)
);

-- Named entities discovered in articles
CREATE TABLE IF NOT EXISTS media_bias.named_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    entity_text VARCHAR(500) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    confidence FLOAT,
    context TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(article_id, result_version_id, entity_text, entity_type, start_char)
);

-- Entity statistics per source (aggregated view)
CREATE TABLE IF NOT EXISTS media_bias.entity_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    entity_text VARCHAR(500) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    source_id VARCHAR(50) NOT NULL,
    mention_count INTEGER DEFAULT 0,
    article_count INTEGER DEFAULT 0,
    avg_confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(result_version_id, entity_text, entity_type, source_id)
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
CREATE INDEX IF NOT EXISTS idx_word_frequencies_version ON media_bias.word_frequencies(result_version_id);
CREATE INDEX IF NOT EXISTS idx_word_frequencies_source ON media_bias.word_frequencies(result_version_id, source_id);
CREATE INDEX IF NOT EXISTS idx_word_frequencies_rank ON media_bias.word_frequencies(result_version_id, source_id, rank);
CREATE INDEX IF NOT EXISTS idx_named_entities_version ON media_bias.named_entities(result_version_id);
CREATE INDEX IF NOT EXISTS idx_named_entities_article ON media_bias.named_entities(article_id);
CREATE INDEX IF NOT EXISTS idx_named_entities_type ON media_bias.named_entities(result_version_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_named_entities_text ON media_bias.named_entities(result_version_id, entity_text);
CREATE INDEX IF NOT EXISTS idx_entity_stats_version ON media_bias.entity_statistics(result_version_id);
CREATE INDEX IF NOT EXISTS idx_entity_stats_source ON media_bias.entity_statistics(result_version_id, source_id);
CREATE INDEX IF NOT EXISTS idx_entity_stats_type ON media_bias.entity_statistics(result_version_id, entity_type);

-- HNSW index for similarity search (if pgvector supports it)
-- CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON media_bias.embeddings
--     USING hnsw (embedding vector_cosine_ops);

-- Sentiment Analysis
CREATE TABLE IF NOT EXISTS media_bias.sentiment_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    model_type VARCHAR(50) NOT NULL,  -- 'llm', 'local', 'hybrid'
    model_name VARCHAR(100),

    overall_sentiment FLOAT NOT NULL,
    overall_confidence FLOAT,
    headline_sentiment FLOAT NOT NULL,
    headline_confidence FLOAT,

    sentiment_reasoning TEXT,
    sentiment_aspects JSONB,

    processed_at TIMESTAMP DEFAULT NOW(),
    processing_time_ms INTEGER,

    UNIQUE(article_id, model_type)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_article ON media_bias.sentiment_analyses(article_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_model ON media_bias.sentiment_analyses(model_type);

-- Materialized view for sentiment aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS media_bias.sentiment_summary AS
SELECT
    sa.model_type,
    n.source_id,
    DATE_TRUNC('day', n.date_posted) as date,
    t.name as topic,
    AVG(sa.overall_sentiment) as avg_sentiment,
    STDDEV(sa.overall_sentiment) as sentiment_stddev,
    COUNT(*) as article_count
FROM media_bias.sentiment_analyses sa
JOIN media_bias.news_articles n ON sa.article_id = n.id
LEFT JOIN media_bias.article_analysis aa ON sa.article_id = aa.article_id
LEFT JOIN media_bias.topics t ON aa.primary_topic_id = t.id
GROUP BY sa.model_type, n.source_id, DATE_TRUNC('day', n.date_posted), t.name;

CREATE INDEX IF NOT EXISTS idx_sentiment_summary_model ON media_bias.sentiment_summary(model_type);
CREATE INDEX IF NOT EXISTS idx_sentiment_summary_source ON media_bias.sentiment_summary(source_id);

-- Additional indexes for multi-model analysis
CREATE INDEX IF NOT EXISTS idx_sentiment_model_article ON media_bias.sentiment_analyses(model_type, article_id);

-- Materialized view for faster topic-model queries
CREATE MATERIALIZED VIEW IF NOT EXISTS media_bias.sentiment_by_topic_model AS
SELECT
    sa.model_type,
    t.name as topic,
    n.source_id,
    AVG(sa.overall_sentiment) as avg_sentiment,
    STDDEV(sa.overall_sentiment) as stddev_sentiment,
    COUNT(*) as article_count
FROM media_bias.sentiment_analyses sa
JOIN media_bias.news_articles n ON sa.article_id = n.id
JOIN media_bias.article_analysis aa ON sa.article_id = aa.article_id
JOIN media_bias.topics t ON aa.primary_topic_id = t.id
WHERE t.topic_id != -1
GROUP BY sa.model_type, t.name, n.source_id;

CREATE INDEX IF NOT EXISTS idx_sentiment_topic_model_model ON media_bias.sentiment_by_topic_model(model_type);
CREATE INDEX IF NOT EXISTS idx_sentiment_topic_model_topic ON media_bias.sentiment_by_topic_model(topic);

-- Function to refresh all sentiment views
CREATE OR REPLACE FUNCTION media_bias.refresh_sentiment_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW media_bias.sentiment_summary;
    REFRESH MATERIALIZED VIEW media_bias.sentiment_by_topic_model;
END;
$$ LANGUAGE plpgsql;

-- Article Summaries
CREATE TABLE IF NOT EXISTS media_bias.article_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    summary_text TEXT NOT NULL,
    method VARCHAR(50) NOT NULL,  -- textrank, lexrank, bart, t5, pegasus, claude, gpt
    summary_length VARCHAR(20),  -- short, medium, long
    sentence_count INTEGER,
    word_count INTEGER,
    compression_ratio FLOAT,  -- original_length / summary_length
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(article_id, result_version_id)
);

CREATE INDEX IF NOT EXISTS idx_article_summaries_article ON media_bias.article_summaries(article_id);
CREATE INDEX IF NOT EXISTS idx_article_summaries_version ON media_bias.article_summaries(result_version_id);
CREATE INDEX IF NOT EXISTS idx_article_summaries_method ON media_bias.article_summaries(result_version_id, method);
