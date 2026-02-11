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
    pipeline_status JSONB DEFAULT '{"embeddings": false, "topics": false, "clustering": false, "word_frequency": false, "ner": false, "summarization": false, "ditwah": false, "ditwah_claims": false}'::jsonb,
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

-- Multi-document summaries for topic/cluster groups
CREATE TABLE IF NOT EXISTS media_bias.multi_doc_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_type VARCHAR(20) NOT NULL,  -- 'topic' | 'cluster'
    group_id TEXT NOT NULL,  -- references topics.id (integer) or event_clusters.id (UUID)
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,  -- multi-doc summarization version
    source_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,  -- topic or cluster version
    summary_text TEXT NOT NULL,
    method VARCHAR(50) NOT NULL,  -- 'openai' | 'gemini' (from version config)
    llm_model VARCHAR(100) NOT NULL,  -- e.g., 'gpt-4o', 'gemini-2.0-flash'
    article_count INTEGER NOT NULL,  -- number of articles summarized
    source_count INTEGER NOT NULL,  -- number of distinct sources
    word_count INTEGER,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(group_type, group_id, result_version_id, source_version_id)
);

CREATE INDEX IF NOT EXISTS idx_multi_doc_summaries_group ON media_bias.multi_doc_summaries(group_type, group_id);
CREATE INDEX IF NOT EXISTS idx_multi_doc_summaries_version ON media_bias.multi_doc_summaries(result_version_id);
CREATE INDEX IF NOT EXISTS idx_multi_doc_summaries_source_version ON media_bias.multi_doc_summaries(source_version_id);
CREATE INDEX IF NOT EXISTS idx_multi_doc_summaries_method ON media_bias.multi_doc_summaries(method);

-- Ditwah Hurricane Hypothesis Analysis
-- Hypotheses for Ditwah analysis
CREATE TABLE IF NOT EXISTS media_bias.ditwah_hypotheses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    hypothesis_key VARCHAR(20) NOT NULL,  -- 'h1', 'h2', etc.
    statement TEXT NOT NULL,
    category VARCHAR(50),  -- 'government_response', 'international_aid', etc.
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(result_version_id, hypothesis_key)
);

-- Stance analysis results per article-hypothesis pair
CREATE TABLE IF NOT EXISTS media_bias.ditwah_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    hypothesis_id UUID NOT NULL REFERENCES media_bias.ditwah_hypotheses(id) ON DELETE CASCADE,
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    agreement_score FLOAT NOT NULL,  -- -1.0 (disagree) to +1.0 (agree)
    confidence FLOAT NOT NULL,       -- 0 to 1
    stance VARCHAR(30) NOT NULL,     -- 'strongly_agree', 'agree', 'neutral', 'disagree', 'strongly_disagree'
    reasoning TEXT,
    supporting_quotes JSONB,         -- Array of quote strings
    llm_provider VARCHAR(50),
    llm_model VARCHAR(100),
    processed_at TIMESTAMP DEFAULT NOW(),
    processing_time_ms INTEGER,
    UNIQUE(article_id, hypothesis_id, result_version_id)
);

-- Indexes for Ditwah tables
CREATE INDEX IF NOT EXISTS idx_ditwah_hypotheses_version ON media_bias.ditwah_hypotheses(result_version_id);
CREATE INDEX IF NOT EXISTS idx_ditwah_analyses_version ON media_bias.ditwah_analyses(result_version_id);
CREATE INDEX IF NOT EXISTS idx_ditwah_analyses_article ON media_bias.ditwah_analyses(article_id);
CREATE INDEX IF NOT EXISTS idx_ditwah_analyses_hypothesis ON media_bias.ditwah_analyses(hypothesis_id);

-- Aggregated view by source
CREATE MATERIALIZED VIEW IF NOT EXISTS media_bias.ditwah_by_source AS
SELECT
    da.result_version_id,
    h.hypothesis_key,
    h.statement,
    n.source_id,
    AVG(da.agreement_score) as avg_agreement,
    STDDEV(da.agreement_score) as stddev_agreement,
    AVG(da.confidence) as avg_confidence,
    COUNT(*) as article_count,
    COUNT(*) FILTER (WHERE da.stance IN ('strongly_agree', 'agree')) as agree_count,
    COUNT(*) FILTER (WHERE da.stance = 'neutral') as neutral_count,
    COUNT(*) FILTER (WHERE da.stance IN ('disagree', 'strongly_disagree')) as disagree_count
FROM media_bias.ditwah_analyses da
JOIN media_bias.ditwah_hypotheses h ON da.hypothesis_id = h.id
JOIN media_bias.news_articles n ON da.article_id = n.id
GROUP BY da.result_version_id, h.hypothesis_key, h.statement, n.source_id;

CREATE INDEX IF NOT EXISTS idx_ditwah_by_source_version ON media_bias.ditwah_by_source(result_version_id);

-- Ditwah Claims Analysis (automatic claim generation + sentiment + stance)
-- Add boolean column to mark Ditwah articles
ALTER TABLE media_bias.news_articles
ADD COLUMN IF NOT EXISTS is_ditwah_cyclone BOOLEAN DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_news_articles_ditwah ON media_bias.news_articles(is_ditwah_cyclone)
WHERE is_ditwah_cyclone = TRUE;

-- Individual article claims (step 1 of two-step process)
CREATE TABLE IF NOT EXISTS media_bias.ditwah_article_claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id) ON DELETE CASCADE,
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    general_claim_id UUID,  -- Will reference ditwah_claims after clustering
    llm_provider VARCHAR(50),
    llm_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(article_id, result_version_id)
);

COMMENT ON TABLE media_bias.ditwah_article_claims IS
  'Individual claims generated for each DITWAH article (step 1: one claim per article)';

COMMENT ON COLUMN media_bias.ditwah_article_claims.general_claim_id IS
  'Links to the general claim this individual claim was clustered into (step 2)';

-- General claims table (step 2: clustered from individual claims)
CREATE TABLE IF NOT EXISTS media_bias.ditwah_claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    claim_category VARCHAR(100),
    claim_order INTEGER,  -- Display order
    article_count INTEGER,  -- How many articles mention this (via individual claims)
    individual_claims_count INTEGER DEFAULT 0,  -- How many individual claims clustered here
    representative_article_id UUID REFERENCES media_bias.news_articles(id),  -- Best example article
    llm_provider VARCHAR(50),
    llm_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(result_version_id, claim_text)
);

COMMENT ON TABLE media_bias.ditwah_claims IS
  'General claims created by clustering similar individual article claims (max ~40 general claims)';

COMMENT ON COLUMN media_bias.ditwah_claims.representative_article_id IS
  'Article that best represents this general claim';

-- Add foreign key constraint after both tables exist
ALTER TABLE media_bias.ditwah_article_claims
ADD CONSTRAINT fk_general_claim
FOREIGN KEY (general_claim_id) REFERENCES media_bias.ditwah_claims(id) ON DELETE SET NULL;

-- Sentiment analysis for claims (links to existing sentiment_analyses table)
CREATE TABLE IF NOT EXISTS media_bias.claim_sentiment (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES media_bias.ditwah_claims(id) ON DELETE CASCADE,
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    source_id VARCHAR(50) NOT NULL,
    -- Sentiment from existing models
    sentiment_score FLOAT,  -- -5 to +5 from primary model
    sentiment_model VARCHAR(50),  -- Which model
    UNIQUE(claim_id, article_id)
);

-- Stance analysis for claims (new LLM analysis)
CREATE TABLE IF NOT EXISTS media_bias.claim_stance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES media_bias.ditwah_claims(id) ON DELETE CASCADE,
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id),
    source_id VARCHAR(50) NOT NULL,
    -- Stance from LLM
    stance_score FLOAT,  -- -1 (disagree) to +1 (agree)
    stance_label VARCHAR(50),  -- 'strongly_agree', 'agree', 'neutral', 'disagree', 'strongly_disagree'
    confidence FLOAT,  -- 0-1
    reasoning TEXT,
    supporting_quotes JSONB,
    llm_provider VARCHAR(50),
    llm_model VARCHAR(100),
    processed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(claim_id, article_id)
);

-- Indexes for article claims
CREATE INDEX IF NOT EXISTS idx_article_claims_article ON media_bias.ditwah_article_claims(article_id);
CREATE INDEX IF NOT EXISTS idx_article_claims_version ON media_bias.ditwah_article_claims(result_version_id);
CREATE INDEX IF NOT EXISTS idx_article_claims_general ON media_bias.ditwah_article_claims(general_claim_id);

-- Indexes for general claims
CREATE INDEX IF NOT EXISTS idx_ditwah_claims_version ON media_bias.ditwah_claims(result_version_id);
CREATE INDEX IF NOT EXISTS idx_ditwah_claims_representative ON media_bias.ditwah_claims(representative_article_id);

-- Indexes for claim sentiment and stance
CREATE INDEX IF NOT EXISTS idx_claim_sentiment_claim ON media_bias.claim_sentiment(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_sentiment_source ON media_bias.claim_sentiment(claim_id, source_id);
CREATE INDEX IF NOT EXISTS idx_claim_stance_claim ON media_bias.claim_stance(claim_id);
CREATE INDEX IF NOT EXISTS idx_claim_stance_source ON media_bias.claim_stance(claim_id, source_id);

-- View: Individual → General claim hierarchy
CREATE OR REPLACE VIEW media_bias.ditwah_claims_hierarchy AS
SELECT
    ac.id as individual_claim_id,
    ac.article_id,
    ac.claim_text as individual_claim,
    gc.id as general_claim_id,
    gc.claim_text as general_claim,
    gc.claim_category,
    n.title as article_title,
    n.source_id,
    n.date_posted,
    ac.result_version_id
FROM media_bias.ditwah_article_claims ac
LEFT JOIN media_bias.ditwah_claims gc ON ac.general_claim_id = gc.id
LEFT JOIN media_bias.news_articles n ON ac.article_id = n.id
ORDER BY gc.claim_order, ac.created_at;

COMMENT ON VIEW media_bias.ditwah_claims_hierarchy IS
  'Shows the mapping from individual article claims to general claims for analysis';
