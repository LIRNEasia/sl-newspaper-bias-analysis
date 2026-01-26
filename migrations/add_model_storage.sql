-- Add BERTopic model storage to result_versions table
-- This allows storing trained models in the database for team collaboration
-- Run with: psql -h localhost -U username -d database -f migrations/add_model_storage.sql

-- Add model_data column to store compressed BERTopic models
ALTER TABLE media_bias.result_versions
ADD COLUMN IF NOT EXISTS model_data BYTEA;

COMMENT ON COLUMN media_bias.result_versions.model_data IS
  'Compressed tar.gz archive of BERTopic model directory (for visualizations). NULL if model not stored in database.';
