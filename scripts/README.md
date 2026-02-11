# Analysis Pipeline Scripts

All pipelines use the version-based architecture. See [CLAUDE.md](../CLAUDE.md) for detailed usage.

## Pipelines

| Directory | Pipeline | Steps | Version Type |
|-----------|----------|-------|--------------|
| `topics/` | Topic Discovery | 01_embeddings → 02_discover | `topics` |
| `clustering/` | Event Clustering | 01_embeddings → 02_cluster | `clustering` |
| `summarization/` | Summarization | 01_generate | `summarization` |
| `sentiment/` | Sentiment Analysis | 01_analyze | N/A (no versions) |
| `ner/` | Named Entity Recognition | 01_extract | `ner` |
| `word_frequency/` | Word Frequency | 01_compute | `word_frequency` |
| `ditwah_claims/` | Ditwah Claims | 01_mark → 02_generate → 04_sentiment_stance | `ditwah_claims` |

## Utilities

- `manage_versions.py` - Manage analysis versions (list, view stats, delete)
- `utilities/create_multi_doc_versions.py` - Create multi-doc summarization versions
- `utilities/load_claims_to_database.py` - Load pre-extracted claims from JSON
