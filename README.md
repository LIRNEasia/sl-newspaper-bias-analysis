# Sri Lanka Newspaper Bias Analysis

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A data-driven analysis framework for detecting media bias in Sri Lankan English newspapers by examining coverage patterns, topic distribution, and event clustering.

## Overview

This project provides a framework for analyzing news articles from Sri Lankan newspapers to identify:

- 📰 **Selection bias**: Which topics each source covers (or ignores)
- 🔍 **Coverage patterns**: How different sources cover the same events
- 🏷️ **Topic discovery**: Data-driven topic categorization using BERTopic
- 📊 **Event clustering**: Grouping articles about the same events across sources
- 😊 **Sentiment analysis**: Emotional tone across sources and topics
- 📝 **Article summarization**: Generate concise summaries using multiple methods

## Features

- 🧠 **Semantic embeddings**: 768-dimensional vectors using local models (no API needed)
- 🎯 **Topic modeling**: BERTopic with UMAP + HDBSCAN clustering
- 🔗 **Event clustering**: Cosine similarity with time-window constraints
- 😊 **Sentiment analysis**: Multiple sentiment models (RoBERTa, VADER, FinBERT, etc.)
- 📝 **Article summarization**: Extractive, transformer, and LLM-based methods
- 📈 **Interactive dashboard**: Streamlit-based visualization with version management
- 🗄️ **Vector database**: PostgreSQL with pgvector extension
- 🔄 **Version management**: Track and compare different analysis configurations

## Tech Stack

- **Python 3.11+**: Core language
- **PostgreSQL 16+ with pgvector**: Database with vector similarity search
- **Sentence Transformers**: Local embedding generation (no API needed)
- **BERTopic**: Topic modeling with UMAP/HDBSCAN
- **Transformers**: Multiple sentiment analysis models
- **Streamlit**: Interactive dashboard
- **pandas, numpy**: Data processing

## Quick Start

### Prerequisites

```bash
# Database
PostgreSQL 16 with pgvector extension

# Python
Python 3.11+
```

### Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sl-newspaper-bias-analysis.git
   cd sl-newspaper-bias-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure database**
   ```bash
   # Copy configuration template
   cp config.yaml.example config.yaml

   # Edit config.yaml with your database credentials
   nano config.yaml
   ```

4. **Set up database schema**
   ```bash
   psql -h localhost -U your_db_user -d your_database -f schema.sql
   ```

5. **Run the analysis pipeline**
   ```bash
   # Generate embeddings for topics
   python3 scripts/topics/01_generate_embeddings.py --version-id <version-id>

   # Discover topics
   python3 scripts/topics/02_discover_topics.py --version-id <version-id>

   # Generate embeddings for clustering
   python3 scripts/clustering/01_generate_embeddings.py --version-id <version-id>

   # Cluster events
   python3 scripts/clustering/02_cluster_events.py --version-id <version-id>

   # Analyze sentiment
   python3 scripts/sentiment/01_analyze_sentiment.py
   ```

6. **Launch dashboard**
   ```bash
   streamlit run dashboard/Home.py
   # Access at http://localhost:8501
   ```

## Project Structure

```
sl-newspaper-bias-analysis/
├── config.yaml.example     # Configuration template
├── schema.sql              # Database schema
├── requirements.txt        # Python dependencies
├── src/
│   ├── db.py              # Database operations
│   ├── llm.py             # LLM client abstraction
│   ├── embeddings.py      # Embedding generation
│   ├── topics.py          # Topic modeling
│   ├── sentiment.py       # Sentiment analysis (3 models)
│   ├── clustering.py      # Event clustering
│   ├── word_frequency.py  # Word frequency analysis
│   ├── ner.py             # Named entity recognition
│   └── versions.py        # Result version management
├── scripts/
│   ├── topics/
│   │   ├── 01_generate_embeddings.py
│   │   └── 02_discover_topics.py
│   ├── clustering/
│   │   ├── 01_generate_embeddings.py
│   │   └── 02_cluster_events.py
│   ├── word_frequency/
│   │   └── 01_compute_word_frequency.py
│   ├── ner/
│   │   └── 01_extract_entities.py
│   ├── manage_versions.py
│   └── 04_analyze_sentiment.py
└── dashboard/
    └── Home.py            # Streamlit dashboard
```

## Dashboard Preview

The dashboard includes multiple interactive tabs:

1. **📊 Coverage Tab**: Article volume and timeline by source
2. **🏷️ Topics Tab**: Top topics, source-topic heatmap, BERTopic visualizations
3. **📰 Events Tab**: Browse event clusters and cross-source coverage
4. **📝 Summaries Tab**: Article summaries with compression statistics
5. **😊 Sentiment Tab**: Sentiment distribution across sources and models

Each analysis tab has its own independent version selector for experimentation.

## Database Schema

### Original Data
- `news_articles` - Scraped newspaper articles

### Result Versioning
- `result_versions` - Configuration-based version tracking for reproducible analysis

### Analysis Tables
- `embeddings` - Article embeddings (768-dim vectors)
- `topics` - Discovered topics
- `article_analysis` - Article-topic assignments
- `event_clusters` - Event clusters
- `article_clusters` - Article-to-cluster mappings
- `article_summaries` - Generated summaries
- `word_frequencies` - Word frequency rankings per source
- `named_entities` - Extracted entities with positions and confidence
- `sentiment_scores` - Sentiment analysis results per model

## Sentiment Analysis

The sentiment analysis system uses multiple sentiment models to analyze article sentiment on a scale from -5 (very negative) to +5 (very positive).

### Available Models

- **RoBERTa** - Twitter-trained, accurate
- **DistilBERT** - Lightweight, general sentiment
- **FinBERT** - Optimized for financial/economic news
- **VADER** - Lexicon-based, very fast
- **TextBlob** - Pattern-based, simple

### Running Sentiment Analysis

```bash
# Run all enabled models (configured in config.yaml)
python3 scripts/sentiment/01_analyze_sentiment.py

# Run specific models only
python3 scripts/sentiment/01_analyze_sentiment.py --models roberta vader

# Test on limited articles
python3 scripts/sentiment/01_analyze_sentiment.py --limit 100
```

### Sentiment Scale

Sentiment scores range from:
- **-5 to -3**: Very negative
- **-2 to -1**: Somewhat negative
- **-0.5 to 0.5**: Neutral
- **1 to 2**: Somewhat positive
- **3 to 5**: Very positive

## Research Methodology

Based on: "The Media Bias Detector: A Framework for Annotating and Analyzing the News at Scale" (UPenn, 2025)

### Adapted for Sri Lankan Context
- ❌ **Skipped**: Political lean (Democrat/Republican) - not applicable to SL politics
- ✅ **Kept**: Topic hierarchy via data-driven discovery
- ✅ **Kept**: Event clustering for coverage comparison
- ✅ **Kept**: Selection bias analysis (topic coverage patterns)
- ⏸️ **Future**: Framing bias analysis (requires tone scoring via LLM)

## Future Enhancements

### With LLM API (Claude/OpenAI)
1. **Tone Analysis**: Score articles on -5 to +5 scale
2. **Article Type Classification**: news/opinion/analysis/editorial
3. **Sentence-level Analysis**: fact/opinion/quote classification
4. **Quote Extraction**: Extract speaker information
5. **Better Topic Labels**: Use LLM to generate descriptive topic names

### Other Improvements
1. **Hierarchical Topics**: Parent-child topic relationships
2. **Time-series Analysis**: Topic trends over time
3. **Source Comparison Metrics**: Quantify selection bias
4. **Framing Analysis**: Compare how sources frame the same events
5. **Export Functionality**: Download analysis results
## Configuration

All configuration is in `config.yaml`:

```yaml
database:
  host: localhost
  name: your_database
  schema: your_schema
  user: your_db_user
  password: "YOUR_PASSWORD"

embeddings:
  provider: local  # local (free) | openai
  model: all-mpnet-base-v2  # all-mpnet-base-v2 | google/embeddinggemma-300m

topics:
  min_topic_size: 10
  diversity: 0.5

clustering:
  similarity_threshold: 0.8
  time_window_days: 7
  min_cluster_size: 2

sentiment:
  enabled_models:
    - roberta
    - vader

summarization:
  method: textrank  # textrank | bart | pegasus | claude | gemini
  summary_length: medium  # short | medium | long
```

## Performance

- **Embedding generation**: Varies by dataset size (CPU-based)
- **Topic discovery**: Fast (minutes)
- **Event clustering**: Fast (minutes)
- **Sentiment analysis**: Depends on model choice (free local models or API-based)
- **Memory usage**: ~2GB RAM during embedding generation
- **Dashboard**: Queries cached for fast load times

## Managing Result Versions

The project uses a version management system to track different analysis configurations. This allows you to experiment with different parameters and compare results.

### List Versions

```bash
# List all versions
python3 scripts/manage_versions.py list

# Filter by analysis type
python3 scripts/manage_versions.py list --type topics
python3 scripts/manage_versions.py list --type clustering
python3 scripts/manage_versions.py list --type word_frequency
```

### View Version Statistics

Before deleting, check what data a version contains:

```bash
python3 scripts/manage_versions.py stats <version-id>
```

This shows:
- Version metadata (name, type, description, dates)
- Data counts (embeddings, topics, clusters, etc.)
- Total records that would be affected

### Delete a Version

**Interactive deletion with safety prompts:**

```bash
python3 scripts/manage_versions.py delete <version-id>
```

This command:
- ✅ Shows version details and statistics
- ✅ Displays all data that will be deleted
- ✅ Requires you to type the version name to confirm
- ✅ Requires you to type 'DELETE' for final confirmation
- ✅ Cascade deletes all related records automatically
- ✅ **Never deletes** original articles in `news_articles` table

**What gets deleted:**
- Embeddings (embedding vectors)
- Topics (discovered topics)
- Article analyses (article-topic assignments)
- Event clusters (grouped events)
- Article-cluster mappings
- Word frequencies (if applicable)

**Programmatic deletion (Python):**

```python
# Safe interactive deletion
from src.versions import delete_version_interactive
delete_version_interactive("version-id-here")

# Direct deletion (no confirmation - use with caution!)
from src.versions import delete_version
success = delete_version("version-id-here")

# Preview what will be deleted
from src.versions import get_version_statistics
stats = get_version_statistics("version-id-here")
print(f"Will delete {sum(stats.values())} records")
```

## License

MIT License - see LICENSE file for details
