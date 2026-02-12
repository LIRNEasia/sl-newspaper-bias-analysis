# Sri Lanka Media Bias Detector

A framework for analyzing media bias in Sri Lankan English newspapers by examining coverage patterns, topic distribution, and event clustering.

## Overview

This project provides tools to analyze news articles from Sri Lankan newspapers to identify:
- **Selection bias**: Which topics each source covers (or ignores)
- **Coverage patterns**: How different sources cover the same events
- **Topic discovery**: Data-driven topic categorization using BERTopic
- **Sentiment analysis**: Emotional tone across sources and topics
- **Event clustering**: Grouping similar articles across time and sources

## ⚠️ Security - READ THIS FIRST

**CRITICAL: Never commit sensitive information to git**

This includes:
- ❌ API keys (OpenAI, Anthropic, Google, etc.)
- ❌ Database passwords
- ❌ Access tokens
- ❌ Private keys
- ❌ Any credentials or secrets

**Always use environment variables or .env files (added to .gitignore) for sensitive data.**

If you accidentally commit secrets:
1. **Immediately rotate/revoke the exposed credentials**
2. **Rewrite git history** to remove them (see the incident resolution in git history)
3. **Never assume deletion is enough** - once pushed, consider credentials compromised

**Proper practices:**
- Use `config.yaml` for non-sensitive configuration (already in .gitignore)
- Load API keys from environment variables: `os.environ.get("API_KEY")`
- For notebooks: Check environment variables, never hardcode keys
- Double-check before committing: `git diff --staged`

## First-Time Setup

If you're setting up this project for the first time:

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sl-newspaper-bias-analysis.git
   cd sl-newspaper-bias-analysis
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create configuration file**
   ```bash
   # Copy the example configuration
   cp config.yaml.example config.yaml

   # Edit config.yaml and add your database credentials
   nano config.yaml  # or use your preferred editor
   ```

   Replace the placeholder values:
   - `your_database` → Your PostgreSQL database name
   - `your_schema` → Your database schema name
   - `your_db_user` → Your database username
   - `YOUR_DATABASE_PASSWORD_HERE` → Your database password

4. **Set up database**
   ```bash
   # Create schema and tables
   psql -h localhost -U your_db_user -d your_database -f schema.sql
   ```

5. **Load your news article data**

   The project expects a table `news_articles` in your schema with columns:
   - `article_id` (integer, primary key)
   - `title` (text)
   - `content` (text)
   - `source` (text) - newspaper name
   - `published_date` (date)
   - `url` (text)

   Import your scraped articles into this table.

6. **Create your first result versions**

   Since topics and clustering are now independent, create separate versions:

   ```bash
   # Create a topic version
   python3 -c "
   from src.versions import create_version, get_default_topic_config
   version_id = create_version('baseline-topics', 'Initial topic analysis', get_default_topic_config(), analysis_type='topics')
   print(f'Topic version ID: {version_id}')
   "

   # Create a clustering version
   python3 -c "
   from src.versions import create_version, get_default_clustering_config
   version_id = create_version('baseline-clustering', 'Initial clustering analysis', get_default_clustering_config(), analysis_type='clustering')
   print(f'Clustering version ID: {version_id}')
   "
   ```

   Or create versions via the dashboard (easier for beginners).

7. **Run the analysis pipelines** (see "Running the Pipeline" section below)

### For Contributing to the Project

If you're contributing to an organization repository:

1. **Fork the repository** on GitHub (click "Fork" button)

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sl-newspaper-bias-analysis.git
   cd sl-newspaper-bias-analysis
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/ORG_NAME/sl-newspaper-bias-analysis.git
   git remote -v  # Verify remotes are set up correctly
   ```

4. **Keep your fork synced**
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

## Project Structure

```
database-analysis/
├── config.yaml              # Configuration (LLM, embeddings, clustering)
├── schema.sql              # Database schema (includes analysis_type for decoupled analyses)
├── requirements.txt        # Python dependencies
├── src/
│   ├── db.py              # Database operations (PostgreSQL + pgvector)
│   ├── llm.py             # LLM client abstraction (Claude/OpenAI/local)
│   ├── embeddings.py      # Embedding generation (local/OpenAI)
│   ├── topics.py          # BERTopic hierarchical topic modeling
│   ├── clustering.py      # Event clustering using embeddings
│   └── versions.py        # Result version management (with analysis_type)
├── scripts/
│   ├── embeddings/         # Shared embedding generation
│   │   └── 01_generate_embeddings.py
│   ├── clustering/         # Event clustering pipeline
│   │   └── 02_cluster_events.py
│   ├── ner/               # Named entity recognition
│   │   └── 01_extract_entities.py
│   ├── sentiment/         # Sentiment analysis
│   │   └── 01_analyze_sentiment.py
│   ├── summarization/     # Article summarization
│   │   └── 01_generate_summaries.py
│   ├── topics/            # Topic analysis pipeline
│   │   └── 02_discover_topics.py
│   ├── word_frequency/    # Word frequency analysis
│   │   └── 01_compute_word_frequency.py
│   └── manage_versions.py # Version management CLI
└── dashboard/
    └── Home.py            # Streamlit dashboard (separate version selectors per tab)
```

## Architecture

### Decoupled Analysis System

The framework uses **independent analysis pipelines** with separate version management:

```
Topics Analysis              Event Clustering            Sentiment Analysis           Summarization
└── Topic Versions          └── Clustering Versions     └── Model-based             └── Summarization Versions
    ├── Embeddings              ├── Embeddings             └── No versioning            ├── Method selection
    └── Topic Discovery         └── Event Clustering                                     └── Length control
```

**Key Design Principles:**
- **Independence**: Change topic parameters without re-running clustering (and vice versa)
- **Efficiency**: Only re-run what actually changed
- **Reproducibility**: Each version tracks exact configuration used
- **Flexibility**: Compare different approaches side-by-side

### Database Schema

**Original Data:**
- `news_articles` - Scraped newspaper articles with title, content, source, date, URL

**Result Versioning:**
- `result_versions` - Configuration-based version tracking for reproducible analysis
  - `analysis_type` column: 'topics', 'clustering', 'summarization', or 'combined' (legacy)
  - Independent versions per analysis type (same name can exist for different types)

**Analysis Tables:**
- `embeddings` - Article embeddings (configurable dimensions, typically 768-dim)
- `topics` - Discovered topics (linked to topic versions)
- `article_analysis` - Article-topic assignments
- `event_clusters` - Event clusters (linked to clustering versions)
- `article_clusters` - Article-to-cluster mappings
- `article_summaries` - Generated summaries (linked to summarization versions)
- `sentiment_scores` - Sentiment analysis results per model

All analysis tables link to `result_versions` for configuration tracking and reproducibility.

## Analysis Pipelines

All pipelines use **independent version management** - you can experiment with different configurations without affecting other analyses.

### 1. Topic Analysis

Discover data-driven topics using BERTopic clustering.

**Create a topic version:**
```bash
# Via Python
python3 -c "
from src.versions import create_version, get_default_topic_config
version_id = create_version('baseline-topics', 'Initial topic analysis', get_default_topic_config(), analysis_type='topics')
print(f'Version ID: {version_id}')
"

# Or via dashboard: Topics tab → "➕ Create New Topics Version"
```

**Run the pipeline:**
```bash
# Generate embeddings (shared across versions, only needed once per model)
python3 scripts/embeddings/01_generate_embeddings.py --model all-mpnet-base-v2

# Discover topics (set PYTHONHASHSEED for reproducibility)
# Embeddings are auto-generated if not already present
PYTHONHASHSEED=42 python3 scripts/topics/02_discover_topics.py --version-id <version-id>
```

**How it works:**
- Uses shared embeddings by model name (auto-generates if missing)
- Uses BERTopic with UMAP + HDBSCAN clustering
- Automatically discovers topics from article content
- Generates keyword-based topic labels
- Stores BERTopic model in database for visualizations and team sharing

### 2. Event Clustering

Group similar articles across sources to identify shared news events.

**Create a clustering version:**
```bash
# Via Python
python3 -c "
from src.versions import create_version, get_default_clustering_config
version_id = create_version('baseline-clustering', 'Initial clustering analysis', get_default_clustering_config(), analysis_type='clustering')
print(f'Version ID: {version_id}')
"

# Or via dashboard: Events tab → "➕ Create New Clustering Version"
```

**Run the pipeline:**
```bash
# Generate embeddings (shared across versions, only needed once per model)
python3 scripts/embeddings/01_generate_embeddings.py --model all-mpnet-base-v2

# Cluster events (embeddings are auto-generated if not already present)
python3 scripts/clustering/02_cluster_events.py --version-id <version-id>
```

**How it works:**
- Uses shared embeddings by model name (auto-generates if missing)
- Groups similar articles using cosine similarity
- Applies time window constraint (default: 7 days)
- Identifies multi-source coverage of same events

### 3. Sentiment Analysis

Analyze emotional tone using multiple sentiment models (no version management needed).

**Run sentiment analysis:**
```bash
# Run all enabled models (configured in config.yaml)
python3 scripts/sentiment/01_analyze_sentiment.py

# Run specific models only
python3 scripts/sentiment/01_analyze_sentiment.py --models roberta vader

# Test on limited articles
python3 scripts/sentiment/01_analyze_sentiment.py --limit 100
```

**Available models:**
- **RoBERTa** - Twitter-trained, accurate
- **DistilBERT** - Lightweight, general sentiment
- **FinBERT** - Optimized for financial/economic news
- **VADER** - Lexicon-based, very fast
- **TextBlob** - Pattern-based, simple

**Configuration** (`config.yaml`):
```yaml
sentiment:
  enabled_models:
    - roberta
    - vader
```

**Output:** Sentiment scores from -5 (very negative) to +5 (very positive) for headlines and full articles.

### 4. Article Summarization

Generate summaries using extractive, abstractive (transformer), or LLM-based methods.

**Create a summarization version:**
```bash
python3 -c "
from src.versions import create_version, get_default_summarization_config
config = get_default_summarization_config()
config['summarization']['method'] = 'textrank'  # or bart, pegasus, claude, gemini, etc.
version_id = create_version('textrank-medium', 'TextRank summarization', config, 'summarization')
print(f'Version ID: {version_id}')
"

# Or via dashboard: Summaries tab → "➕ Create New Summarization Version"
```

**Run the pipeline:**
```bash
python3 scripts/summarization/01_generate_summaries.py --version-id <version-id>
```

**Available methods:**

**Extractive (Fast, Free):**
- **TextRank** / **LexRank** - Graph-based sentence selection

**Abstractive - Standard Length (Free, Local):**
- **BART**, **T5**, **Pegasus** - Transformer models (~500-1000 word articles)

**Abstractive - Long Context (Free, Local):**
- **LED**, **LongT5**, **BigBird-Pegasus** - Handle multi-page documents

**LLM-Based (API, High Quality):**
- **Claude Sonnet 4**, **GPT-4o**, **Gemini 2.0 Flash** - Unlimited length, best quality

See "Article Summarization" section below for detailed comparison and configuration options.

## Dashboard

**Launch the dashboard:**
```bash
streamlit run dashboard/Home.py
# Access at http://localhost:8501
```

**Features:**
- **📊 Coverage Tab** - Article volume and timeline (version-independent)
- **🏷️ Topics Tab** - Topic distribution, source heatmaps, BERTopic visualizations
- **📰 Events Tab** - Event clusters, multi-source coverage analysis
- **📝 Summaries Tab** - Article summaries with compression statistics
- **😊 Sentiment Tab** - Sentiment distribution across sources and models

Each analysis tab has its own **independent version selector** - experiment with different configurations without affecting other analyses.

**Sharing the dashboard:**

*Local access (SSH tunnel):*
```bash
ssh -L 8501:localhost:8501 user@server-ip
# Then open http://localhost:8501 on your local machine
```

*Public access (Cloudflare Tunnel):*
```bash
cloudflared tunnel --url http://localhost:8501
# Provides a public https:// URL
```

## Version Management

The framework uses **independent versioning** for each analysis type, enabling reproducible experiments without interference.

### Key Concepts

**Analysis Types:**
- `topics` - Topic discovery versions
- `clustering` - Event clustering versions
- `summarization` - Summarization method versions
- `combined` - Legacy (not recommended)

**Benefits:**
- **Independence** - Change topic parameters without re-running clustering
- **Efficiency** - Only re-run what changed
- **Clarity** - Each analysis has its own version history
- **Reproducibility** - Exact configuration tracking per result
- **Flexibility** - Same version name can exist across different analysis types

### Creating Versions

**Via Dashboard (Recommended):**
1. Navigate to the relevant tab (Topics, Events, or Summaries)
2. Click "➕ Create New Version"
3. Enter name and description
4. Edit configuration JSON
5. Click "Create Version"

**Programmatically (Python):**
```python
from src.versions import create_version, get_default_topic_config

config = get_default_topic_config()
config['topics']['min_topic_size'] = 15  # Customize as needed

version_id = create_version(
    name="small-topics",
    description="Testing smaller minimum topic size",
    configuration=config,
    analysis_type='topics'  # or 'clustering', 'summarization'
)
```

After creating a version, run the appropriate pipeline scripts (see Analysis Pipelines section above).

### Key Configuration Parameters

**Topics:**
- `embeddings.model` - Embedding model (all-mpnet-base-v2, embeddinggemma-300m)
- `topics.min_topic_size` - Minimum articles per topic
- `topics.diversity` - Keyword diversity in labels
- `topics.umap` / `topics.hdbscan` - Clustering algorithm parameters

**Clustering:**
- `embeddings.model` - Embedding model
- `clustering.similarity_threshold` - Cosine similarity threshold (0-1)
- `clustering.time_window_days` - Time constraint for grouping
- `clustering.min_cluster_size` - Minimum articles per cluster

**Summarization:**
- `summarization.method` - Method (textrank, bart, pegasus, claude, gemini, etc.)
- `summarization.summary_length` - short/medium/long
- `summarization.llm_model` - LLM model if using API method

*Note: Random seed is fixed at `42` for reproducibility.*

### Managing Versions (CLI)

**List versions:**
```bash
python3 scripts/manage_versions.py list                  # All versions
python3 scripts/manage_versions.py list --type topics    # Filter by type
```

**View statistics:**
```bash
python3 scripts/manage_versions.py stats <version-id>
```

**Delete version (interactive):**
```bash
python3 scripts/manage_versions.py delete <version-id>
```

⚠️ **Deletion is permanent** - removes all associated results (embeddings, topics, clusters) but **never deletes original articles**.

### Best Practices

1. **Start with baselines** - Create default configuration versions first
2. **Descriptive names** - Use names that explain what changed (e.g., "small-topics", "high-similarity")
3. **One change at a time** - Modify single parameters for clear A/B comparisons
4. **Document reasoning** - Explain why you're testing specific parameters
5. **Independent experiments** - Topic and clustering versions don't affect each other

## Article Summarization

The project supports multiple summarization methods to generate concise summaries of news articles. Each method runs as an independent version, allowing experimentation with different approaches.

### Supported Methods

**Extractive Methods (Free, Fast):**
- **TextRank**: Graph-based ranking algorithm that selects important sentences
- **LexRank**: Similar to TextRank, using sentence similarity for ranking

**Abstractive Methods (Local Models - Standard Length):**
- **BART** (`facebook/bart-large-cnn`): 1.6GB, excellent quality, handles articles up to ~750 words
- **T5** (`t5-base`): 890MB, good quality, handles articles up to ~380 words
- **Pegasus** (`google/pegasus-xsum`): 2.2GB, best quality, handles articles up to ~750 words

**Abstractive Methods (Local Models - Long Context):**
- **LED** (`allenai/led-base-16384`): 200MB, handles articles up to ~12,000 words, uses efficient long-range attention
- **LongT5** (`google/long-t5-tglobal-base`): 250MB, handles articles up to ~3,000 words, good for long documents
- **BigBird-Pegasus** (`google/bigbird-pegasus-large-arxiv`): 2.8GB, handles articles up to ~3,000 words
  - **Note**: This is the "large" model (no base variant exists) and will be slow on CPU (10-20s per article)
  - Trained on academic papers, may need fine-tuning for news articles

**LLM-Based Methods (API):**
- **Claude Sonnet 4**: Highest quality, handles all article lengths
- **GPT-4o**: Similar quality to Claude, comparable cost
- **Gemini 2.0 Flash**: Excellent quality, competitive pricing, handles all article lengths

**Note on Length Control:**
- **Extractive methods** (TextRank, LexRank): Use `target_sentences` to select top N sentences
- **Transformer models** (BART, T5, etc.): Generate naturally without hard limits; target lengths are informational
- **LLM methods** (Claude, GPT, Gemini): Use target lengths in prompt; have max_tokens limit for cost control

### Creating a Summarization Version

**Via Dashboard:**
1. Navigate to Summaries tab
2. Click "➕ Create New Summarization Version"
3. Enter name (e.g., "textrank-medium")
4. Edit configuration JSON
5. Click "Create Version"

**Programmatically:**
```python
from src.versions import create_version, get_default_summarization_config

# Create a TextRank version
config = get_default_summarization_config()
config['summarization']['method'] = 'textrank'
config['summarization']['summary_length'] = 'medium'

version_id = create_version(
    name="textrank-medium",
    description="Extractive summarization using TextRank",
    configuration=config,
    analysis_type='summarization'
)
print(f"Created version: {version_id}")

# Create a BART version
config = get_default_summarization_config()
config['summarization']['method'] = 'bart'
config['summarization']['summary_length'] = 'short'

version_id = create_version(
    name="bart-short",
    description="Abstractive summarization with BART (short)",
    configuration=config,
    analysis_type='summarization'
)

# Create a Claude version
config = get_default_summarization_config()
config['summarization']['method'] = 'claude'
config['summarization']['summary_length'] = 'medium'

version_id = create_version(
    name="claude-medium",
    description="LLM-based summarization with Claude",
    configuration=config,
    analysis_type='summarization'
)

# Create a Gemini version
config = get_default_summarization_config()
config['summarization']['method'] = 'gemini'
config['summarization']['llm_model'] = 'gemini-2.0-flash'
config['summarization']['summary_length'] = 'medium'

version_id = create_version(
    name="gemini-medium",
    description="LLM-based summarization with Gemini",
    configuration=config,
    analysis_type='summarization'
)
```

### Running the Pipeline

**Single-step pipeline:**
```bash
python3 scripts/summarization/01_generate_summaries.py --version-id <version-id>

# Optional: adjust batch size
python3 scripts/summarization/01_generate_summaries.py --version-id <version-id> --batch-size 100
```

**Example workflow:**
```bash
# 1. Create versions for comparison
python3 -c "from src.versions import *; print(create_version('textrank-medium', 'TextRank medium', get_default_summarization_config(), 'summarization'))"

# 2. Run pipeline
python3 scripts/summarization/01_generate_summaries.py --version-id <version-id>

# 3. View results in dashboard
streamlit run dashboard/Home.py
```

### Configuration Options

**Summary Length Targets:**
- **short**: 3 sentences / 50 words
- **medium**: 5 sentences / 100 words (default)
- **long**: 8 sentences / 150 words

**Method-Specific Settings:**
```yaml
summarization:
  method: textrank  # textrank | lexrank | bart | t5 | pegasus | led | longt5 | bigbird-pegasus | claude | gpt | gemini
  summary_length: medium  # short | medium | long

  # Length targets - how they're used:
  # - Extractive methods: Controls number of sentences extracted
  # - Transformers: Informational only (models use natural stopping)
  # - LLMs: Used in prompt as guidance
  short_sentences: 3
  short_words: 50
  medium_sentences: 5
  medium_words: 100
  long_sentences: 8
  long_words: 150

  # Transformer model settings
  max_input_length: 1024       # Max tokens for standard models (BART/T5/Pegasus)
  chunk_long_articles: true    # Split articles longer than max_input_length

  # LLM settings (has hard max_tokens limit for cost control)
  llm_model: claude-sonnet-4-20250514
  llm_temperature: 0.0
```

### Performance Comparison

Based on ~500 word articles (CPU performance):

| Method | Speed (CPU) | Quality | Cost | Model Size | Max Length |
|--------|-------------|---------|------|------------|------------|
| TextRank | 10-50ms | Good | Free | None | Unlimited |
| LexRank | 10-50ms | Good | Free | None | Unlimited |
| BART | 5-10s | High | Free | 1.6GB | ~750 words |
| T5 | 3-8s | Good | Free | 890MB | ~380 words |
| Pegasus | 5-12s | Excellent | Free | 2.2GB | ~750 words |
| LED | 8-15s | Good | Free | 200MB | ~12,000 words |
| LongT5 | 6-12s | Good | Free | 250MB | ~3,000 words |
| BigBird-Pegasus | 10-20s | High | Free | 2.8GB | ~3,000 words |
| Claude | 1-3s | Excellent | ~$0.001/article | API | Unlimited |
| GPT-4 | 1-3s | Excellent | ~$0.001/article | API | Unlimited |
| Gemini | 1-3s | Excellent | ~$0.0005/article | API | Unlimited |

**Notes:**
- GPU speeds are typically 5-10x faster than CPU
- Long-context models (LED, LongT5, BigBird-Pegasus) excel at articles >1000 words
- All transformer models run on CPU by default to avoid CUDA tokenization issues
- **Summary lengths are natural/variable** - transformers decide appropriate length per article
- LLM methods have cost per article; local transformers are free

**Recommendations:**
- **Quick experimentation**: Start with TextRank (fastest, no setup)
- **Best free quality (short articles)**: Pegasus (up to ~750 words)
- **Best free quality (long articles)**: LED or LongT5 (handles multi-page documents)
- **Production (best quality/cost)**: Gemini Flash (fastest, cheapest) or Claude Sonnet 4 (highest quality)
- **Comparison**: Run all methods on a sample to evaluate for your use case

## Dashboard Features

### 📊 Coverage Tab
- **Version-independent** - shows all articles in the database
- Article volume by source (bar chart)
- Coverage timeline (daily article counts)

### 🏷️ Topics Tab
- **Independent version selector** - select from topic versions only
- Create new topic versions directly in the tab
- Top 20 topics by article count
- Topic-source heatmap showing coverage distribution
- Topic focus comparison across sources
- Selection bias indicators (topics with highest coverage variance)
- Interactive BERTopic visualizations (topic similarity map, hierarchical clustering)

### 📰 Events Tab
- **Independent version selector** - select from clustering versions only
- Create new clustering versions directly in the tab
- Browse top event clusters
- See which sources covered each event
- View all articles in an event cluster
- Multi-source event analysis

### 📝 Summaries Tab
- **Independent version selector** - select from summarization versions only
- Create new summarization versions directly in the tab
- View article summaries with original text comparison
- Statistics: compression ratio, word count, processing time
- Filter by source and search by title
- Compare summarization methods (TextRank, LexRank, BART, T5, Pegasus, LED, LongT5, BigBird-Pegasus, Claude, GPT)
- Source-level compression and performance metrics

### 😊 Sentiment Tab
- **Multi-model comparison** - compare results from different sentiment models
- View sentiment distribution by source
- Article-level sentiment scores with confidence metrics
- Filter by model type and news source
- Sentiment trends over time

## Embedding Models

The project supports multiple embedding models for generating article embeddings used in topic analysis and event clustering.

### Current Default: all-mpnet-base-v2
- **Dimensions**: 768
- **Size**: ~420MB
- **Languages**: English primarily
- **Quality**: Excellent for English text

### Alternative: EmbeddingGemma
- **Model**: `google/embeddinggemma-300m`
- **Dimensions**: 768 (default), or 512/256/128 with Matryoshka truncation
- **Size**: ~308MB
- **Languages**: 100+ multilingual support
- **Task-optimized**: Uses task-specific prompts for better clustering/classification
- **Access**: Gated model - requires HuggingFace authentication (see setup below)

**Key Features of EmbeddingGemma:**
- **Task Prompts**: Automatically uses task-specific prompts based on analysis type
  - Topics analysis → "task: classification"
  - Clustering analysis → "task: clustering"
  - Summarization → "task: retrieval"
- **Matryoshka Representation Learning**: Can truncate to 512, 256, or 128 dimensions while maintaining reasonable quality (trade quality for speed/storage)
- **Multilingual**: Future-proof if analyzing Sinhala/Tamil news articles

**To use EmbeddingGemma:**

0. **Authenticate with HuggingFace** (required for gated model):
   ```bash
   # Install huggingface-cli if not already installed
   pip install huggingface-hub

   # Login to HuggingFace
   huggingface-cli login

   # Accept the model license at:
   # https://huggingface.co/google/embeddinggemma-300m
   ```

1. Update `config.yaml`:
   ```yaml
   embeddings:
     provider: local
     model: google/embeddinggemma-300m
     task: null  # auto-detect from analysis type (topics→classification, clustering→clustering)
     matryoshka_dim: null  # or 512, 256, 128 for smaller dimensions
   ```

2. Create a new version to test it:
   ```python
   from src.versions import create_version, get_default_topic_config

   config = get_default_topic_config()
   config['embeddings']['model'] = 'google/embeddinggemma-300m'

   version_id = create_version(
       name="embeddinggemma-topics",
       description="Testing EmbeddingGemma for topic analysis",
       configuration=config,
       analysis_type='topics'
   )
   ```

3. Run the pipeline as usual:
   ```bash
   python3 scripts/embeddings/01_generate_embeddings.py --model google/embeddinggemma-300m
   PYTHONHASHSEED=42 python3 scripts/topics/02_discover_topics.py --version-id <version-id>
   ```

**Testing Matryoshka Truncation:**
```python
# Create version with 256-dimensional embeddings
config = get_default_topic_config()
config['embeddings']['model'] = 'google/embeddinggemma-300m'
config['embeddings']['matryoshka_dim'] = 256  # Reduce from 768 to 256

version_id = create_version(
    name="embeddinggemma-256d",
    description="EmbeddingGemma with 256 dimensions (Matryoshka truncation)",
    configuration=config,
    analysis_type='topics'
)
```

**Comparison Strategy:**
The version system makes it easy to A/B test different embedding models:
1. Keep baseline version with `all-mpnet-base-v2`
2. Create new version with `google/embeddinggemma-300m`
3. Run both pipelines and compare results in the dashboard
4. Evaluate topic quality, coverage, and clustering performance

## Configuration

### config.yaml

```yaml
# LLM Configuration (for future tone/type analysis)
llm:
  provider: claude  # claude | openai | gemini | local
  model: claude-sonnet-4-20250514
  temperature: 0.0

# Embeddings
embeddings:
  provider: local  # local (free) | openai
  model: all-mpnet-base-v2  # all-mpnet-base-v2 | google/embeddinggemma-300m
  batch_size: 64

  # EmbeddingGemma-specific options (ignored for other models)
  task: null                 # auto-detect from analysis type, or specify: clustering | classification | retrieval
  matryoshka_dim: null       # null (use 768) or 512 | 256 | 128 for smaller dimensions

# Topic Modeling
topics:
  min_topic_size: 10  # Minimum articles per topic
  diversity: 0.5      # Keyword diversity

# Clustering
clustering:
  similarity_threshold: 0.8   # Cosine similarity threshold
  time_window_days: 7         # Only cluster articles within 7 days
  min_cluster_size: 2         # Minimum articles per cluster

# Summarization
summarization:
  method: textrank             # textrank | lexrank | bart | t5 | pegasus | led | longt5 | bigbird-pegasus | claude | gpt | gemini
  summary_length: medium       # short | medium | long
  short_sentences: 3
  short_words: 50
  medium_sentences: 5
  medium_words: 100
  long_sentences: 8
  long_words: 150
  max_input_length: 1024       # Max tokens for standard models (ignored for long-context models)
  chunk_long_articles: true    # Split articles longer than max_input_length
  llm_model: claude-sonnet-4-20250514
  llm_temperature: 0.0
```

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

## Troubleshooting

### Dashboard not loading
```bash
# Check if Streamlit is running
pgrep -f streamlit

# Restart dashboard
pkill -f streamlit
streamlit run dashboard/Home.py
```

### Database connection issues
```bash
# Test connection
PGPASSWORD='your_password' psql -h localhost -U your_db_user -d your_database -c "SELECT COUNT(*) FROM your_schema.news_articles;"

# Check pgvector extension
psql -d your_database -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Embedding generation fails
```bash
# Check available memory (embeddings use ~2GB RAM)
free -h

# Process in smaller batches (edit src/embeddings.py)
# Change batch_size from 100 to 50
```

## Research Methodology

Based on: "The Media Bias Detector: A Framework for Annotating and Analyzing the News at Scale" (UPenn, 2025)

### Adapted for Sri Lankan Context
- ❌ **Skipped**: Political lean (Democrat/Republican) - not applicable to SL politics
- ✅ **Kept**: Topic hierarchy via data-driven discovery
- ✅ **Kept**: Event clustering for coverage comparison
- ✅ **Kept**: Selection bias analysis (topic coverage patterns)
- ⏸️ **Future**: Framing bias analysis (requires tone scoring via LLM)

## License & Attribution

Based on the Media Bias Detector framework from University of Pennsylvania.
Adapted for Sri Lankan newspaper analysis (2025).

## Contact

For questions or issues, refer to this documentation or check the code comments.
