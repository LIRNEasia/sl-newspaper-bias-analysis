# Sri Lanka Media Bias Detector

A dashboard for analyzing media bias in Sri Lankan English newspapers by examining coverage patterns, topic distribution, and event clustering.

## Overview

This project analyzes 8,365 articles from 4 Sri Lankan newspapers (Daily News, The Morning, Daily FT, The Island) covering November-December 2025 to identify:
- **Selection bias**: Which topics each source covers (or ignores)
- **Coverage patterns**: How different sources cover the same events
- **Topic discovery**: Data-driven topic categorization using BERTopic

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
│   ├── topics/            # Topic analysis pipeline
│   │   ├── 01_generate_embeddings.py
│   │   └── 02_discover_topics.py
│   └── clustering/        # Event clustering pipeline
│       ├── 01_generate_embeddings.py
│       └── 02_cluster_events.py
└── dashboard/
    └── app.py             # Streamlit dashboard (separate version selectors per tab)
```

## Database Schema

### Original Data
- `news_articles` - Scraped newspaper articles (8,365 articles)

### Result Versioning
- `result_versions` - Configuration-based version tracking for reproducible analysis
  - **NEW:** `analysis_type` column distinguishes between 'topics', 'clustering', and 'combined' versions
  - Topics and clustering are now **independent analyses** with separate versions
  - Same version name can exist for both topics and clustering (e.g., "baseline-topics" and "baseline-clustering")

### Analysis Tables
- `embeddings` - Article embeddings (768-dim vectors from all-mpnet-base-v2)
- `topics` - Discovered topics (linked to topic versions)
- `article_analysis` - Article-topic assignments
- `event_clusters` - Event clusters (linked to clustering versions)
- `article_clusters` - Article-to-cluster mappings

Note: All analysis tables are linked to `result_versions` for configuration tracking and reproducibility. Topics and clustering can now be run independently without interfering with each other.

## Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Architecture | ✅ Decoupled | Topics and clustering are independent analyses |
| Result Versioning | ✅ Complete | Decoupled version management with analysis_type |
| Topic Analysis | ✅ Complete | Independent pipeline with separate versions |
| Event Clustering | ✅ Complete | Independent pipeline with separate versions |
| Embeddings | ✅ Complete | Shared across both analyses (can use different models) |
| Tone Analysis | ⏸️ Skipped | Requires LLM API (Claude/OpenAI) |
| Article Type | ⏸️ Skipped | Requires LLM API |
| Dashboard | ✅ Running | Streamlit app with per-tab version selectors |

## Key Findings

### Topics Discovered
- **Top topics**: Sri Lanka politics, flooding/disasters, sports (netball, cricket), education, economy
- **Coverage**: 77% of articles successfully categorized into topics
- **Outliers**: 1,910 articles (23%) don't fit into specific topics

### Event Clusters
- **Total clusters**: 1,717 events
- **Multi-source coverage**: 87% of clusters covered by 2+ sources
- **Average cluster size**: 4.4 articles
- **Top event**: UN allocates $4.5M for Sri Lanka disaster relief (72 articles, 4 sources)

### Major Events (Nov-Dec 2025)
1. Cyclone Ditwah aftermath - 56 articles across all sources
2. Economic crisis response - 56 articles
3. Disaster relief fundraising - 47 articles
4. Weather warnings and flooding - multiple clusters

## Running the Project

### Prerequisites
```bash
# Database
PostgreSQL 16 with pgvector extension

# Python environment
Python 3.11+
pip install -r requirements.txt
```

### Database Connection
Update `config.yaml` with your database credentials:
```yaml
database:
  host: localhost
  port: 5432
  name: your_database
  schema: your_schema
  user: your_db_user
  password: "YOUR_PASSWORD"
```

### Running the Pipeline

**Important**: Topics and clustering are now **independent analyses** with separate pipelines. Each has its own version management.

#### Architecture: Two Independent Pipelines

```
Topics Analysis              Clustering Analysis
└── Topic Versions           └── Clustering Versions
    ├── Embeddings               ├── Embeddings
    └── Topic Discovery          └── Event Clustering
```

Both analyses depend on embeddings, but not on each other.

#### Topic Analysis Pipeline

**1. Create a Topic Version**
```bash
# Via Python
python3 -c "
from src.versions import create_version, get_default_topic_config
version_id = create_version('baseline-topics', 'Initial topic analysis', get_default_topic_config(), analysis_type='topics')
print(f'Version ID: {version_id}')
"
```

Or create via the dashboard (Topics tab → "➕ Create New Topics Version").

**2. Generate Embeddings**
```bash
python3 scripts/topics/01_generate_embeddings.py --version-id <version-id>
```
- Uses local `all-mpnet-base-v2` model (free, no API needed)
- Takes ~30 minutes for 8,365 articles on CPU
- Stores 768-dimensional vectors in PostgreSQL

**3. Discover Topics**
```bash
# For reproducible results, set PYTHONHASHSEED
PYTHONHASHSEED=42 python3 scripts/topics/02_discover_topics.py --version-id <version-id>
```
- Uses BERTopic with UMAP + HDBSCAN clustering
- Discovers topics automatically from data
- Generates keyword-based topic labels
- Takes ~2-3 minutes
- **Reproducible**: Set `PYTHONHASHSEED=42` to ensure identical results across runs with the same configuration
- **Model Storage**: Automatically saves model to database for team collaboration
  - Models stored as compressed archives (~6-8 MB each) in PostgreSQL
  - No local filesystem storage - keeps your workspace clean
  - Enables visualizations to work on any machine with database access
  - See `migrations/README.md` for migration instructions

#### Event Clustering Pipeline

**1. Create a Clustering Version**
```bash
# Via Python
python3 -c "
from src.versions import create_version, get_default_clustering_config
version_id = create_version('baseline-clustering', 'Initial clustering analysis', get_default_clustering_config(), analysis_type='clustering')
print(f'Version ID: {version_id}')
"
```

Or create via the dashboard (Events tab → "➕ Create New Clustering Version").

**2. Generate Embeddings**
```bash
python3 scripts/clustering/01_generate_embeddings.py --version-id <version-id>
```
- Same embedding process as topics
- Can use different embedding models if needed

**3. Cluster Events**
```bash
python3 scripts/clustering/02_cluster_events.py --version-id <version-id>
```
- Groups similar articles using cosine similarity (threshold: 0.8)
- Applies 7-day time window constraint
- Takes ~10 minutes

#### Run Dashboard
```bash
streamlit run dashboard/app.py
```
- Access at: http://localhost:8501
- **Topics tab**: Select topic version independently
- **Events tab**: Select clustering version independently
- Each tab manages its own versions
- Auto-refreshes when code changes

### Sharing the Dashboard

**Local access via SSH tunnel:**
```bash
# On your local machine
ssh -L 8501:localhost:8501 amanda@server-ip
# Then open http://localhost:8501
```

**Public access via Cloudflare Tunnel:**
```bash
./cloudflared-linux-amd64 tunnel --url http://localhost:8501
# Provides a public https:// URL
```

## Result Versioning System

The project supports **decoupled result versioning** for reproducible analysis. Topics and clustering are now independent analyses with separate version management.

### Key Concepts

- **Analysis Types**: Each version has an `analysis_type` - either 'topics', 'clustering', or 'combined' (legacy)
- **Independence**: Topic and clustering versions don't interfere with each other
- **Same Names OK**: You can have "baseline" for topics AND "baseline" for clustering
- **Separate Configurations**: Topic versions only track topic/embedding config, clustering versions only track clustering/embedding config

### Why Decoupled Versions?

- **Independence**: Change topic parameters without re-running clustering (and vice versa)
- **Efficiency**: No wasted computation - only re-run what actually changed
- **Clarity**: Each analysis has its own versions, making experiments easier to track
- **Reproducibility**: Still tracks exactly which configuration produced which results

### Creating Versions

**Option 1: Via Dashboard**

**Topics Tab:**
1. Click "➕ Create New Topics Version"
2. Enter a name (e.g., "baseline", "small-topics")
3. Optionally add a description
4. Edit the JSON configuration (only embeddings + topics)
5. Click "Create Version"

**Events Tab:**
1. Click "➕ Create New Clustering Version"
2. Enter a name (e.g., "baseline", "high-similarity")
3. Optionally add a description
4. Edit the JSON configuration (only embeddings + clustering)
5. Click "Create Version"

**Option 2: Programmatically**

**For Topics:**
```python
from src.versions import create_version, get_default_topic_config

config = get_default_topic_config()
config['topics']['min_topic_size'] = 15  # Modify as needed

version_id = create_version(
    name="small-topics",
    description="Smaller topic size for more granular categorization",
    configuration=config,
    analysis_type='topics'
)
print(f"Created version: {version_id}")
```

**For Clustering:**
```python
from src.versions import create_version, get_default_clustering_config

config = get_default_clustering_config()
config['clustering']['similarity_threshold'] = 0.85  # Modify as needed

version_id = create_version(
    name="high-similarity",
    description="Testing stricter clustering with 0.85 threshold",
    configuration=config,
    analysis_type='clustering'
)
print(f"Created version: {version_id}")
```

### Running Pipeline for a Version

After creating a version, run the appropriate pipeline:

**For Topic Versions:**
```bash
python3 scripts/topics/01_generate_embeddings.py --version-id <version-id>
python3 scripts/topics/02_discover_topics.py --version-id <version-id>
```

**For Clustering Versions:**
```bash
python3 scripts/clustering/01_generate_embeddings.py --version-id <version-id>
python3 scripts/clustering/02_cluster_events.py --version-id <version-id>
```

The dashboard automatically detects versions and shows pipeline completion status in each tab.

### Version Configuration

**Topic Versions Track:**
- `random_seed`: Ensures reproducible topic modeling
- `embeddings.model`: Embedding model used
- `topics.min_topic_size`: Minimum articles per topic
- `topics.diversity`: Keyword diversity in topic labels
- `topics.umap` & `topics.hdbscan`: UMAP/HDBSCAN parameters

**Clustering Versions Track:**
- `random_seed`: Ensures reproducible clustering
- `embeddings.model`: Embedding model used
- `clustering.similarity_threshold`: Cosine similarity threshold for event clustering
- `clustering.time_window_days`: Time constraint for clustering
- `clustering.min_cluster_size`: Minimum articles per cluster

### Best Practices

1. **Baseline versions first**: Create a "baseline" version for both topics and clustering with default settings
2. **Descriptive names**: Use names that explain what changed (e.g., "small-topics", "high-similarity-clustering")
3. **One parameter at a time**: When experimenting, change one parameter per version for clear comparisons
4. **Document reasoning**: Use the description field to explain why you're testing these parameters
5. **Independent experimentation**: Feel free to create multiple topic versions without worrying about clustering (and vice versa)

### Managing Versions

The project includes a CLI tool for managing versions:

**List all versions:**
```bash
python3 scripts/manage_versions.py list

# Filter by analysis type
python3 scripts/manage_versions.py list --type topics
python3 scripts/manage_versions.py list --type clustering
```

**View version statistics:**
```bash
python3 scripts/manage_versions.py stats <version-id>
```

**Delete a version (interactive with confirmation):**
```bash
python3 scripts/manage_versions.py delete <version-id>
```

The delete command will:
1. Show version details and analysis type
2. Display statistics of what will be deleted (embeddings, topics, clusters, etc.)
3. Ask you to type the version name to confirm
4. Ask for final confirmation by typing 'DELETE'
5. Cascade delete all associated results

**Important Notes:**
- Deletion is **permanent and cannot be undone**
- Original articles in `news_articles` table are **never deleted**
- Only analysis results (embeddings, topics, clusters) are removed
- Cascade deletion automatically removes all related records from:
  - `embeddings` table
  - `topics` table
  - `article_analysis` table
  - `event_clusters` table
  - `article_clusters` table
  - `word_frequencies` table (if applicable)

**Programmatic deletion (Python):**
```python
from src.versions import delete_version_interactive, delete_version

# Interactive deletion with confirmation prompts
delete_version_interactive(version_id)

# Direct deletion (use with caution - no confirmation)
success = delete_version(version_id)
```

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

## Configuration

### config.yaml

```yaml
# LLM Configuration (for future tone/type analysis)
llm:
  provider: claude  # claude | openai | local
  model: claude-sonnet-4-20250514
  temperature: 0.0

# Embeddings
embeddings:
  provider: local  # local (free) | openai
  model: all-mpnet-base-v2
  dimensions: 768

# Topic Modeling
topics:
  min_topic_size: 10  # Minimum articles per topic
  diversity: 0.5      # Keyword diversity

# Clustering
clustering:
  similarity_threshold: 0.8   # Cosine similarity threshold
  time_window_days: 7         # Only cluster articles within 7 days
  min_cluster_size: 2         # Minimum articles per cluster
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
streamlit run dashboard/app.py
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
