# DITWAH Claims Analysis - Quick Start Guide

## 🎯 What's New

The DITWAH claims analysis now uses a **two-step process** with **zero-cost local LLM** (Ollama):

1. **Step 1:** Generate individual claims (one per article)
2. **Step 2:** Cluster similar claims into general claims (max ~40)
3. **Step 3:** Analyze sentiment and stance for each general claim

**Benefits:**
- ✅ **$0 cost** - uses local Llama 3.1 via Ollama
- ✅ Better claim quality (LLM focuses on one article at a time)
- ✅ Controlled number of general claims (~40 max)
- ✅ Better clustering (automatic grouping of similar claims)
- ✅ Traceability (see which articles contribute to each claim)

---

## 📋 Prerequisites

### 1. Run Database Migration

```bash
# Apply the new schema changes
PGPASSWORD='<YOUR_PASSWORD>' psql -h localhost -U ai_agent -d taf_media -f migrations/001_add_article_claims_table.sql
```

**Verify migration succeeded:**
```sql
-- Should return 1 row
SELECT COUNT(*) FROM information_schema.tables
WHERE table_schema = 'media_bias' AND table_name = 'ditwah_article_claims';
```

### 2. Ensure Ollama is Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start it
ollama serve

# Verify Llama 3.1 is downloaded
ollama list | grep llama3.1

# If not downloaded, pull it
ollama pull llama3.1:latest
```

### 3. Ensure DITWAH Articles are Marked

```bash
# Check if DITWAH articles are marked
psql -h localhost -U ai_agent -d taf_media -c "SELECT COUNT(*) FROM media_bias.news_articles WHERE is_ditwah_cyclone = TRUE;"

# If count is 0, run the marking script
python3 scripts/ditwah_claims/01_mark_ditwah_articles.py
```

---

## 🚀 Running the Pipeline

### Step 0: Create a Version

```bash
python3 -c "
from src.versions import create_version, get_default_ditwah_claims_config
version_id = create_version(
    name='ditwah-v1',
    description='DITWAH claims analysis with local LLM',
    configuration=get_default_ditwah_claims_config(),
    analysis_type='ditwah_claims'
)
print(f'Version ID: {version_id}')
"
```

**Copy the version ID** - you'll need it for the next steps.

### Step 1: Generate Individual Claims (1-2 hours)

```bash
# Replace <version-id> with your actual version ID
python3 scripts/ditwah_claims/02_generate_individual_claims.py --version-id <version-id>
```

**What this does:**
- Processes each DITWAH article
- Sends article to local LLM (Llama 3.1)
- Generates ONE specific claim per article
- Stores in `ditwah_article_claims` table

**Expected output:**
```
✅ Generated 78 individual claims from 80 articles
Success rate: 97.5%
```

### Step 2: Cluster Claims into General Claims (10-15 minutes)

```bash
python3 scripts/ditwah_claims/03_cluster_claims.py --version-id <version-id>
```

**What this does:**
- Generates embeddings for individual claims (fast, local)
- Clusters similar claims using hierarchical clustering
- Generates ~40 general claims using LLM
- Links individual claims to general claims

**Expected output:**
```
✅ Created 35 clusters
✅ Generated 33 general claims
Average claims per cluster: 2.4
```

### Step 3: Analyze Sentiment & Stance (1-2 hours)

```bash
python3 scripts/ditwah_claims/04_analyze_sentiment_stance.py --version-id <version-id>
```

**What this does:**
- Links existing sentiment data (from previous sentiment analysis)
- Generates stance analysis (agree/disagree/neutral) using LLM
- Stores results in `claim_sentiment` and `claim_stance` tables

**Expected output:**
```
✅ Linked 280 sentiment records
✅ Total stance records created: 280
```

---

## 📊 Viewing Results in Dashboard

### 1. Start the Dashboard

```bash
streamlit run dashboard/app.py
```

### 2. Navigate to DITWAH Claims Tab

- Open http://localhost:8501
- Click on "🌀 Ditwah Claims" tab
- Select your version from the dropdown

### 3. Search and Explore Claims

**Search by keyword:**
- Type "government" → see all claims about government response
- Type "aid" → see claims about humanitarian aid
- Type "damage" → see infrastructure/economic damage claims

**Select a claim:**
- Choose from dropdown
- See two main visualizations:

#### Sentiment Distribution (100% Stacked Bar)
- Shows % of articles by sentiment category
- Very Negative (dark red) → Very Positive (dark green)
- Compare how each newspaper source feels about the claim

#### Stance Distribution (100% Stacked Bar)
- Shows % of articles by stance
- Disagree (red) → Neutral (yellow) → Agree (green)
- Compare which sources support or oppose the claim

**Sample Articles:**
- See actual articles mentioning this claim
- View sentiment and stance scores

---

## 🔧 Troubleshooting

### Issue: "No individual claims found"

**Solution:**
```bash
# Check if step 1 was completed
psql -h localhost -U ai_agent -d taf_media -c "
SELECT COUNT(*) FROM media_bias.ditwah_article_claims WHERE result_version_id = '<version-id>';
"

# If 0, re-run step 1
python3 scripts/ditwah_claims/02_generate_individual_claims.py --version-id <version-id>
```

### Issue: "LLM connection error"

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Restart Ollama
pkill ollama
ollama serve

# Verify model is available
ollama list
ollama run llama3.1:latest "Test prompt"
```

### Issue: "Pipeline is slow"

**Expected times with local LLM:**
- Step 1 (Individual claims): 1-2 hours
- Step 2 (Clustering): 10-15 minutes
- Step 3 (Sentiment & Stance): 1-2 hours
- **Total: 2.5-4 hours**

**To speed up:**
1. Use GPU if available (10x faster)
2. Use smaller model: `llama3.2:3b` (3x faster, slightly lower quality)
3. Run overnight

### Issue: "Dashboard shows wrong data"

**Solution:**
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Restart dashboard
pkill -f streamlit
streamlit run dashboard/app.py
```

---

## 📈 Performance Comparison

| Aspect | Paid API (Mistral) | **Local LLM (Llama 3.1)** |
|--------|-------------------|--------------------------|
| **Cost** | $5-10 per run | **$0** ✅ |
| **Time** | 30-60 min | 2-4 hours |
| **Quality** | Excellent | Very Good |
| **Privacy** | Data sent to API | **100% local** ✅ |

---

## 🎯 Next Steps

1. **Run the pipeline** on your DITWAH data
2. **Explore the dashboard** to see claim distributions
3. **Compare sources** - which newspapers are most supportive/critical?
4. **Refine clustering** - adjust `max_general_claims` in config.yaml if needed
5. **Share results** - export visualizations or share dashboard URL

---

## 📝 Configuration Tweaks

### To get fewer general claims (e.g., 25 instead of 40):

Edit `config.yaml`:
```yaml
ditwah_claims:
  generation:
    max_general_claims: 25  # Change from 40 to 25
```

### To use a faster model:

Edit `config.yaml`:
```yaml
ditwah_claims:
  llm:
    model: llama3.2:3b  # Faster, smaller model
```

Then re-run steps 1-3.

---

## 🆘 Need Help?

- Check `migrations/README.md` for schema migration details
- Check `CLAUDE.md` for overall project documentation
- View logs in pipeline output for detailed error messages

**Common files:**
- `src/ditwah_claims.py` - Core claim generation functions
- `dashboard/app.py` - Dashboard visualization code
- `config.yaml` - Configuration settings
- `schema.sql` - Database schema
