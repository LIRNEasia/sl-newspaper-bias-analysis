# Ditwah Claims Analysis - Quick Start Guide

## Prerequisites

1. **Mistral API Key** - Sign up at https://console.mistral.ai/
2. **Python dependencies installed** - `pip install -r requirements.txt`
3. **Ditwah articles marked** - Run `01_mark_ditwah_articles.py` first

## Step 1: Configure API Key

```bash
# Create .env file from template
cp .env.example .env

# Edit .env and add your Mistral API key
nano .env  # or use your favorite editor
```

Add this line:
```
MISTRAL_API_KEY=your_actual_api_key_here
```

## Step 2: Test Mistral Integration

```bash
python3 test_mistral.py
```

You should see:
```
✅ API key found
✅ Basic text generation works
✅ JSON mode works
✅ ALL TESTS PASSED
```

## Step 3: Create a Version

**Option A: Via Python**

```bash
python3 -c "
from src.versions import create_version, get_default_ditwah_claims_config

version_id = create_version(
    name='baseline-ditwah-mistral',
    description='Baseline Ditwah claims analysis using Mistral AI',
    configuration=get_default_ditwah_claims_config(),
    analysis_type='ditwah_claims'
)

print(f'✅ Version created: {version_id}')
"
```

**Option B: Via Dashboard**

```bash
streamlit run dashboard/app.py
```

1. Go to "🌀 Ditwah Claims" tab
2. Click "➕ Create New Claims Version"
3. Enter name: `baseline-ditwah-mistral`
4. Add description (optional)
5. Review/edit configuration (optional)
6. Click "Create Version"

## Step 4: Mark Ditwah Articles

```bash
# Only needed if not done before
python3 scripts/ditwah_claims/01_mark_ditwah_articles.py
```

This identifies articles about Cyclone Ditwah and sets `is_ditwah_cyclone = 1`.

## Step 5: Run the Pipeline

```bash
# Replace <version-id> with the UUID from Step 3
python3 scripts/ditwah_claims/02_generate_claims.py --version-id <version-id>
```

**What it does:**
1. Filters Ditwah articles (~56 articles)
2. Generates 15 claims using Mistral AI
3. For each claim:
   - Finds matching articles
   - Fetches sentiment scores (from existing data)
   - Analyzes stance with Mistral AI
4. Stores everything in database

**Expected runtime:** 5-10 minutes

**Expected output:**
```
✅ Stored 15 claims to database
✅ Stored 250+ sentiment records
✅ Stored 250+ stance records
✅ Pipeline complete
```

## Step 6: View Results

```bash
# Start dashboard (if not already running)
streamlit run dashboard/app.py
```

**Navigate to "🌀 Ditwah Claims" tab:**

1. Select your version from dropdown
2. Use search to filter claims (e.g., "government", "aid")
3. Click on a claim to see:
   - Sentiment by source (bar chart with error bars)
   - Stance by source (average stance + distribution)
   - Sample articles with scores and supporting quotes

## Verify Database

```bash
psql -h localhost -U your_db_username -d taf_media
```

```sql
-- Check claims
SELECT COUNT(*) FROM media_bias.ditwah_claims
WHERE result_version_id = '<your-version-id>';

-- Check sentiment
SELECT COUNT(*) FROM media_bias.claim_sentiment cs
JOIN media_bias.ditwah_claims dc ON cs.claim_id = dc.id
WHERE dc.result_version_id = '<your-version-id>';

-- Check stance
SELECT COUNT(*) FROM media_bias.claim_stance cs
JOIN media_bias.ditwah_claims dc ON cs.claim_id = dc.id
WHERE dc.result_version_id = '<your-version-id>';

-- View sample claims
SELECT claim_text, claim_category, article_count
FROM media_bias.ditwah_claims
WHERE result_version_id = '<your-version-id>'
ORDER BY article_count DESC
LIMIT 5;
```

## Configuration Options

Edit `config.yaml` to customize:

```yaml
ditwah_claims:
  generation:
    num_claims: 15  # How many claims to generate
    categories:     # Claim categories
      - government_response
      - humanitarian_aid
      - infrastructure_damage
      - economic_impact
      - international_response
      - casualties_and_displacement

  llm:
    provider: mistral
    model: mistral-large-latest
    temperature: 0.3  # Higher = more creative

  stance:
    batch_size: 5  # Articles per LLM call (lower = less rate limiting)
```

## Troubleshooting

### Rate Limiting

**Problem:** See many "Rate limit hit, retrying..." messages

**Solution:**
- Pipeline will automatically retry (up to 5 times)
- Reduce `batch_size` in config.yaml
- Wait a few minutes between runs

### No Claims Generated

**Problem:** "No claims generated" error

**Check:**
1. Ditwah articles marked? `SELECT COUNT(*) FROM news_articles WHERE is_ditwah_cyclone = 1`
2. API key valid? `echo $MISTRAL_API_KEY`
3. Check logs for API errors

### Dashboard Empty

**Problem:** Dashboard shows no data

**Check:**
1. Version complete? Check pipeline_status in database
2. Data exists? Run SQL queries above
3. Right version selected in dropdown?
4. Clear cache: Dashboard → hamburger menu → Clear cache

### API Errors

**Problem:** "MISTRAL_API_KEY not found"

**Solution:**
```bash
# Check if .env file exists
ls -la .env

# Load environment variables
source .env  # or restart your terminal

# Or export directly
export MISTRAL_API_KEY=your_key_here
```

## Cost Estimate

**Mistral Large pricing** (as of Jan 2025):
- Input: $2 per 1M tokens
- Output: $6 per 1M tokens

**For baseline run (~56 articles, 15 claims):**
- Claims generation: ~20K input, ~2K output = $0.05
- Stance analysis: ~150K input, ~15K output = $0.39
- **Total: ~$0.44 per run**

## Next Steps

### Experiment with Parameters

Create new versions with different settings:

**More granular claims:**
```python
config = get_default_ditwah_claims_config()
config['generation']['num_claims'] = 25  # More claims
create_version('granular-claims', '25 claims for finer analysis', config, 'ditwah_claims')
```

**Different LLM temperature:**
```python
config = get_default_ditwah_claims_config()
config['llm']['temperature'] = 0.0  # More deterministic
create_version('deterministic', 'Lower temperature for consistency', config, 'ditwah_claims')
```

### Compare Results

Use the dashboard to compare different versions side-by-side and see how parameters affect claim generation and stance detection.

## Support

For issues or questions:
1. Check `DITWAH_CLAIMS_IMPLEMENTATION.md` for detailed architecture
2. Check logs in terminal output
3. Verify database contents with SQL queries above
