# Ditwah Claims Implementation - Database-Based with Mistral AI

## Overview

This implementation adds Mistral AI support and converts the Ditwah claims analysis pipeline from CSV-based to database-based storage. The dashboard UI already has all necessary components and will automatically work with database-stored data.

## What Changed

### 1. Mistral AI Integration (`src/llm.py`)

**Added:**
- `MistralLLM` class with rate limit retry logic
- Exponential backoff for 429 errors (1s, 2s, 4s, 8s, 16s delays)
- JSON mode support for structured claim generation
- Token usage tracking

**Factory function updated:**
- `get_llm()` now supports `provider: mistral`

### 2. Database Storage (`src/ditwah_claims.py`)

**Removed:**
- CSV file generation
- Pandas DataFrame operations
- `OUTPUT_DIR` constant

**Added:**
- `store_claim_sentiment()` - Stores sentiment records with ON CONFLICT handling
- `store_claim_stance()` - Stores stance records with ON CONFLICT handling

**Updated:**
- `generate_claims_pipeline()` - Completely rewritten for database-first storage
  - Stores claims directly to `ditwah_claims` table
  - Stores sentiment to `claim_sentiment` table
  - Stores stance to `claim_stance` table
  - All operations are idempotent (can re-run safely)

### 3. Configuration (`config.yaml`)

**Added sections:**
```yaml
# Mistral AI configuration
mistral:
  model: mistral-large-latest
  temperature: 0.2
  max_tokens: 4096

# Ditwah Claims Analysis
ditwah_claims:
  generation:
    num_claims: 15
    min_articles_mentioning: 3
    categories: [...]
  llm:
    provider: mistral
    model: mistral-large-latest
  sentiment:
    primary_model: roberta
  stance:
    batch_size: 5
    temperature: 0.0
```

### 4. Version Management (`src/versions.py`)

**Updated:**
- `get_default_ditwah_claims_config()` - Now reads from `config.yaml`
- `update_pipeline_status()` - Supports `ditwah_claims` step
- Pipeline completion logic updated for `ditwah_claims` analysis type

### 5. Pipeline Script (`scripts/ditwah_claims/02_generate_claims.py`)

**Updated:**
- Reads configuration from new structure
- Updates pipeline status on completion
- Shows database storage info in summary

### 6. Dependencies (`requirements.txt`)

**Added:**
- `mistralai>=1.0.0`

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file (use `.env.example` as template):

```bash
cp .env.example .env
# Edit .env and add your Mistral API key
```

Add to `.env`:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Test Mistral Integration

```bash
python3 test_mistral.py
```

Expected output:
```
✅ API key found
✅ Basic text generation works
✅ JSON mode works
✅ Config-based loading works
✅ ALL TESTS PASSED
```

## Usage

### Create a Version

```bash
python3 -c "
from src.versions import create_version, get_default_ditwah_claims_config
version_id = create_version(
    'baseline-ditwah-mistral',
    'Baseline Ditwah claims analysis using Mistral AI',
    get_default_ditwah_claims_config(),
    analysis_type='ditwah_claims'
)
print(f'Version ID: {version_id}')
"
```

### Run the Pipeline

```bash
# 1. Mark Ditwah articles (if not already done)
python3 scripts/ditwah_claims/01_mark_ditwah_articles.py

# 2. Generate claims and analyze sentiment/stance
python3 scripts/ditwah_claims/02_generate_claims.py --version-id <version-id>
```

### Verify Results

**Check database:**
```sql
-- Claims stored
SELECT COUNT(*) FROM media_bias.ditwah_claims
WHERE result_version_id = '<version-id>';

-- Sentiment stored
SELECT COUNT(*) FROM media_bias.claim_sentiment cs
JOIN media_bias.ditwah_claims dc ON cs.claim_id = dc.id
WHERE dc.result_version_id = '<version-id>';

-- Stance stored
SELECT COUNT(*) FROM media_bias.claim_stance cs
JOIN media_bias.ditwah_claims dc ON cs.claim_id = dc.id
WHERE dc.result_version_id = '<version-id>';
```

**View in dashboard:**
```bash
streamlit run dashboard/app.py
```

Navigate to "🌀 Ditwah Claims" tab and select your version.

## Architecture

### Data Flow

```
1. Filter Articles
   └─> Ditwah articles (is_ditwah_cyclone = 1)

2. Generate Claims (LLM)
   └─> Store to ditwah_claims table

3. For each claim:
   ├─> Find matching articles (keyword-based)
   ├─> Fetch sentiment from sentiment_analyses table
   │   └─> Store to claim_sentiment table
   └─> Analyze stance with LLM
       └─> Store to claim_stance table

4. Update article counts
   └─> Update ditwah_claims.article_count
```

### Database Tables

**ditwah_claims**
- Stores generated claims
- Linked to result_version_id
- Tracks LLM provider and model

**claim_sentiment**
- Links claims to articles
- Stores sentiment scores (from existing sentiment analysis)
- One record per claim-article pair

**claim_stance**
- Links claims to articles
- Stores LLM-generated stance analysis
- Includes reasoning and supporting quotes
- One record per claim-article pair

### Idempotency

All database operations use `ON CONFLICT ... DO UPDATE`:
- Safe to re-run pipeline
- Updates existing records if configuration changes
- No duplicate entries

## Error Handling

### Mistral Rate Limiting

The implementation includes exponential backoff:
- 429 errors trigger automatic retries
- Wait times: 1s, 2s, 4s, 8s, 16s
- Max 5 retries before failing
- Clear logging of retry attempts

### Missing Data

- No Ditwah articles → Pipeline returns error, doesn't proceed
- No claims generated → Pipeline returns error
- No matching articles for claim → Log warning, continue with next claim
- Missing sentiment data → Log warning, skip article

### Database Errors

- Transactions with rollback on failure
- `ON CONFLICT` for idempotency
- Clear error logging

## Performance

### LLM API Calls

**Claims generation:** 1 call (all articles bundled)

**Stance analysis:** ~N/5 calls
- Batch size configurable (default: 5 articles per call)
- Example: 15 claims × 50 articles avg ÷ 5 batch = ~150 calls

**Estimated time:** 5-10 minutes (depends on rate limits)

### Database Operations

- Bulk inserts with `ON CONFLICT`
- Indexed on claim_id and source_id
- Dashboard queries cached (ttl=300s)

## Backward Compatibility

### Preserved

✅ All existing LLM providers (Claude, OpenAI, local)
✅ Dashboard loader functions (already query database)
✅ Version management
✅ Database schema
✅ Search functionality

### Migration Notes

- Old CSV files are not automatically imported
- Pipeline must be re-run to populate database
- This is acceptable since LLM output is non-deterministic anyway

## Troubleshooting

### Mistral API Key Issues

```bash
# Check if key is set
echo $MISTRAL_API_KEY

# Set for current session
export MISTRAL_API_KEY=your_key_here

# Or add to .env file
echo "MISTRAL_API_KEY=your_key_here" >> .env
```

### Rate Limiting

If you see many retry messages:
- Pipeline will automatically retry
- Consider reducing batch_size in config
- Wait between pipeline runs

### No Claims Generated

Check:
1. Are Ditwah articles marked? Run `01_mark_ditwah_articles.py`
2. Is LLM configured correctly? Test with `test_mistral.py`
3. Check logs for API errors

### Dashboard Not Showing Data

1. Verify data in database (see SQL queries above)
2. Check version is complete: `pipeline_status->>'ditwah_claims' = true`
3. Refresh dashboard page
4. Clear Streamlit cache (hamburger menu → Clear cache)

## Files Changed

| File | Changes |
|------|---------|
| `src/llm.py` | Added MistralLLM class + factory update |
| `src/ditwah_claims.py` | Rewritten for database storage |
| `config.yaml` | Added mistral + ditwah_claims config |
| `src/versions.py` | Updated default config + pipeline status |
| `scripts/ditwah_claims/02_generate_claims.py` | Updated for new config structure |
| `requirements.txt` | Added mistralai>=1.0.0 |
| `.env.example` | Created with MISTRAL_API_KEY |
| `test_mistral.py` | Created for testing integration |

## Next Steps

1. **Set up API key** - Add MISTRAL_API_KEY to .env
2. **Test integration** - Run `python3 test_mistral.py`
3. **Create version** - Use `create_version()` or dashboard
4. **Run pipeline** - Generate claims and analyze stance
5. **View results** - Check dashboard for visualizations

## Success Criteria

✅ Mistral AI calls succeed with retry logic
✅ Claims stored in database (no CSV files)
✅ Dashboard displays claims with search and visualizations
✅ Sentiment analysis by source shown in charts
✅ Stance analysis by source shown in charts
✅ All existing functionality still works
✅ Error handling works for rate limits and missing data
