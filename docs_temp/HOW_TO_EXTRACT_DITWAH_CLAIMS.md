# How to Extract Claims from Ditwah Articles Using Mistral AI

## Quick Start

### 1. Set up Mistral API Key

Create a `.env` file in the project root:

```bash
echo "MISTRAL_API_KEY=your_mistral_api_key_here" >> .env
```

Or export it temporarily:

```bash
export MISTRAL_API_KEY=your_mistral_api_key_here
```

### 2. Test Mistral Connection

```bash
python3 test_mistral.py
```

Expected output:
```
✅ API key found
✅ Basic text generation works
✅ JSON mode works
✅ ALL TESTS PASSED
```

## Option 1: Simple Example (Recommended for Learning)

Use this to understand the core logic:

```bash
python3 simple_ditwah_claims_example.py
```

**What it does:**
1. Queries database for articles where `is_ditwah_cyclone = 1`
2. Sends article summaries to Mistral AI
3. Extracts 15 claims like "How government took actions on disaster"
4. Saves results to `ditwah_claims.json`

**Output:**
```
CLAIMS EXTRACTED:
==================================================

1. [government_response]
   The government deployed 500 military personnel for rescue operations
   Confidence: 0.85

2. [humanitarian_aid]
   UN allocated $4.5 million for disaster relief
   Confidence: 0.92

3. [infrastructure_damage]
   Major roads and bridges were damaged in southern provinces
   Confidence: 0.78
...
```

## Option 2: Full Featured Script

For production use with logging and error handling:

```bash
python3 extract_ditwah_claims_mistral.py
```

**Features:**
- ✅ Comprehensive logging
- ✅ Error handling with retries (for rate limits)
- ✅ Groups claims by category
- ✅ Saves to JSON file
- ✅ Shows token usage

## Option 3: Full Pipeline (Database Integration)

For integration with the existing dashboard and version management:

```bash
# Step 1: Create a version
python3 -c "
from src.versions import create_version, get_default_ditwah_claims_config
version_id = create_version(
    'baseline-ditwah',
    'Ditwah claims analysis with Mistral',
    get_default_ditwah_claims_config(),
    analysis_type='ditwah_claims'
)
print(f'Version ID: {version_id}')
"

# Step 2: Run the pipeline
python3 scripts/ditwah_claims/02_generate_claims.py --version-id <version-id>
```

**What the full pipeline does:**
1. Generates claims (same as above)
2. **Also** analyzes sentiment for each claim
3. **Also** analyzes stance (agree/disagree/neutral)
4. Stores everything in the database
5. Makes it available in the dashboard

## Understanding the Prompt

The key to getting good claims is the prompt. Here's what makes it work:

```python
prompt = f"""Analyze these {len(articles)} articles about Cyclone Ditwah.

Extract {num_claims} specific, verifiable claims like:
- How the government took actions on the disaster
- What humanitarian aid was provided
- Infrastructure damage and casualties

Instructions:
1. Claims should be SPECIFIC and VERIFIABLE
   ✅ "Government deployed 500 troops"
   ❌ "Government responded well"

2. Claims should appear in MULTIPLE articles (3-4+)

3. Categorize each claim:
   - government_response
   - humanitarian_aid
   - infrastructure_damage
   - economic_impact
   - international_response
   - casualties_and_displacement

Return JSON:
[
  {{
    "claim_text": "...",
    "claim_category": "...",
    "confidence": 0.85
  }}
]
"""
```

## Customizing the Claims

### Extract More/Fewer Claims

Edit the script:

```python
claims = extract_claims_with_mistral(articles, num_claims=20)  # Change to 20
```

### Add New Categories

Edit the `categories` list:

```python
categories = [
    "government_response",
    "humanitarian_aid",
    "infrastructure_damage",
    "economic_impact",
    "international_response",
    "casualties_and_displacement",
    "media_coverage",          # NEW
    "public_reaction"          # NEW
]
```

### Focus on Specific Topics

Modify the prompt:

```python
prompt = f"""...

Focus ONLY on claims about government actions:
- Emergency response measures
- Relief distribution
- Military deployment
- Government statements

...
"""
```

## Troubleshooting

### "No Ditwah articles found"

Make sure articles are marked in the database:

```bash
python3 scripts/ditwah_claims/01_mark_ditwah_articles.py
```

Or check manually:

```sql
SELECT COUNT(*) FROM media_bias.news_articles WHERE is_ditwah_cyclone = 1;
```

### "MISTRAL_API_KEY not found"

```bash
# Check if set
echo $MISTRAL_API_KEY

# Set it
export MISTRAL_API_KEY=your_key_here

# Or add to .env file
echo "MISTRAL_API_KEY=your_key_here" >> .env
```

### Rate Limit Errors

The Mistral client automatically retries with exponential backoff:
- 1st retry: wait 1 second
- 2nd retry: wait 2 seconds
- 3rd retry: wait 4 seconds
- 4th retry: wait 8 seconds
- 5th retry: wait 16 seconds

If you still hit limits, reduce the number of claims or wait a bit between runs.

### Invalid JSON Response

The script uses `json_mode=True` which forces Mistral to return valid JSON. If you still get errors:

1. Check the raw response in the logs
2. Try reducing `num_claims` (fewer claims = simpler JSON)
3. Try increasing `temperature` to 0.3 (slightly more creative)

## Example Claims Generated

Here are some example claims the system might extract:

**Government Response:**
- "The government established emergency relief centers in affected districts"
- "President declared a state of emergency in flood-affected areas"
- "Military deployed 500 personnel for rescue and relief operations"

**Humanitarian Aid:**
- "UN allocated $4.5 million for disaster relief efforts"
- "Red Cross distributed food and water to 10,000 families"
- "Foreign countries pledged millions in aid"

**Infrastructure Damage:**
- "Over 500 homes were destroyed by flooding"
- "Major roads and bridges in southern provinces were damaged"
- "Power outages affected thousands of households"

**Casualties:**
- "Cyclone Ditwah resulted in 15 confirmed deaths"
- "Over 10,000 families were displaced"
- "Hundreds injured in storm-related incidents"

## API Costs

Mistral AI pricing (as of 2025):
- **mistral-large-latest**: $0.002 per 1K input tokens, $0.006 per 1K output tokens

For this use case:
- Input: ~50K tokens (all articles)
- Output: ~2K tokens (15 claims)
- **Cost per run: ~$0.10 - $0.20**

Very affordable for this type of analysis!

## Next Steps

After extracting claims, you can:

1. **Manually review** them in `ditwah_claims.json`
2. **Run sentiment analysis** to see how positive/negative each claim is
3. **Run stance analysis** to see which sources agree/disagree with each claim
4. **Visualize** in the dashboard (if using full pipeline)

For full analysis with stance and sentiment, use the complete pipeline:

```bash
python3 scripts/ditwah_claims/02_generate_claims.py --version-id <version-id>
```

Then view in the dashboard at the "🌀 Ditwah Claims" tab.
