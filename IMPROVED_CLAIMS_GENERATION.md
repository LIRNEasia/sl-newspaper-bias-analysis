# Improved Claims Generation - Documentation

## Overview
This document describes the refinements made to the Ditwah claims generation process to produce better, more meaningful claims with at least 20 articles per general claim.

## Changes Made

### 1. Refined Individual Claim Generation Prompt

**Previous Approach:**
- Asked for "specific and factual" claims with exact numbers
- Examples: "Rs. 5 billion allocated", "15 deaths and 50,000 displaced"
- **Problem**: Too specific, leading to many unique claims that don't cluster well

**New Approach:**
- Asks for **thematic claims** that capture the main assertion
- Focuses on broader themes rather than specific numbers
- Examples: "Government response was inadequate", "Severe humanitarian crisis"
- **Benefit**: Individual claims are more likely to cluster together, creating larger groups

**Key Improvements:**
```
✓ Focus on PRIMARY THEME rather than specific details
✓ Thematic enough to apply to multiple similar articles
✓ Still verifiable and debatable
✓ Avoids overly specific numbers unless they're the main point
```

### 2. Enhanced General Claim Synthesis Prompt

**Previous Approach:**
- Basic instruction to "capture the common theme"
- Limited guidance on how to synthesize

**New Approach:**
- Shows up to 15 example individual claims (increased from 10)
- Emphasizes finding the "COMMON ASSERTION" across all claims
- Asks: "What is the ONE main point that all these articles are making?"
- Provides clearer examples of good general claims
- Instructs to be neutral in tone

**Key Improvements:**
```
✓ Better guidance on synthesizing multiple claims
✓ More context for the LLM (15 examples instead of 10)
✓ Emphasis on neutrality and verifiability
✓ Clearer instructions on balancing breadth and meaningfulness
```

### 3. Optimized Clustering Configuration

**Previous Settings:**
```yaml
max_general_claims: 40
min_cluster_size: 3
similarity_threshold: 0.75
min_articles_per_claim: 3
```

**New Settings:**
```yaml
max_general_claims: 15      # Reduced from 40
min_cluster_size: 20        # Increased from 3
similarity_threshold: 0.65  # Reduced from 0.75 (more lenient clustering)
min_articles_per_claim: 20  # Increased from 3
```

**Rationale:**
- **Fewer general claims (15)**: Creates larger, more meaningful clusters
- **Higher min_cluster_size (20)**: Ensures each general claim has substantial coverage
- **Lower similarity threshold (0.65)**: More lenient clustering groups related claims together
- **Higher min_articles_per_claim (20)**: Enforces the requirement for substantial article coverage

### 4. Thematic Claim Categories

The prompt now explicitly guides the LLM to focus on these themes:
- Government response effectiveness/inadequacy
- Scale of humanitarian impact
- International aid provision or absence
- Infrastructure damage severity
- Economic consequences
- Relief efforts speed/quality

## Expected Outcomes

### Before Changes:
- ~40 general claims with varying article counts (some with only 3-5 articles)
- Many overly specific claims that are hard to compare across sources
- Fragmented analysis with limited data per claim

### After Changes:
- ~10-15 general claims, each with 20+ articles
- Thematic claims that capture broader narratives
- Richer analysis with more data points per claim
- Better cross-source comparisons
- More meaningful bias detection

## Example Comparison

### Before (Too Specific):
```
Individual Claims:
1. "Government allocated Rs. 5 billion for relief"
2. "Government announced Rs. 5B fund for victims"
3. "Rs. 5 billion released for cyclone relief"

General Claim: "Government allocated Rs. 5 billion for Cyclone Ditwah relief"
Articles: 8
```

### After (Thematic):
```
Individual Claims:
1. "Government response to Cyclone Ditwah was inadequate"
2. "Government failed to provide timely relief"
3. "Authorities slow to respond to cyclone victims"
4. "Government relief efforts criticized for being insufficient"
... (20+ similar thematic claims)

General Claim: "Government response to Cyclone Ditwah was inadequate and poorly coordinated"
Articles: 25+
```

## How to Use

1. **Mark Ditwah Articles:**
   ```bash
   python3 scripts/ditwah_claims/01_mark_ditwah_articles.py
   ```

2. **Generate Individual Claims (with new prompt):**
   ```bash
   python3 scripts/ditwah_claims/02_generate_individual_claims.py --version-id <uuid>
   ```

3. **Cluster into General Claims (with new config):**
   ```bash
   python3 scripts/ditwah_claims/03_cluster_claims.py --version-id <uuid>
   ```

4. **Analyze Sentiment & Stance:**
   ```bash
   python3 scripts/ditwah_claims/04_analyze_sentiment_stance.py --version-id <uuid>
   ```

## Monitoring Results

After regenerating claims, check:
- **Number of general claims**: Should be ~10-15 (not 40)
- **Articles per claim**: Each should have 20+ articles
- **Claim quality**: Should be thematic and broadly applicable
- **Clustering effectiveness**: Similar themes should be grouped together

## Notes

- The improved prompts work best with capable LLMs (Llama 3.1+, Claude, GPT-4)
- Lower-quality LLMs may struggle with thematic abstraction
- You can adjust `similarity_threshold` if clusters are too large/small (range: 0.5-0.8)
- If you have fewer than 300 articles total, consider reducing `min_cluster_size` to 10-15

## Files Modified

1. `src/ditwah_claims.py`:
   - `generate_individual_claim_for_article()` - Enhanced prompt
   - `generate_general_claim_from_cluster()` - Enhanced prompt

2. `config.yaml`:
   - `ditwah_claims.generation.max_general_claims`: 40 → 15
   - `ditwah_claims.generation.min_articles_per_claim`: 3 → 20
   - `ditwah_claims.clustering.min_cluster_size`: 3 → 20
   - `ditwah_claims.clustering.similarity_threshold`: 0.75 → 0.65

---

**Created**: 2026-02-16
**Version**: 1.0
