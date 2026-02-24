# Claim Page Interpretation Enhancements

## What's New

I've enhanced the claim interpretations to provide richer, more narrative analysis of how different newspapers cover claims. The enhancements include:

### 1. Individual Source Narratives

**For Sentiment:**
Each newspaper now gets a personalized narrative like:
- "**The Morning** expresses predominantly negative sentiment (68.5%) about the claim that 'The Sri Lankan government has taken adequate measures...'"
- "**Daily Mirror** maintains largely neutral coverage (55.2%) when reporting on the claim that..."

**For Stance:**
Each newspaper gets stance-specific narratives:
- "**The Morning** predominantly disagrees (72.3%) with the claim that 'The Sri Lankan government has taken adequate measures...'"
- "**Ada Derana** moderately agrees (45.7%) with the claim that..."

### 2. Cross-Source Comparison

**Sentiment Comparisons:**
- **Unanimous patterns**: "All sources express predominantly negative sentiment about this claim, indicating a consensus in negative perception"
- **Divergent patterns**: "The Morning is significantly more negative (70.5%) compared to Daily Mirror (25.3% negative), revealing different emotional framings"
- **Mixed patterns**: "Ada Derana maintains balanced emotional tone, while The Morning uses more emotionally charged language"

**Stance Comparisons:**
- **Unanimous agreement**: "All sources predominantly agree with this claim, demonstrating consensus"
- **Unanimous disagreement**: "All sources predominantly disagree with this claim, showing clear consensus in rejecting its validity"
- **Clear divide**: "The Morning and Daily Mirror tend to support it, while Ada Derana and News First tend to oppose it, revealing polarized media positions"
- **Three-way split**: "Sources are divided—some agree, some disagree, while others remain neutral"
- **Stark contrast**: "The Morning shows 75.5% agreement compared to Daily Mirror's 15.2% agreement, highlighting substantial differences"

### 3. Intensity Levels

Interpretations now use intensity descriptors based on percentage:
- **70%+**: "overwhelmingly"
- **50-69%**: "predominantly"
- **30-49%**: "moderately"
- **<30%**: "somewhat"

Example: "The Morning overwhelmingly disagrees (85.2%) with the claim..."

## How It Appears in the Dashboard

### Sentiment Analysis Section
```
📊 Sentiment Distribution
[Visualization]

💡 Most positive coverage: Daily Mirror (65.3% positive) | Most negative coverage: The Morning (70.1% negative)

📖 Detailed Sentiment Analysis (expandable)
  └─ Overall Sentiment Landscape: [aggregate stats]
  └─ Individual Source Sentiment:
      ├─ The Morning expresses predominantly negative sentiment (70.1%) about the claim that "..."
      ├─ Daily Mirror expresses moderately positive sentiment (45.2%) about the claim that "..."
      └─ Ada Derana maintains largely neutral coverage (55.6%) when reporting on the claim that "..."
  └─ Cross-Source Comparison:
      └─ Divergent Sentiment: The Morning is significantly more negative compared to Daily Mirror...
  └─ [Additional insights]
```

### Stance Analysis Section
```
⚖️ Stance Distribution
[Visualization]

💡 Most supportive: Daily Mirror (58.3% agree) | Most critical: The Morning (75.2% disagree)

📖 Detailed Stance Analysis (expandable)
  └─ Overall Stance Patterns: [aggregate stats]
  └─ Individual Source Stance:
      ├─ The Morning predominantly disagrees (75.2%) with the claim that "..."
      ├─ Daily Mirror moderately agrees (58.3%) with the claim that "..."
      └─ Ada Derana maintains a neutral position (62.1%) on the claim that "..."
  └─ Cross-Source Comparison:
      └─ Clear Divide: Daily Mirror tends to support it, while The Morning tends to oppose it...
  └─ [Additional insights]
```

## Key Features

✅ **Personalized narratives** for each newspaper source
✅ **Direct reference** to the actual claim text in interpretations
✅ **Comparative analysis** showing consensus vs. divergence
✅ **Natural language** descriptions (e.g., "The Morning feels...", "Daily Mirror disagrees...")
✅ **Intensity qualifiers** (overwhelmingly, predominantly, moderately, somewhat)
✅ **Pattern detection** (unanimous, divided, three-way split, stark contrast)

## Example Output

For a claim: "The Sri Lankan government has taken adequate measures to support Cyclone Ditwah victims"

**Individual Narratives:**
- **The Morning** overwhelmingly disagrees (82.5%) with the claim that "The Sri Lankan government has taken adequate measures to support Cyclone Ditwah victims"
- **Daily Mirror** moderately agrees (48.3%) with the claim that "The Sri Lankan government has taken adequate measures to support Cyclone Ditwah victims"
- **Ada Derana** maintains a neutral position (65.2%) on the claim that "The Sri Lankan government has taken adequate measures to support Cyclone Ditwah victims"

**Comparison:**
"Sources are divided on this claim. Daily Mirror tends to support it, while The Morning tends to oppose it, revealing polarized media positions."

## Files Modified

- `dashboard/components/interpretations.py`:
  - Added `_generate_individual_source_sentiment_narratives()`
  - Added `_generate_sentiment_comparison()`
  - Added `_generate_individual_source_stance_narratives()`
  - Added `_generate_stance_comparison()`
  - Added `_get_intensity_level()` helper function
  - Enhanced `generate_sentiment_interpretation()` and `generate_stance_interpretation()`

No changes needed to the page file—the enhancements automatically work with the existing structure!
