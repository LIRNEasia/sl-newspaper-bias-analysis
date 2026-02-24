# Claims Page Interpretation Enhancements

## Overview
Enhanced the **Media Bias Analysis: Combined Sentiment & Stance Insights** section in the Ditwah Claims page to provide comprehensive, narrative-driven interpretations that combine both sentiment and stance analysis for each newspaper source.

## What Was Changed

### 1. Enhanced `interpretations.py`

#### New Function: `_generate_combined_source_narratives()`
**Purpose:** Generates individual narrative interpretations for each newspaper that combine both sentiment AND stance in natural language.

**Example Output:**
```
📰 Individual Newspaper Analysis:

**The Morning** feels **predominantly negative** (65.3%) about the claim that "The Sri Lankan government has taken swift action in responding to Cyclone Ditwah...". At the same time, **The Morning** **predominantly disagrees** (58.2%) with this claim.

**Daily News** feels **moderately positive** (42.1%) about the claim that "The Sri Lankan government has taken swift action in responding to Cyclone Ditwah...". At the same time, **Daily News** **moderately agrees** (45.8%) with this claim.

**Sunday Times** maintains **neutral emotional tone** (55.4%) when discussing the claim that "The Sri Lankan government has taken swift action in responding to Cyclone Ditwah...". However, **Sunday Times** takes a **neutral factual position** (62.3%), neither clearly supporting nor opposing it.
```

**Key Features:**
- Combines sentiment percentages with stance percentages for each source
- Uses intensity descriptors (overwhelmingly, predominantly, moderately, somewhat)
- Includes the actual claim text in the narrative
- Shows both emotional framing AND factual position together

---

#### New Function: `_generate_combined_source_comparison()`
**Purpose:** Generates cross-source comparative analysis showing how newspapers differ in their combined sentiment-stance profiles.

**Example Outputs:**

**Scenario 1 - Unanimous Agreement:**
```
🔍 Cross-Source Comparative Analysis:

🤝 Unanimous Negative Rejection: All sources (The Morning, Daily News, Sunday Times) express predominantly negative sentiment AND disagree with this claim, showing complete consensus in both emotional tone and factual position. This represents the strongest form of media alignment against the claim.
```

**Scenario 2 - Divided Coverage:**
```
🔍 Cross-Source Comparative Analysis:

⚖️ Divided Coverage: Sources show different sentiment-stance combinations. **The Morning** (negative + disagree), **Daily News** (positive + agree), **Sunday Times** (neutral on both). This diversity reveals significant disagreement in how newspapers perceive and frame this claim.

📊 Sentiment vs Stance Split: **The Morning** is most negative emotionally (68.5%), but **Island** disagrees most factually (72.1%), showing that emotional tone doesn't always match factual position.

🔥 High Polarization: Sources show extreme variation (sentiment range: 45.3%, stance range: 52.7%), indicating deeply divided media coverage with no consensus on how to perceive or evaluate this claim.
```

**Key Features:**
- Identifies unanimous patterns (all sources agree/disagree)
- Groups sources by their sentiment-stance profiles
- Highlights contradictions (e.g., positive but disagree, negative but agree)
- Calculates polarization indices showing media division
- Compares most extreme sources across both dimensions

---

### 2. Updated `generate_combined_bias_interpretation()`
- Added `claim_text` parameter to enable claim-specific narratives
- Integrated the two new analysis functions at the beginning of the output
- Reordered sections to show:
  1. **Overall Bias Landscape** - aggregate summary
  2. **📰 Individual Newspaper Analysis** - NEW detailed per-source narratives
  3. **🔍 Cross-Source Comparative Analysis** - NEW comparative insights
  4. **Bias Through Framing** - sentiment-stance alignment patterns
  5. **Source Bias Profiles** - classification of editorial styles
  6. **Editorial Strategy** - variance analysis

---

### 3. Updated `8_Ditwah_Claims.py`
- Modified line 401 to pass `claim['claim_text']` to `generate_combined_bias_interpretation()`
- This enables the new narrative functions to reference the specific claim being analyzed

---

## How It Works

### Data Flow
```
sentiment_df + stance_df + claim_text
           ↓
generate_combined_bias_interpretation()
           ↓
  _generate_combined_source_narratives()  → Individual newspaper narratives
           ↓
  _generate_combined_source_comparison()  → Cross-newspaper comparisons
           ↓
  [existing analysis functions...]
           ↓
     Complete interpretation text
```

### Sentiment-Stance Profiles
The system now classifies each source into one of these profiles:

1. **Negative + Disagree** - Emotionally critical AND factually rejects claim
2. **Positive + Agree** - Emotionally supportive AND factually accepts claim
3. **Negative + Agree** - Acknowledges claim but frames negatively
4. **Positive + Disagree** - Rejects claim but frames positively (softens criticism)
5. **Neutral + Neutral** - Balanced factual reporting

### Comparison Patterns
The system detects these patterns across sources:

- **🤝 Unanimous patterns** - All sources align (e.g., all negative + disagree)
- **⚖️ Divided coverage** - Sources split into different profiles
- **📊 Sentiment vs Stance Split** - Most emotional ≠ most factual source
- **🔥 High Polarization** - Large variance indicating no consensus
- **🤝 Strong Consensus** - Small variance indicating agreement

---

## Benefits

### For Users
1. **Clear Individual Insights**: Each newspaper's position is explained in plain language
2. **Easy Comparison**: Immediately see which sources agree/disagree and why
3. **Bias Detection**: Identify when emotional tone doesn't match factual position
4. **Consensus Detection**: Understand if media agrees or is divided on a claim

### For Analysis
1. **Holistic View**: Sentiment and stance analyzed together, not separately
2. **Narrative Format**: Natural language instead of just percentages
3. **Contextual**: References the actual claim being discussed
4. **Actionable**: Clear patterns help understand media landscape

---

## Example Use Cases

### Use Case 1: Government Response Claims
**Claim:** "The Sri Lankan government responded swiftly to Cyclone Ditwah"

**Analysis Output:**
- Shows which newspapers are critical (negative + disagree)
- Shows which are supportive (positive + agree)
- Identifies government-aligned vs opposition-aligned media
- Reveals if any sources support claim but with negative framing

### Use Case 2: Casualty Count Claims
**Claim:** "Cyclone Ditwah caused 50 deaths"

**Analysis Output:**
- Shows if all sources agree on facts (unanimous stance)
- Shows if sentiment differs despite factual agreement
- Identifies which sources are most emotional vs most neutral
- Reveals media polarization levels

### Use Case 3: International Aid Claims
**Claim:** "International community provided insufficient aid"

**Analysis Output:**
- Groups newspapers by their stance (some agree, some disagree)
- Shows emotional framing differences
- Identifies nationalist vs internationalist perspectives
- Reveals editorial strategy differences

---

## Testing

### To test the enhancements:

1. **Start the dashboard:**
   ```bash
   cd /home/ranushi/Taf_claude/sl-newspaper-bias-analysis/dashboard
   streamlit run Home.py
   ```

2. **Navigate to Claims page:**
   - Go to "🌀 Ditwah Claims" in the sidebar
   - Select a version with claim data

3. **Select a claim:**
   - Choose a claim with multiple newspaper sources covering it

4. **Scroll to "🎯 Media Bias Analysis":**
   - You should now see:
     - **📰 Individual Newspaper Analysis** section with detailed narratives for each source
     - **🔍 Cross-Source Comparative Analysis** section showing how sources differ

5. **Verify the narratives:**
   - Check that each newspaper has a combined sentiment + stance narrative
   - Check that comparisons show patterns like "unanimous," "divided," etc.
   - Check that the claim text appears in the narratives

---

## Technical Notes

### Column Naming
After merging sentiment and stance dataframes, columns are suffixed:
- `neutral_pct` → `neutral_pct_sent` (from sentiment_df)
- `neutral_pct` → `neutral_pct_stance` (from stance_df)

The new functions correctly handle these suffixed column names.

### Intensity Levels
Uses the existing `_get_intensity_level()` helper:
- ≥70%: "overwhelmingly"
- ≥50%: "predominantly"
- ≥30%: "moderately"
- <30%: "somewhat"

### Claim Text Handling
- Long claims (>80 chars) are truncated with "..."
- Short claims are shown in full
- Claim text is optional (functions work without it)

---

## Future Enhancements (Optional)

1. **Visual Sentiment-Stance Matrix**: 2D chart showing sources positioned by sentiment (x-axis) and stance (y-axis)

2. **Temporal Analysis**: Show how a newspaper's sentiment-stance profile changes over time for the same claim

3. **Source Credibility Scoring**: Weight narratives by source reliability metrics

4. **Claim Category Patterns**: Show if certain newspapers consistently disagree with government-related claims vs disaster-related claims

5. **Export Functionality**: Allow exporting these interpretations as PDF reports

---

## Files Modified

- ✅ `dashboard/components/interpretations.py` - Added 2 new functions, updated 1 function
- ✅ `dashboard/pages/8_Ditwah_Claims.py` - Updated to pass claim_text parameter
- ✅ `CLAIM_INTERPRETATION_ENHANCEMENTS.md` - Created this documentation

---

## Summary

The enhanced interpretations now provide **human-readable, claim-specific narratives** that combine sentiment and stance analysis for each newspaper source, along with **comprehensive cross-source comparisons** showing patterns of agreement, division, and polarization. This makes it much easier to understand media bias patterns at a glance.
