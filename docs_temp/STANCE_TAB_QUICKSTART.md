# Stance Distribution Tab - Quick Start Guide

## Where to Find It

1. Open your browser to: **http://localhost:8501**
2. Look for the navigation buttons at the top
3. Click the **⚖️ Stance** button (third row, leftmost button)

## What You'll See

### Step 1: Select a Version
```
┌─────────────────────────────────────────────────────────┐
│ ⚖️ Stance Distribution Analysis                        │
│                                                          │
│ Version: [ditwah-v1 ▼]  [➕ Create New Version]        │
└─────────────────────────────────────────────────────────┘
```

### Step 2: Overview Metrics (Section 1)
```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Total Claims │Most Controve │Strongest Con │Avg Confidence│
│      29      │    1.414     │    0.003     │    85.2%     │
│              │"Doppler..."  │"Government..." │             │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### Step 3: Polarization Dashboard (Section 2)
```
📊 Claim Polarization Dashboard
Visualize which claims generate agreement vs. disagreement

┌─────────────────────────────────────────────────────────┐
│ Filter by Category: [All ▼]  Min Articles: [2 ────]    │
│ Sort by: [Controversy (High) ▼]  Show Top N: [20 ──]   │
└─────────────────────────────────────────────────────────┘

          HEATMAP (Claim × Source)
     Daily News  The Morning  Daily FT  The Island
Claim 1   🟢         🟢         🟢        🟢      (All agree)
Claim 2   🔴         🟡         🟢        🟡      (Mixed)
Claim 3   🔴         🔴         🟢        🟢      (Polarized)
...

Color Scale: 🔴 Disagree (-1) → 🟡 Neutral (0) → 🟢 Agree (+1)
```

### Step 4: Source Alignment Matrix (Section 3)
```
🤝 Source Alignment Matrix
See which news sources tend to agree or disagree

         Daily News  The Morning  Daily FT  The Island
Daily News    100%      59.2%      63.8%     57.5%
The Morning            100%       59.6%     55.8%
Daily FT                          100%      58.1%
The Island                                  100%

Higher % = More agreement between sources
```

### Step 5: Confidence Explorer (Section 4)
```
🎯 Confidence-Weighted Stance Explorer

Minimum Confidence: [50% ────────] Category: [All ▼]

     BUBBLE CHART

  High Controversy
      ╱│╲
     ○ │ ●  ← Size = Article count
    ╱  │  ╲    Color = Confidence
   ────┼────  X-axis = Stance (-1 to +1)
       │
  Low Controversy

Quadrant Analysis:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ High Agree + │ High Agree + │High Disagree │High Disagree │
│Controversial │  Consensus   │Controversial │  Consensus   │
│      5       │      8       │      3       │      4       │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### Step 6: Claim Deep Dive (Section 5)
```
🔍 Claim Deep Dive
Select a claim to see detailed analysis

┌─────────────────────────────────────────────────────────┐
│ Select a claim: [Search or select...             ▼]    │
└─────────────────────────────────────────────────────────┘

Selected: "Government is taking steps to address Cyclone Ditwah aftermath"

Category: government  |  Total Articles: 12  |  Sources: 4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Stance Distribution by Source

Daily News:    [████████ Agree 80%   ][█ N 10%][█ D 10%]
The Morning:   [█████ Agree 50%      ][███ Neutral 30%][██ Disagree 20%]
Daily FT:      [██████ Agree 60%     ][██ Neutral 20%][██ Disagree 20%]
The Island:    [███████ Agree 70%    ][██ Neutral 20%][█ Disagree 10%]

┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Source       │ Agree        │ Neutral      │ Disagree     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Daily News   │ 8            │ 1            │ 1            │
│ The Morning  │ 6            │ 3            │ 2            │
│ Daily FT     │ 6            │ 2            │ 2            │
│ The Island   │ 7            │ 2            │ 1            │
└──────────────┴──────────────┴──────────────┴──────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sub-tabs:
┌───────────────┬───────────────┬───────────────┐
│📝 Supporting  │📰 Article     │⚖️ Source      │
│   Quotes      │   List        │  Comparison   │
└───────────────┴───────────────┴───────────────┘

[📝 Supporting Quotes Tab - ACTIVE]

### ✅ Strongly Agree (4 articles)

▶ Daily News - "Government announces relief package for..."
  > "The government has allocated Rs. 5 billion for immediate relief"
  > "President emphasized the importance of swift response"
  Published: 2025-11-28 | Stance Score: 0.85

▶ The Morning - "Relief efforts underway after Cyclone Ditwah"
  > "Officials confirmed the deployment of emergency teams"
  > "Relief packages are being distributed to affected areas"
  Published: 2025-11-29 | Stance Score: 0.90

### ✅ Agree (8 articles)
[Expand to see more...]

### ⚖️ Neutral (6 articles)
[Expand to see more...]

### ❌ Disagree (2 articles)
[Expand to see more...]
```

## Common Workflows

### Workflow 1: Find Most Controversial Claims
1. Go to **Section 2: Polarization Dashboard**
2. Keep sort as "Controversy (High)"
3. Look at top claims in the list
4. Click claim text to auto-scroll to Deep Dive

### Workflow 2: Compare Sources
1. Go to **Section 3: Source Alignment Matrix**
2. Identify source pairs with low alignment (< 55%)
3. Note which sources frequently disagree
4. Go to Deep Dive and select claims to see specific differences

### Workflow 3: Analyze Single Claim
1. Scroll to **Section 5: Claim Deep Dive**
2. Use search box to find claim by keyword
3. Select claim from dropdown
4. Review stance distribution across sources
5. Click through sub-tabs:
   - **Quotes**: See actual evidence
   - **Article List**: Browse all articles
   - **Source Comparison**: Visual comparison

### Workflow 4: High-Confidence Claims Only
1. Go to **Section 4: Confidence Explorer**
2. Move "Minimum Confidence" slider to 80%
3. View filtered bubble chart
4. Click bubbles to see claim details
5. Focus on claims with high confidence + high controversy

## Keyboard Shortcuts

- **Ctrl+K** or **Cmd+K**: Focus search box
- **Tab**: Navigate between inputs
- **Enter**: Select from dropdown
- **Esc**: Close expanders/modals

## Tips

### 🎯 Best Practices
- Start with **Overview Metrics** to understand data scope
- Use **Polarization Dashboard** to identify interesting patterns
- **Source Alignment** reveals editorial relationships
- **Confidence Explorer** helps filter noise
- **Claim Deep Dive** provides detailed evidence

### ⚡ Performance Tips
- First load may take 2-3 seconds (data loading)
- Subsequent loads are instant (5-minute cache)
- Use filters to reduce data displayed
- Collapse expanders when not needed

### 🔍 Finding Specific Claims
Use the searchable dropdown in **Claim Deep Dive**:
- Type keywords: "government", "casualties", "aid"
- Dropdown filters as you type
- Shows category in parentheses for context

### 📊 Understanding Metrics
- **Controversy Index**: Higher = more disagreement (stddev of stance scores)
- **Alignment %**: Higher = more agreement between sources
- **Stance Score**: -1 (disagree) to +1 (agree), 0 = neutral
- **Confidence**: 0-100%, model's certainty in prediction

## Color Legend

### Stance Colors
- 🟢 **Green (#2D6A4F)**: Agree / Strongly Agree
- 🟡 **Yellow (#FFD93D)**: Neutral
- 🔴 **Red (#C9184A)**: Disagree / Strongly Disagree

### Source Colors
- 🔵 **Blue**: Daily News
- 🟠 **Orange**: The Morning
- 🟢 **Green**: Daily FT
- 🔴 **Red**: The Island

## Troubleshooting

### "No stance data found"
**Solution**: Run the stance analysis pipeline for this version first
```bash
python3 scripts/ditwah_claims/analyze_stance.py --version-id <version-id>
```

### Charts not loading
**Solution**:
1. Refresh page (Ctrl+R / Cmd+R)
2. Clear Streamlit cache (Press 'C' in browser, then "Clear cache")
3. Check browser console for errors

### Dropdown not searchable
**Solution**: Click inside dropdown and start typing - it auto-filters

### Slow performance
**Solution**:
1. Reduce "Show Top N Claims" slider in Polarization Dashboard
2. Apply category filter to reduce data
3. Increase minimum articles filter

## Data Interpretation

### What is "Controversial"?
A claim is controversial when sources disagree strongly:
- High controversy (>1.0): Some sources strongly agree, others disagree
- Medium controversy (0.5-1.0): Mixed opinions across sources
- Low controversy (<0.5): Most sources have similar stance
- Zero controversy (0.0): All sources agree completely

### What does Alignment % Mean?
Source alignment shows how often two sources take the same stance:
- 70-100%: Strong alignment (editorial similarity)
- 55-70%: Moderate alignment (some differences)
- <55%: Low alignment (frequent disagreements)

### How to Identify Bias?
Look for patterns:
1. **Selection bias**: One source consistently disagrees on specific categories
2. **Framing bias**: Same stance but different confidence levels
3. **Quote selection**: Check Supporting Quotes tab - do sources use different evidence?

## Example Analysis

### Scenario: Analyzing Government Response Claims

1. **Filter by category**: Select "government" in Polarization Dashboard
2. **Sort by controversy**: Keep "Controversy (High)" selected
3. **Identify pattern**: Do certain sources consistently agree/disagree?
4. **Deep dive**: Select top controversial claim
5. **Compare quotes**: Check Supporting Quotes tab
6. **Conclusion**: Are disagreements factual or interpretive?

### Scenario: Finding Source Pairs to Compare

1. Go to **Source Alignment Matrix**
2. Find lowest alignment pair (e.g., Daily News ↔ The Island: 57.5%)
3. Note the alignment percentage
4. Go to **Polarization Dashboard**
5. Filter to show only those sources (future feature)
6. Identify claims where they disagree most

## Next Steps

After exploring the Stance Distribution tab:

1. **Export findings**: Take screenshots or notes
2. **Compare versions**: Switch versions to see how stance changes
3. **Share insights**: Discuss patterns with team
4. **Investigate further**: Read full articles for controversial claims

## Need Help?

- **Documentation**: See `STANCE_TAB_IMPLEMENTATION.md` for technical details
- **Testing**: Run `python3 test_stance_tab.py` to verify data
- **Issues**: Check Streamlit terminal for error messages
- **Questions**: Review this guide or check the main `CLAUDE.md`

---

**Quick Reference Card**

| Section | Purpose | Key Action |
|---------|---------|------------|
| Overview | High-level stats | Check total claims & confidence |
| Polarization | Find controversy | Sort by controversy, filter categories |
| Alignment | Source relationships | Check alignment matrix |
| Confidence | Filter by quality | Adjust confidence slider |
| Deep Dive | Detailed analysis | Select claim, read quotes |

**Remember**: Start broad (Overview) → narrow down (Filters) → deep dive (Individual claims)
