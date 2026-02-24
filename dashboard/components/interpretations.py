"""
Graph interpretation generation for sentiment and stance analysis.

This module provides functions to generate natural language interpretations
of sentiment and stance distribution data, helping users understand patterns
and insights in the visualizations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# ============================================================================
# SENTIMENT INTERPRETATION
# ============================================================================

def generate_sentiment_interpretation(sentiment_df: pd.DataFrame, claim_text: str = "") -> str:
    """
    Generate a comprehensive interpretation of sentiment distribution data.

    Args:
        sentiment_df: DataFrame with columns:
            - source_name, total
            - very_negative_pct, negative_pct, neutral_pct, positive_pct, very_positive_pct
        claim_text: The actual claim being analyzed

    Returns:
        Markdown-formatted interpretation text
    """
    if sentiment_df.empty:
        return "_No sentiment data available for this claim._"

    # Convert percentage columns to float to avoid Decimal type issues
    pct_columns = ['very_negative_pct', 'negative_pct', 'neutral_pct', 'positive_pct', 'very_positive_pct']
    for col in pct_columns:
        if col in sentiment_df.columns:
            sentiment_df[col] = sentiment_df[col].astype(float)

    # Convert total to int to avoid Decimal type issues
    if 'total' in sentiment_df.columns:
        sentiment_df['total'] = sentiment_df['total'].astype(int)

    sections = []

    # Overall sentiment landscape
    overview = _analyze_sentiment_overview(sentiment_df, claim_text)
    if overview:
        sections.append(overview)

    # Individual source narratives (NEW)
    source_narratives = _generate_individual_source_sentiment_narratives(sentiment_df, claim_text)
    if source_narratives:
        sections.append(source_narratives)

    # Comparative analysis across sources (NEW)
    comparison = _generate_sentiment_comparison(sentiment_df, claim_text)
    if comparison:
        sections.append(comparison)

    # Source-specific extremes
    extremes = _analyze_sentiment_extremes(sentiment_df, claim_text)
    if extremes:
        sections.append(extremes)

    # Consensus vs divergence
    consensus = _analyze_sentiment_consensus(sentiment_df, claim_text)
    if consensus:
        sections.append(consensus)

    # Neutral coverage analysis
    neutral = _analyze_neutral_coverage(sentiment_df, claim_text)
    if neutral:
        sections.append(neutral)

    return "\n\n".join(sections)


def _analyze_sentiment_overview(df: pd.DataFrame, claim_text: str = "") -> str:
    """Analyze overall sentiment landscape."""
    # Calculate aggregate percentages
    total_articles = df['total'].sum()

    # Aggregate counts
    agg_very_neg = (df['very_negative_pct'] * df['total'] / 100).sum()
    agg_neg = (df['negative_pct'] * df['total'] / 100).sum()
    agg_neutral = (df['neutral_pct'] * df['total'] / 100).sum()
    agg_pos = (df['positive_pct'] * df['total'] / 100).sum()
    agg_very_pos = (df['very_positive_pct'] * df['total'] / 100).sum()

    # Calculate percentages
    total_negative_pct = (agg_very_neg + agg_neg) / total_articles * 100
    total_positive_pct = (agg_pos + agg_very_pos) / total_articles * 100
    total_neutral_pct = agg_neutral / total_articles * 100

    # Determine dominant sentiment
    sentiments = {
        'negative': total_negative_pct,
        'positive': total_positive_pct,
        'neutral': total_neutral_pct
    }
    dominant = max(sentiments, key=sentiments.get)

    # Build interpretation
    if dominant == 'negative':
        balance_text = f"predominantly **negative** ({total_negative_pct:.1f}% negative vs {total_positive_pct:.1f}% positive)"
    elif dominant == 'positive':
        balance_text = f"predominantly **positive** ({total_positive_pct:.1f}% positive vs {total_negative_pct:.1f}% negative)"
    else:
        balance_text = f"predominantly **neutral** ({total_neutral_pct:.1f}% neutral), with {total_negative_pct:.1f}% negative and {total_positive_pct:.1f}% positive"

    return f"**Overall Sentiment Landscape:** Across all sources covering this claim, sentiment is {balance_text}. A total of {int(total_articles)} articles were analyzed."


def _analyze_sentiment_extremes(df: pd.DataFrame, claim_text: str = "") -> str:
    """Identify sources with most positive/negative coverage."""
    # Calculate combined positive and negative percentages
    df = df.copy()
    df['total_positive'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative'] = df['negative_pct'] + df['very_negative_pct']

    # Find extremes
    most_positive = df.loc[df['total_positive'].idxmax()]
    most_negative = df.loc[df['total_negative'].idxmax()]

    parts = []

    # Most positive source
    if most_positive['total_positive'] > 15:  # Only mention if significant
        if claim_text:
            parts.append(
                f"**{most_positive['source_name']}** feels {most_positive['total_positive']:.1f}% positive "
                f"about the claim that \"{claim_text}\""
            )
        else:
            parts.append(
                f"**{most_positive['source_name']}** shows {most_positive['total_positive']:.1f}% positive sentiment"
            )

    # Most negative source
    if most_negative['total_negative'] > 15:  # Only mention if significant
        if claim_text:
            parts.append(
                f"**{most_negative['source_name']}** feels {most_negative['total_negative']:.1f}% negative "
                f"about the claim that \"{claim_text}\""
            )
        else:
            parts.append(
                f"**{most_negative['source_name']}** shows {most_negative['total_negative']:.1f}% negative sentiment"
            )

    if not parts:
        return ""

    return "**Source-Specific Insights:** " + " ".join(parts)


def _analyze_sentiment_consensus(df: pd.DataFrame, claim_text: str = "") -> str:
    """Analyze consensus vs divergence in sentiment."""
    # Calculate standard deviations
    pos_std = (df['positive_pct'] + df['very_positive_pct']).std()
    neg_std = (df['negative_pct'] + df['very_negative_pct']).std()

    avg_std = (pos_std + neg_std) / 2

    if avg_std < 15:
        pattern = "**Consensus Pattern:**"
        description = (
            f"Sources show relatively similar sentiment distributions "
            f"(average variation: {avg_std:.1f}%), indicating a shared perspective on this claim."
        )
    elif avg_std > 25:
        pattern = "**Divergence Pattern:**"
        description = (
            f"Sources display widely varying sentiment distributions "
            f"(average variation: {avg_std:.1f}%), indicating different editorial approaches "
            f"or interpretations of this claim."
        )
    else:
        pattern = "**Mixed Pattern:**"
        description = (
            f"Sources show moderate variation in sentiment "
            f"(average variation: {avg_std:.1f}%), with some consensus but also notable differences."
        )

    return f"{pattern} {description}"


def _analyze_neutral_coverage(df: pd.DataFrame, claim_text: str = "") -> str:
    """Identify sources with high neutral coverage."""
    high_neutral_threshold = 40

    neutral_sources = df[df['neutral_pct'] > high_neutral_threshold].sort_values(
        'neutral_pct', ascending=False
    )

    if neutral_sources.empty:
        return ""

    source_list = ", ".join(
        f"**{row['source_name']}** ({row['neutral_pct']:.1f}%)"
        for _, row in neutral_sources.iterrows()
    )

    return (
        f"**Neutral Coverage:** The following sources maintain notably neutral coverage: "
        f"{source_list}. This suggests more factual, less emotionally charged reporting on this claim."
    )


def _generate_individual_source_sentiment_narratives(df: pd.DataFrame, claim_text: str = "") -> str:
    """Generate narrative interpretations for each source's sentiment."""
    df = df.copy()
    df['total_positive'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative'] = df['negative_pct'] + df['very_negative_pct']

    narratives = []

    for _, row in df.iterrows():
        source = row['source_name']
        total_pos = row['total_positive']
        total_neg = row['total_negative']
        neutral = row['neutral_pct']

        # Determine dominant sentiment
        if total_neg > total_pos and total_neg > neutral:
            # Negative dominant
            intensity = _get_intensity_level(total_neg)
            if claim_text:
                narrative = (
                    f"**{source}** expresses {intensity} negative sentiment ({total_neg:.1f}%) "
                    f"about the claim that \"{claim_text[:80]}{'...' if len(claim_text) > 80 else ''}\""
                )
            else:
                narrative = f"**{source}** shows {intensity} negative coverage ({total_neg:.1f}%)"
        elif total_pos > total_neg and total_pos > neutral:
            # Positive dominant
            intensity = _get_intensity_level(total_pos)
            if claim_text:
                narrative = (
                    f"**{source}** expresses {intensity} positive sentiment ({total_pos:.1f}%) "
                    f"about the claim that \"{claim_text[:80]}{'...' if len(claim_text) > 80 else ''}\""
                )
            else:
                narrative = f"**{source}** shows {intensity} positive coverage ({total_pos:.1f}%)"
        else:
            # Neutral dominant
            if claim_text:
                narrative = (
                    f"**{source}** maintains largely neutral coverage ({neutral:.1f}%) "
                    f"when reporting on the claim that \"{claim_text[:80]}{'...' if len(claim_text) > 80 else ''}\""
                )
            else:
                narrative = f"**{source}** maintains neutral coverage ({neutral:.1f}%)"

        narratives.append(narrative)

    if not narratives:
        return ""

    return "**Individual Source Sentiment:**\n\n" + "\n\n".join(narratives)


def _generate_sentiment_comparison(df: pd.DataFrame, claim_text: str = "") -> str:
    """Generate comparative analysis showing how sources differ."""
    df = df.copy()
    df['total_positive'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative'] = df['negative_pct'] + df['very_negative_pct']

    comparisons = []

    # Check if all sources agree on sentiment direction
    all_negative = all(df['total_negative'] > df['total_positive'])
    all_positive = all(df['total_positive'] > df['total_negative'])
    all_neutral = all(df['neutral_pct'] > 50)

    if all_negative:
        comparisons.append(
            f"**Unanimous Negative Sentiment:** All sources ({', '.join(df['source_name'].tolist())}) "
            f"express predominantly negative sentiment about this claim, indicating a consensus in negative perception."
        )
    elif all_positive:
        comparisons.append(
            f"**Unanimous Positive Sentiment:** All sources ({', '.join(df['source_name'].tolist())}) "
            f"express predominantly positive sentiment about this claim, showing consensus in favorable perception."
        )
    elif all_neutral:
        comparisons.append(
            f"**Unanimous Neutral Stance:** All sources maintain neutral emotional tone, "
            f"suggesting factual reporting without emotional bias."
        )
    else:
        # Sources disagree - identify the differences
        most_negative = df.loc[df['total_negative'].idxmax()]
        most_positive = df.loc[df['total_positive'].idxmax()]

        if most_negative['total_negative'] - most_positive['total_negative'] > 20:
            comparisons.append(
                f"**Divergent Sentiment:** **{most_negative['source_name']}** is significantly more negative "
                f"({most_negative['total_negative']:.1f}%) compared to **{most_positive['source_name']}** "
                f"({most_positive['total_negative']:.1f}% negative), revealing different emotional framings of the same claim."
            )

        # Identify moderate vs extreme sources
        moderate_sources = df[(df['neutral_pct'] > 40)]
        extreme_sources = df[(df['total_negative'] > 50) | (df['total_positive'] > 50)]

        if not moderate_sources.empty and not extreme_sources.empty:
            moderate_list = ", ".join(moderate_sources['source_name'].tolist())
            extreme_list = ", ".join(extreme_sources['source_name'].tolist())
            comparisons.append(
                f"**Reporting Styles Differ:** {moderate_list} maintain(s) balanced emotional tone, "
                f"while {extreme_list} use(s) more emotionally charged language."
            )

    if not comparisons:
        return ""

    return "**Cross-Source Comparison:**\n\n" + "\n\n".join(comparisons)


def _get_intensity_level(percentage: float) -> str:
    """Get intensity descriptor based on percentage."""
    if percentage >= 70:
        return "overwhelmingly"
    elif percentage >= 50:
        return "predominantly"
    elif percentage >= 30:
        return "moderately"
    else:
        return "somewhat"


# ============================================================================
# STANCE INTERPRETATION
# ============================================================================

def generate_stance_interpretation(stance_df: pd.DataFrame, claim_text: str = "") -> str:
    """
    Generate a comprehensive interpretation of stance distribution data.

    Args:
        stance_df: DataFrame with columns:
            - source_name, total
            - agree_pct, neutral_pct, disagree_pct
        claim_text: The actual claim being analyzed

    Returns:
        Markdown-formatted interpretation text
    """
    if stance_df.empty:
        return "_No stance data available for this claim._"

    # Convert percentage columns to float to avoid Decimal type issues
    pct_columns = ['agree_pct', 'neutral_pct', 'disagree_pct']
    for col in pct_columns:
        if col in stance_df.columns:
            stance_df[col] = stance_df[col].astype(float)

    # Convert total to int to avoid Decimal type issues
    if 'total' in stance_df.columns:
        stance_df['total'] = stance_df['total'].astype(int)

    sections = []

    # Overall stance patterns
    overview = _analyze_stance_overview(stance_df, claim_text)
    if overview:
        sections.append(overview)

    # Individual source narratives (NEW)
    source_narratives = _generate_individual_source_stance_narratives(stance_df, claim_text)
    if source_narratives:
        sections.append(source_narratives)

    # Comparative analysis across sources (NEW)
    comparison = _generate_stance_comparison(stance_df, claim_text)
    if comparison:
        sections.append(comparison)

    # Source-specific positions
    extremes = _analyze_stance_extremes(stance_df, claim_text)
    if extremes:
        sections.append(extremes)

    # Polarization and consensus
    consensus = _analyze_stance_consensus(stance_df, claim_text)
    if consensus:
        sections.append(consensus)

    return "\n\n".join(sections)


def _analyze_stance_overview(df: pd.DataFrame, claim_text: str = "") -> str:
    """Analyze overall stance patterns."""
    # Calculate aggregate percentages
    total_articles = df['total'].sum()

    agg_agree = (df['agree_pct'] * df['total'] / 100).sum()
    agg_neutral = (df['neutral_pct'] * df['total'] / 100).sum()
    agg_disagree = (df['disagree_pct'] * df['total'] / 100).sum()

    agree_pct = agg_agree / total_articles * 100
    neutral_pct = agg_neutral / total_articles * 100
    disagree_pct = agg_disagree / total_articles * 100

    # Determine dominant stance
    stances = {
        'supportive': agree_pct,
        'neutral': neutral_pct,
        'critical': disagree_pct
    }
    dominant = max(stances, key=stances.get)

    # Build interpretation
    if dominant == 'supportive':
        description = (
            f"predominantly **supportive** ({agree_pct:.1f}% agree vs {disagree_pct:.1f}% disagree), "
            f"with {neutral_pct:.1f}% taking a neutral stance"
        )
    elif dominant == 'critical':
        description = (
            f"predominantly **critical** ({disagree_pct:.1f}% disagree vs {agree_pct:.1f}% agree), "
            f"with {neutral_pct:.1f}% taking a neutral stance"
        )
    else:
        description = (
            f"predominantly **neutral** ({neutral_pct:.1f}% neutral), "
            f"with {agree_pct:.1f}% supportive and {disagree_pct:.1f}% critical"
        )

    return (
        f"**Overall Stance Patterns:** Coverage of this claim is {description}. "
        f"A total of {int(total_articles)} articles were analyzed."
    )


def _analyze_stance_extremes(df: pd.DataFrame, claim_text: str = "") -> str:
    """Identify sources with strongest positions."""
    most_supportive = df.loc[df['agree_pct'].idxmax()]
    most_critical = df.loc[df['disagree_pct'].idxmax()]

    parts = []

    # Most supportive source
    if most_supportive['agree_pct'] > 20:  # Only mention if significant
        if claim_text:
            parts.append(
                f"**{most_supportive['source_name']}** agrees {most_supportive['agree_pct']:.1f}% "
                f"with the claim that \"{claim_text}\""
            )
        else:
            parts.append(
                f"**{most_supportive['source_name']}** shows {most_supportive['agree_pct']:.1f}% agreement"
            )

    # Most critical source
    if most_critical['disagree_pct'] > 20:  # Only mention if significant
        if claim_text:
            parts.append(
                f"**{most_critical['source_name']}** disagrees {most_critical['disagree_pct']:.1f}% "
                f"with the claim that \"{claim_text}\""
            )
        else:
            parts.append(
                f"**{most_critical['source_name']}** shows {most_critical['disagree_pct']:.1f}% disagreement"
            )

    if not parts:
        return ""

    return "**Source Positions:** " + ", while ".join(parts) + "."


def _analyze_stance_consensus(df: pd.DataFrame, claim_text: str = "") -> str:
    """Analyze polarization and consensus patterns."""
    # Calculate standard deviations
    agree_std = df['agree_pct'].std()
    disagree_std = df['disagree_pct'].std()
    avg_neutral = df['neutral_pct'].mean()

    avg_std = (agree_std + disagree_std) / 2

    # Check for polarization (low neutral, high variation)
    is_polarized = avg_neutral < 30 and avg_std > 20
    is_consensus = avg_std < 15

    if is_polarized:
        return (
            f"**Polarization Indicator:** Sources show strong divergence in stance "
            f"(variation: {avg_std:.1f}%) with relatively few neutral articles ({avg_neutral:.1f}% average), "
            f"suggesting this is a contentious claim with clearly divided opinions."
        )
    elif is_consensus:
        return (
            f"**Consensus Pattern:** Sources show similar stance distributions "
            f"(variation: {avg_std:.1f}%), indicating broad agreement on how to approach this claim."
        )
    else:
        # Check which stance varies most
        if agree_std > disagree_std:
            varying_aspect = "supportive coverage"
        else:
            varying_aspect = "critical coverage"

        return (
            f"**Mixed Consensus:** While sources generally align, there is moderate variation in "
            f"{varying_aspect} (variation: {avg_std:.1f}%), with {avg_neutral:.1f}% neutral coverage on average."
        )


def _generate_individual_source_stance_narratives(df: pd.DataFrame, claim_text: str = "") -> str:
    """Generate narrative interpretations for each source's stance."""
    narratives = []

    for _, row in df.iterrows():
        source = row['source_name']
        agree = row['agree_pct']
        disagree = row['disagree_pct']
        neutral = row['neutral_pct']

        # Determine dominant stance
        if disagree > agree and disagree > neutral:
            # Disagree dominant
            intensity = _get_intensity_level(disagree)
            if claim_text:
                narrative = (
                    f"**{source}** {intensity} disagrees ({disagree:.1f}%) "
                    f"with the claim that \"{claim_text[:80]}{'...' if len(claim_text) > 80 else ''}\""
                )
            else:
                narrative = f"**{source}** shows {intensity} disagreement ({disagree:.1f}%)"
        elif agree > disagree and agree > neutral:
            # Agree dominant
            intensity = _get_intensity_level(agree)
            if claim_text:
                narrative = (
                    f"**{source}** {intensity} agrees ({agree:.1f}%) "
                    f"with the claim that \"{claim_text[:80]}{'...' if len(claim_text) > 80 else ''}\""
                )
            else:
                narrative = f"**{source}** shows {intensity} agreement ({agree:.1f}%)"
        else:
            # Neutral dominant
            if claim_text:
                narrative = (
                    f"**{source}** maintains a neutral position ({neutral:.1f}%) "
                    f"on the claim that \"{claim_text[:80]}{'...' if len(claim_text) > 80 else ''}\""
                )
            else:
                narrative = f"**{source}** maintains a neutral position ({neutral:.1f}%)"

        narratives.append(narrative)

    if not narratives:
        return ""

    return "**Individual Source Stance:**\n\n" + "\n\n".join(narratives)


def _generate_stance_comparison(df: pd.DataFrame, claim_text: str = "") -> str:
    """Generate comparative analysis showing how sources differ in their stance."""
    comparisons = []

    # Check if all sources agree on stance direction
    all_disagree = all(df['disagree_pct'] > df['agree_pct'])
    all_agree = all(df['agree_pct'] > df['disagree_pct'])
    all_neutral = all(df['neutral_pct'] > 50)

    if all_disagree:
        comparisons.append(
            f"**Unanimous Disagreement:** All sources ({', '.join(df['source_name'].tolist())}) "
            f"predominantly disagree with this claim, showing a clear consensus in rejecting its validity."
        )
    elif all_agree:
        comparisons.append(
            f"**Unanimous Agreement:** All sources ({', '.join(df['source_name'].tolist())}) "
            f"predominantly agree with this claim, demonstrating consensus in accepting its validity."
        )
    elif all_neutral:
        comparisons.append(
            f"**Unanimous Neutrality:** All sources maintain neutral positions, "
            f"suggesting they neither explicitly endorse nor reject the claim."
        )
    else:
        # Sources have different stances - identify the split
        agreeing_sources = df[df['agree_pct'] > df['disagree_pct']]
        disagreeing_sources = df[df['disagree_pct'] > df['agree_pct']]
        neutral_sources = df[df['neutral_pct'] > 50]

        if not agreeing_sources.empty and not disagreeing_sources.empty:
            agree_list = ", ".join(agreeing_sources['source_name'].tolist())
            disagree_list = ", ".join(disagreeing_sources['source_name'].tolist())

            if not neutral_sources.empty:
                neutral_list = ", ".join(neutral_sources['source_name'].tolist())
                comparisons.append(
                    f"**Three-Way Split:** Sources are divided on this claim. "
                    f"**{agree_list}** tend(s) to agree, **{disagree_list}** tend(s) to disagree, "
                    f"while **{neutral_list}** remain(s) neutral, revealing no media consensus."
                )
            else:
                comparisons.append(
                    f"**Clear Divide:** Sources are split on this claim. "
                    f"**{agree_list}** tend(s) to support it, while **{disagree_list}** tend(s) to oppose it, "
                    f"revealing polarized media positions."
                )

        # Compare extremes
        most_agreeing = df.loc[df['agree_pct'].idxmax()]
        most_disagreeing = df.loc[df['disagree_pct'].idxmax()]

        if most_agreeing['agree_pct'] - most_disagreeing['agree_pct'] > 30:
            comparisons.append(
                f"**Stark Contrast:** **{most_agreeing['source_name']}** shows {most_agreeing['agree_pct']:.1f}% agreement "
                f"compared to **{most_disagreeing['source_name']}**'s {most_disagreeing['agree_pct']:.1f}% agreement, "
                f"highlighting substantial differences in how sources evaluate this claim's validity."
            )

    if not comparisons:
        return ""

    return "**Cross-Source Comparison:**\n\n" + "\n\n".join(comparisons)


# ============================================================================
# COMBINED SENTIMENT + STANCE INTERPRETATION (MEDIA BIAS ANALYSIS)
# ============================================================================

def generate_combined_bias_interpretation(
    sentiment_df: pd.DataFrame,
    stance_df: pd.DataFrame,
    claim_text: str = ""
) -> str:
    """
    Generate a comprehensive media bias interpretation combining sentiment and stance data.

    This analysis reveals how sources differ in their emotional framing (sentiment)
    versus their factual position (stance), which are key indicators of media bias.

    Args:
        sentiment_df: DataFrame with sentiment distribution by source
        stance_df: DataFrame with stance distribution by source
        claim_text: The actual claim being analyzed

    Returns:
        Markdown-formatted interpretation highlighting bias patterns
    """
    if sentiment_df.empty or stance_df.empty:
        return "_Insufficient data for combined bias analysis._"

    # Ensure data types
    pct_columns_sent = ['very_negative_pct', 'negative_pct', 'neutral_pct', 'positive_pct', 'very_positive_pct']
    for col in pct_columns_sent:
        if col in sentiment_df.columns:
            sentiment_df[col] = sentiment_df[col].astype(float)

    pct_columns_stance = ['agree_pct', 'neutral_pct', 'disagree_pct']
    for col in pct_columns_stance:
        if col in stance_df.columns:
            stance_df[col] = stance_df[col].astype(float)

    # Merge datasets on source_name
    merged_df = pd.merge(
        sentiment_df,
        stance_df,
        on='source_name',
        suffixes=('_sent', '_stance')
    )

    if merged_df.empty:
        return "_Unable to merge sentiment and stance data for bias analysis._"

    sections = []

    # Overall bias landscape - REMOVED
    # overview = _analyze_bias_overview(merged_df)
    # if overview:
    #     sections.append(overview)

    # NEW: Individual source combined narratives (sentiment + stance together)
    individual_narratives = _generate_combined_source_narratives(merged_df, claim_text)
    if individual_narratives:
        sections.append(individual_narratives)

    # NEW: Cross-source comparative analysis - REMOVED
    # comparison = _generate_combined_source_comparison(merged_df, claim_text)
    # if comparison:
    #     sections.append(comparison)

    # Sentiment-stance alignment analysis (Bias Through Framing) - REMOVED
    # alignment = _analyze_sentiment_stance_alignment(merged_df)
    # if alignment:
    #     sections.append(alignment)

    # Source-level bias patterns
    source_patterns = _analyze_source_bias_patterns(merged_df)
    if source_patterns:
        sections.append(source_patterns)

    # Editorial framing analysis
    framing = _analyze_editorial_framing(merged_df)
    if framing:
        sections.append(framing)

    return "\n\n".join(sections)


def _generate_combined_source_narratives(df: pd.DataFrame, claim_text: str = "") -> str:
    """
    Generate individual narratives for each source combining sentiment and stance.

    Example: "The Morning feels 65% negative about the claim that the Sri Lankan
    government has taken swift action. At the same time, The Morning disagrees
    45% with this claim."
    """
    df = df.copy()
    df['total_positive_sent'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative_sent'] = df['negative_pct'] + df['very_negative_pct']

    narratives = []
    claim_short = claim_text[:80] + '...' if len(claim_text) > 80 else claim_text

    for _, row in df.iterrows():
        source = row['source_name']

        # Sentiment part
        total_pos = row['total_positive_sent']
        total_neg = row['total_negative_sent']
        neutral_sent = row['neutral_pct_sent']

        # Stance part
        agree = row['agree_pct']
        disagree = row['disagree_pct']
        neutral_stance = row['neutral_pct_stance']

        # Build narrative
        narrative_parts = []

        # Sentiment narrative
        if total_neg > total_pos and total_neg > neutral_sent:
            intensity = _get_intensity_level(total_neg)
            if claim_text:
                sentiment_text = (
                    f"**{source}** feels **{intensity} negative** ({total_neg:.1f}%) about the claim "
                    f"that \"{claim_short}\""
                )
            else:
                sentiment_text = f"**{source}** expresses {intensity} negative sentiment ({total_neg:.1f}%)"
        elif total_pos > total_neg and total_pos > neutral_sent:
            intensity = _get_intensity_level(total_pos)
            if claim_text:
                sentiment_text = (
                    f"**{source}** feels **{intensity} positive** ({total_pos:.1f}%) about the claim "
                    f"that \"{claim_short}\""
                )
            else:
                sentiment_text = f"**{source}** expresses {intensity} positive sentiment ({total_pos:.1f}%)"
        else:
            if claim_text:
                sentiment_text = (
                    f"**{source}** maintains **neutral emotional tone** ({neutral_sent:.1f}%) when discussing "
                    f"the claim that \"{claim_short}\""
                )
            else:
                sentiment_text = f"**{source}** maintains neutral emotional tone ({neutral_sent:.1f}%)"

        narrative_parts.append(sentiment_text)

        # Stance narrative
        if disagree > agree and disagree > neutral_stance:
            intensity = _get_intensity_level(disagree)
            if claim_text:
                stance_text = (
                    f"At the same time, **{source}** **{intensity} disagrees** ({disagree:.1f}%) "
                    f"with this claim"
                )
            else:
                stance_text = f"while {intensity} disagreeing ({disagree:.1f}%) with it"
        elif agree > disagree and agree > neutral_stance:
            intensity = _get_intensity_level(agree)
            if claim_text:
                stance_text = (
                    f"At the same time, **{source}** **{intensity} agrees** ({agree:.1f}%) "
                    f"with this claim"
                )
            else:
                stance_text = f"while {intensity} agreeing ({agree:.1f}%) with it"
        else:
            if claim_text:
                stance_text = (
                    f"However, **{source}** takes a **neutral factual position** ({neutral_stance:.1f}%), "
                    f"neither clearly supporting nor opposing it"
                )
            else:
                stance_text = f"while maintaining a neutral position ({neutral_stance:.1f}%)"

        narrative_parts.append(stance_text)

        # Combine sentiment and stance
        full_narrative = ". ".join(narrative_parts) + "."
        narratives.append(full_narrative)

    if not narratives:
        return ""

    return "**📰 Individual Newspaper Analysis:**\n\n" + "\n\n".join(narratives)


def _generate_combined_source_comparison(df: pd.DataFrame, claim_text: str = "") -> str:
    """
    Generate cross-source comparative analysis showing how newspapers differ
    in both sentiment and stance.

    Example: "While The Morning is negative and disagrees, Daily News is
    positive and agrees, showing polarized coverage."
    """
    df = df.copy()
    df['total_positive_sent'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative_sent'] = df['negative_pct'] + df['very_negative_pct']

    comparisons = []

    # 1. Check for unanimous patterns
    all_negative = all(df['total_negative_sent'] > df['total_positive_sent'])
    all_positive = all(df['total_positive_sent'] > df['total_negative_sent'])
    all_disagree = all(df['disagree_pct'] > df['agree_pct'])
    all_agree = all(df['agree_pct'] > df['disagree_pct'])

    if all_negative and all_disagree:
        source_list = ", ".join(df['source_name'].tolist())
        comparisons.append(
            f"**🤝 Unanimous Negative Rejection:** All sources ({source_list}) express predominantly "
            f"**negative sentiment** AND **disagree** with this claim, showing complete consensus in "
            f"both emotional tone and factual position. This represents the strongest form of media alignment "
            f"against the claim."
        )
    elif all_positive and all_agree:
        source_list = ", ".join(df['source_name'].tolist())
        comparisons.append(
            f"**🤝 Unanimous Positive Support:** All sources ({source_list}) express predominantly "
            f"**positive sentiment** AND **agree** with this claim, demonstrating complete consensus in "
            f"both emotional framing and factual acceptance. This represents unified media support."
        )
    elif all_negative and all_agree:
        source_list = ", ".join(df['source_name'].tolist())
        comparisons.append(
            f"**🔄 Negative Support Pattern:** All sources ({source_list}) express **negative sentiment** "
            f"while **agreeing** with the claim. This unusual pattern suggests sources acknowledge the claim's "
            f"validity but frame it negatively, possibly indicating concern about its implications."
        )
    elif all_positive and all_disagree:
        source_list = ", ".join(df['source_name'].tolist())
        comparisons.append(
            f"**🔄 Positive Rejection Pattern:** All sources ({source_list}) express **positive sentiment** "
            f"while **disagreeing** with the claim. This suggests sources reject the claim but maintain "
            f"optimistic framing, possibly to soften criticism."
        )
    else:
        # Sources have different patterns - create detailed comparison

        # 2. Identify sources by their sentiment-stance profile
        negative_disagree = df[(df['total_negative_sent'] > df['total_positive_sent']) &
                               (df['disagree_pct'] > df['agree_pct'])]
        positive_agree = df[(df['total_positive_sent'] > df['total_negative_sent']) &
                           (df['agree_pct'] > df['disagree_pct'])]
        negative_agree = df[(df['total_negative_sent'] > df['total_positive_sent']) &
                           (df['agree_pct'] > df['disagree_pct'])]
        positive_disagree = df[(df['total_positive_sent'] > df['total_negative_sent']) &
                              (df['disagree_pct'] > df['agree_pct'])]
        neutral_both = df[(df['neutral_pct_sent'] > 40) & (df['neutral_pct_stance'] > 40)]

        # Build comparison narratives
        groups = []
        if not negative_disagree.empty:
            sources = ", ".join(negative_disagree['source_name'].tolist())
            groups.append(f"**{sources}** (negative + disagree)")
        if not positive_agree.empty:
            sources = ", ".join(positive_agree['source_name'].tolist())
            groups.append(f"**{sources}** (positive + agree)")
        if not negative_agree.empty:
            sources = ", ".join(negative_agree['source_name'].tolist())
            groups.append(f"**{sources}** (negative but agree)")
        if not positive_disagree.empty:
            sources = ", ".join(positive_disagree['source_name'].tolist())
            groups.append(f"**{sources}** (positive but disagree)")
        if not neutral_both.empty:
            sources = ", ".join(neutral_both['source_name'].tolist())
            groups.append(f"**{sources}** (neutral on both)")

        if len(groups) >= 2:
            comparisons.append(
                f"**⚖️ Divided Coverage:** Sources show different sentiment-stance combinations. "
                f"{', '.join(groups)}. This diversity reveals significant disagreement in how newspapers "
                f"perceive and frame this claim."
            )

        # 3. Highlight most extreme differences
        most_negative_source = df.loc[df['total_negative_sent'].idxmax()]
        most_positive_source = df.loc[df['total_positive_sent'].idxmax()]
        most_disagree_source = df.loc[df['disagree_pct'].idxmax()]
        most_agree_source = df.loc[df['agree_pct'].idxmax()]

        # Check if sentiment extremes differ from stance extremes
        if most_negative_source['source_name'] != most_disagree_source['source_name']:
            comparisons.append(
                f"**📊 Sentiment vs Stance Split:** **{most_negative_source['source_name']}** is most negative "
                f"emotionally ({most_negative_source['total_negative_sent']:.1f}%), but "
                f"**{most_disagree_source['source_name']}** disagrees most factually "
                f"({most_disagree_source['disagree_pct']:.1f}%), showing that emotional tone doesn't always "
                f"match factual position."
            )

        if most_positive_source['source_name'] != most_agree_source['source_name']:
            comparisons.append(
                f"**📊 Support Pattern Variation:** **{most_positive_source['source_name']}** is most positive "
                f"emotionally ({most_positive_source['total_positive_sent']:.1f}%), while "
                f"**{most_agree_source['source_name']}** agrees most factually "
                f"({most_agree_source['agree_pct']:.1f}%), indicating different ways of supporting claims."
            )

        # 4. Calculate polarization index
        sent_range = df['total_negative_sent'].max() - df['total_negative_sent'].min()
        stance_range = df['disagree_pct'].max() - df['disagree_pct'].min()

        if sent_range > 40 or stance_range > 40:
            comparisons.append(
                f"**🔥 High Polarization:** Sources show extreme variation (sentiment range: {sent_range:.1f}%, "
                f"stance range: {stance_range:.1f}%), indicating deeply divided media coverage with no consensus "
                f"on how to perceive or evaluate this claim."
            )
        elif sent_range < 20 and stance_range < 20:
            comparisons.append(
                f"**🤝 Strong Consensus:** Despite some differences, sources show relatively similar patterns "
                f"(sentiment range: {sent_range:.1f}%, stance range: {stance_range:.1f}%), suggesting broad "
                f"agreement on both the emotional tone and factual validity of this claim."
            )

    if not comparisons:
        return ""

    return "**🔍 Cross-Source Comparative Analysis:**\n\n" + "\n\n".join(comparisons)


def _analyze_bias_overview(df: pd.DataFrame) -> str:
    """Analyze overall bias landscape combining sentiment and stance."""
    # Calculate aggregate metrics
    df = df.copy()
    df['total_positive_sent'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative_sent'] = df['negative_pct'] + df['very_negative_pct']

    # Overall sentiment tone
    avg_positive = df['total_positive_sent'].mean()
    avg_negative = df['total_negative_sent'].mean()

    # Overall stance position
    avg_agree = df['agree_pct'].mean()
    avg_disagree = df['disagree_pct'].mean()

    # Determine bias pattern
    sentiment_tone = "positive" if avg_positive > avg_negative else "negative"
    stance_position = "supportive" if avg_agree > avg_disagree else "critical"

    # Check for alignment
    is_aligned = (sentiment_tone == "positive" and stance_position == "supportive") or \
                 (sentiment_tone == "negative" and stance_position == "critical")

    if is_aligned:
        pattern_desc = (
            f"Sources show **aligned coverage** where emotional tone and factual position match: "
            f"on average, {avg_positive:.1f}% positive sentiment aligns with {avg_agree:.1f}% supportive stance, "
            f"while {avg_negative:.1f}% negative sentiment aligns with {avg_disagree:.1f}% critical stance. "
            f"This suggests transparent reporting where emotional framing matches editorial position."
        )
    else:
        pattern_desc = (
            f"Sources show **misaligned coverage** where emotional tone contradicts factual position: "
            f"{avg_positive if sentiment_tone == 'positive' else avg_negative:.1f}% {sentiment_tone} sentiment "
            f"contrasts with {avg_agree if stance_position == 'supportive' else avg_disagree:.1f}% {stance_position} stance. "
            f"This divergence may indicate subtle bias through emotional framing that differs from stated positions."
        )

    return f"**Media Bias Landscape:** {pattern_desc}"


def _analyze_sentiment_stance_alignment(df: pd.DataFrame) -> str:
    """Identify sources with misaligned sentiment and stance (bias indicators)."""
    df = df.copy()
    df['total_positive_sent'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative_sent'] = df['negative_pct'] + df['very_negative_pct']

    # Calculate alignment scores
    # Positive alignment: positive sentiment + agree stance
    # Negative alignment: negative sentiment + disagree stance
    df['positive_alignment'] = (df['total_positive_sent'] + df['agree_pct']) / 2
    df['negative_alignment'] = (df['total_negative_sent'] + df['disagree_pct']) / 2

    # Identify misalignments (high positive sentiment but disagree, or negative sentiment but agree)
    df['positive_disagree_bias'] = df['total_positive_sent'] - df['agree_pct']
    df['negative_agree_bias'] = df['total_negative_sent'] - df['disagree_pct']

    bias_examples = []

    # Find sources with significant positive framing but disagreement (criticism with positive spin)
    pos_disagree_sources = df[df['positive_disagree_bias'] > 20].sort_values('positive_disagree_bias', ascending=False)
    if not pos_disagree_sources.empty:
        source = pos_disagree_sources.iloc[0]
        bias_examples.append(
            f"**{source['source_name']}** uses notably positive framing ({source['total_positive_sent']:.1f}% positive) "
            f"while mostly disagreeing with the claim ({source['disagree_pct']:.1f}% disagree), "
            f"suggesting **critical coverage softened by positive language**."
        )

    # Find sources with negative framing but agreement (support with negative tone)
    neg_agree_sources = df[(df['total_negative_sent'] > df['total_positive_sent']) & (df['agree_pct'] > df['disagree_pct'])]
    if not neg_agree_sources.empty:
        source = neg_agree_sources.iloc[0]
        bias_examples.append(
            f"**{source['source_name']}** employs negative framing ({source['total_negative_sent']:.1f}% negative) "
            f"while supporting the claim ({source['agree_pct']:.1f}% agree), "
            f"indicating **supportive coverage with emotionally charged language**."
        )

    if not bias_examples:
        return (
            f"**Sentiment-Stance Alignment:** Most sources show consistent alignment between their emotional tone "
            f"and factual position, suggesting straightforward reporting without contradictory framing."
        )

    return f"**Bias Through Framing:** {' '.join(bias_examples)}"


def _analyze_source_bias_patterns(df: pd.DataFrame) -> str:
    """Identify distinct bias patterns across sources."""
    df = df.copy()
    df['total_positive_sent'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative_sent'] = df['negative_pct'] + df['very_negative_pct']

    # Classify sources by their dominant pattern
    patterns = []

    for _, row in df.iterrows():
        source = row['source_name']

        # Pattern 1: Strongly biased (high sentiment + high stance in same direction)
        if row['total_positive_sent'] > 40 and row['agree_pct'] > 40:
            patterns.append(f"**{source}**: Strongly pro-claim (positive tone + supportive stance)")
        elif row['total_negative_sent'] > 40 and row['disagree_pct'] > 40:
            patterns.append(f"**{source}**: Strongly anti-claim (negative tone + critical stance)")

        # Pattern 2: Neutral balanced
        elif row['neutral_pct_sent'] > 50 and row['neutral_pct_stance'] > 50:
            patterns.append(f"**{source}**: Balanced reporting (neutral tone + neutral stance)")

        # Pattern 3: Emotional but neutral stance
        elif (row['total_positive_sent'] > 30 or row['total_negative_sent'] > 30) and row['neutral_pct_stance'] > 50:
            tone = "positive" if row['total_positive_sent'] > row['total_negative_sent'] else "negative"
            patterns.append(f"**{source}**: Emotionally charged but factually neutral ({tone} tone, neutral stance)")

    if not patterns:
        return ""

    return f"**Source Bias Profiles:** {'; '.join(patterns[:4])}{'...' if len(patterns) > 4 else ''}"


def _analyze_editorial_framing(df: pd.DataFrame) -> str:
    """Analyze editorial framing strategies across sources."""
    df = df.copy()
    df['total_positive_sent'] = df['positive_pct'] + df['very_positive_pct']
    df['total_negative_sent'] = df['negative_pct'] + df['very_negative_pct']

    # Calculate variance in sentiment vs stance
    sent_variance = (df['total_positive_sent'].std() + df['total_negative_sent'].std()) / 2
    stance_variance = (df['agree_pct'].std() + df['disagree_pct'].std()) / 2

    if sent_variance > stance_variance * 1.5:
        # High sentiment variation, lower stance variation
        return (
            f"**Editorial Strategy:** Sources show more variation in emotional framing (variation: {sent_variance:.1f}%) "
            f"than in factual positions (variation: {stance_variance:.1f}%), suggesting bias is **primarily expressed through tone** "
            f"rather than through different factual claims. This indicates subtle bias where sources agree on facts but differ in how they emotionally frame them."
        )
    elif stance_variance > sent_variance * 1.5:
        # High stance variation, lower sentiment variation
        return (
            f"**Editorial Strategy:** Sources show more variation in factual positions (variation: {stance_variance:.1f}%) "
            f"than in emotional tone (variation: {sent_variance:.1f}%), suggesting bias is **primarily expressed through position-taking** "
            f"rather than emotional language. Sources have different factual interpretations but maintain similar emotional restraint."
        )
    else:
        # Balanced variation
        return (
            f"**Editorial Strategy:** Sources show similar levels of variation in both emotional framing and factual positions "
            f"(sentiment variation: {sent_variance:.1f}%, stance variation: {stance_variance:.1f}%), "
            f"indicating comprehensive editorial divergence where both tone and position differ across sources."
        )
