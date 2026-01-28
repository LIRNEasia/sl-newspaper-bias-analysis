"""Reusable chart components for the dashboard."""

import streamlit as st
import plotly.graph_objects as go


def render_multi_model_stacked_bars(df, model_colors):
    """Render stacked bars grouped by source, showing all models.

    Args:
        df: DataFrame with columns: source_name, model_type, overall_sentiment
        model_colors: Dict mapping model names to colors
    """
    # Calculate percentages for each model
    results = []
    for source in df['source_name'].unique():
        for model in df['model_type'].unique():
            subset = df[(df['source_name'] == source) & (df['model_type'] == model)]

            if len(subset) == 0:
                continue

            total = len(subset)
            negative = len(subset[subset['overall_sentiment'] < -0.5])
            neutral = len(subset[(subset['overall_sentiment'] >= -0.5) &
                                 (subset['overall_sentiment'] <= 0.5)])
            positive = len(subset[subset['overall_sentiment'] > 0.5])

            results.append({
                'source': source,
                'model': model,
                'negative_pct': (negative / total) * 100,
                'neutral_pct': (neutral / total) * 100,
                'positive_pct': (positive / total) * 100
            })

    import pandas as pd
    results_df = pd.DataFrame(results)

    # Create grouped bar chart (one per model)
    for model in sorted(df['model_type'].unique()):
        model_data = results_df[results_df['model'] == model]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Negative (< -0.5)',
            x=model_data['source'],
            y=model_data['negative_pct'],
            marker_color='#d62728',
            text=model_data['negative_pct'].round(1).astype(str) + '%',
            textposition='inside'
        ))

        fig.add_trace(go.Bar(
            name='Neutral (-0.5 to 0.5)',
            x=model_data['source'],
            y=model_data['neutral_pct'],
            marker_color='#7f7f7f',
            text=model_data['neutral_pct'].round(1).astype(str) + '%',
            textposition='inside'
        ))

        fig.add_trace(go.Bar(
            name='Positive (> 0.5)',
            x=model_data['source'],
            y=model_data['positive_pct'],
            marker_color='#2ca02c',
            text=model_data['positive_pct'].round(1).astype(str) + '%',
            textposition='inside'
        ))

        fig.update_layout(
            barmode='stack',
            height=300,
            title=f"{model.upper()} Model",
            xaxis_title="Source",
            yaxis_title="Percentage (%)",
            yaxis_range=[0, 100],
            showlegend=True
        )

        st.plotly_chart(fig, width='stretch')


def render_source_model_comparison(df, model_colors):
    """Grouped bar chart: avg sentiment by source for each model.

    Args:
        df: DataFrame with columns: source_name, model_type, overall_sentiment
        model_colors: Dict mapping model names to colors
    """
    # Calculate average sentiment per source per model
    agg = df.groupby(['source_name', 'model_type'])['overall_sentiment'].mean().reset_index()

    fig = go.Figure()

    for model in sorted(df['model_type'].unique()):
        model_data = agg[agg['model_type'] == model]

        fig.add_trace(go.Bar(
            name=model.upper(),
            x=model_data['source_name'],
            y=model_data['overall_sentiment'],
            marker_color=model_colors.get(model, '#999')
        ))

    fig.update_layout(
        barmode='group',
        height=400,
        xaxis_title="News Source",
        yaxis_title="Average Sentiment (-5 to +5)",
        yaxis_range=[-5, 5]
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")

    st.plotly_chart(fig, width='stretch')


def render_model_agreement_heatmap(df):
    """Create correlation heatmap showing model agreement.

    Args:
        df: DataFrame with columns: source_id, topic, model_type, overall_sentiment
    """
    # Pivot to get one column per model
    pivot = df.pivot_table(
        values='overall_sentiment',
        index=['source_id', 'topic'],
        columns='model_type',
        aggfunc='mean'
    )

    # Calculate correlation matrix
    corr = pivot.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.values.round(2),
        texttemplate='%{text}',
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        height=400,
        xaxis_title="Model",
        yaxis_title="Model"
    )

    st.plotly_chart(fig, width='stretch')
