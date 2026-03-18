"""Ditwah Cyclone — Deep Dive Page."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data.loaders import (
    load_overview_stats,
    load_ditwah_timeline,
    load_available_models,
    load_topic_sentiment,
    load_sentiment_by_source,
    load_sentiment_percentage_by_source,
    load_chunk_topic_by_source,
    load_chunk_outlet_totals,
    load_chunk_topics,
    load_stance_polarization_matrix,
    load_entity_stance_summary,
    load_topics_with_keywords,
    load_topic_by_source,
    load_outlet_totals,
    load_all_outlet_totals,
)
from components.source_mapping import SOURCE_NAMES, SOURCE_COLORS
from components.version_selector import render_version_selector
from components.styling import apply_page_style

apply_page_style()

st.title("Ditwah Cyclone — Deep Dive")

# ============================================================================
# Settings Popover
# ============================================================================

OUTLETS = ["dailynews_en", "themorning_en", "ft_en", "island_en"]

with st.popover("⚙️ Settings"):
    st.markdown("**Version Selectors**")
    chunk_topics_version_id = render_version_selector("chunk_topics")
    topics_version_id = render_version_selector("topics")
    claims_version_id = render_version_selector("ditwah_claims")
    entity_stance_version_id = render_version_selector("entity_stance")

    st.markdown("**Sentiment Model**")
    available_models = load_available_models()
    model_list = [m["model_type"] for m in available_models] if available_models else []
    MODEL_DISPLAY_NAMES = {
        "roberta": "RoBERTa",
        "distilbert": "DistilBERT",
        "finbert": "FinBERT",
        "vader": "VADER",
        "textblob": "TextBlob",
        "local": "Local (RoBERTa)",
    }
    sentiment_model = st.selectbox(
        "Select Sentiment Model",
        options=model_list,
        format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x.upper()),
        key="ditwah_sentiment_model",
    ) if model_list else None

st.divider()

# ============================================================================
# Section 1: Coverage Analysis
# ============================================================================

st.header("1 Coverage Analysis")

stats = load_overview_stats()

# Articles per source
ditwah_df = pd.DataFrame(stats["ditwah_by_source"])
if not ditwah_df.empty:
    ditwah_df["source_name"] = ditwah_df["source_id"].map(SOURCE_NAMES)

    col_cov1, _spacer, col_cov2 = st.columns([5, 1, 5])

    with col_cov1:
        st.caption("Article count")
        fig = px.bar(
            ditwah_df,
            x="source_name",
            y="count",
            color="source_name",
            color_discrete_map=SOURCE_COLORS,
            labels={"count": "Articles", "source_name": "Source"},
            text="count",
        )
        fig.update_traces(textposition="outside", width=0.4)
        fig.update_layout(showlegend=False, height=400, xaxis_title="Source", yaxis_title="Articles",
                          yaxis=dict(range=[0, ditwah_df["count"].max() * 1.15]))
        st.plotly_chart(fig, use_container_width=True)

    with col_cov2:
        st.caption("% of each outlet's total articles")
        outlet_totals = load_all_outlet_totals()
        if outlet_totals:
            ditwah_df["total"] = ditwah_df["source_id"].map(outlet_totals)
            ditwah_df["pct"] = ditwah_df["count"] / ditwah_df["total"] * 100
            fig2 = px.bar(
                ditwah_df,
                x="source_name",
                y="pct",
                color="source_name",
                color_discrete_map=SOURCE_COLORS,
                labels={"pct": "% of Total Articles", "source_name": "Source"},
                text=ditwah_df["pct"].round(1).astype(str) + "%",
            )
            fig2.update_traces(textposition="outside", width=0.4)
            fig2.update_layout(showlegend=False, height=400, xaxis_title="Source", yaxis_title="% of Total Articles",
                               yaxis=dict(range=[0, ditwah_df["pct"].max() * 1.15]))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Outlet totals not available.")
else:
    st.info("No Ditwah Cyclone articles found.")



# Topic coverage table
if not topics_version_id:
    st.info("Select a Topics version in ⚙️ Settings to view topic coverage analysis.")
else:
    import json as _json_cov

    topics_with_kw = load_topics_with_keywords(topics_version_id, limit=15)
    topic_source_data = load_topic_by_source(topics_version_id)
    outlet_totals = load_outlet_totals()

    if topics_with_kw and topic_source_data and outlet_totals:
        st.subheader("Topic Coverage by Source")
        st.markdown("Which topics show the largest coverage differences between outlets?")

        ts_df_bias = pd.DataFrame(topic_source_data)
        all_outlets = [sid for sid in SOURCE_NAMES.keys() if sid in outlet_totals]
        top_names = [t["name"] for t in topics_with_kw]
        ts_bias_filtered = ts_df_bias[ts_df_bias["topic"].isin(top_names)]

        bias_rows = []
        for t in topics_with_kw:
            tname = t["name"]
            proportions = {}
            for sid in all_outlets:
                count = ts_bias_filtered[
                    (ts_bias_filtered["topic"] == tname) & (ts_bias_filtered["source_id"] == sid)
                ]["count"].sum()
                total = outlet_totals.get(sid, 1)
                proportions[sid] = (count / total) * 100

            vals = list(proportions.values())
            spread = max(vals) - min(vals)
            max_outlet = SOURCE_NAMES[max(proportions, key=proportions.get)]
            min_outlet = SOURCE_NAMES[min(proportions, key=proportions.get)]

            aspect = tname
            try:
                if t.get("description"):
                    desc_data = _json_cov.loads(t["description"])
                    if desc_data.get("aspect"):
                        aspect = desc_data["aspect"]
            except (ValueError, TypeError):
                pass

            row = {"Aspect": aspect, "Topic": tname}
            for sid in all_outlets:
                row[SOURCE_NAMES[sid]] = round(proportions[sid], 1)
            row["Spread"] = round(spread, 1)
            row["Most"] = max_outlet
            row["Least"] = min_outlet
            bias_rows.append(row)

        bias_df = pd.DataFrame(bias_rows)
        display_cols = ["Aspect"] + [SOURCE_NAMES[s] for s in all_outlets] + ["Spread", "Most", "Least"]
        st.dataframe(
            bias_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Spread": st.column_config.NumberColumn(format="%.1f pp"),
                **{SOURCE_NAMES[s]: st.column_config.NumberColumn(format="%.1f%%") for s in all_outlets},
            },
        )

st.divider()

# ============================================================================
# Section 2: Sub-topic Analysis
# ============================================================================

st.header("2 Sub-topic Analysis")

if not chunk_topics_version_id:
    st.info("Select a Chunk Topics version in ⚙️ Settings to view sub-topic analysis.")
else:
    topics = load_chunk_topics(chunk_topics_version_id)
    topic_source_data = load_chunk_topic_by_source(chunk_topics_version_id)
    outlet_totals = load_chunk_outlet_totals(chunk_topics_version_id)

    if not topics or not topic_source_data:
        st.warning("No chunk topic data found for this version. Run the pipeline first.")
    else:
        ts_df = pd.DataFrame(topic_source_data)
        all_outlets = [o for o in OUTLETS if o in (outlet_totals or {})]
        outlet_names = [SOURCE_NAMES[o] for o in all_outlets]

        def _topic_label(t):
            desc = t.get("description")
            if desc:
                try:
                    parsed = json.loads(desc) if isinstance(desc, str) else desc
                    claim = parsed.get("claim")
                    if claim:
                        return f"{claim}"
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
            keywords = t.get("keywords") or []
            return f"{', '.join(keywords[:5]).title()}"

        bias_rows = []
        for t in topics:
            tname = t["name"]
            proportions = {}
            for o in all_outlets:
                count = ts_df[
                    (ts_df["topic_name"] == tname) & (ts_df["source_id"] == o)
                ]["count"].sum()
                total = outlet_totals.get(o, 1)
                proportions[o] = (count / total) * 100

            vals = list(proportions.values())
            spread = max(vals) - min(vals)
            max_outlet = SOURCE_NAMES[max(proportions, key=proportions.get)]
            min_outlet = SOURCE_NAMES[min(proportions, key=proportions.get)]

            row = {"Topic": f"T{t['topic_id']}", "Label": _topic_label(t)}
            for o in all_outlets:
                row[SOURCE_NAMES[o]] = round(proportions[o], 1)
            row["Spread"] = round(spread, 1)
            row["Most"] = max_outlet
            row["Least"] = min_outlet
            bias_rows.append(row)

        bias_df = pd.DataFrame(bias_rows).sort_values("Spread", ascending=False)
        display_cols = ["Topic", "Label"] + outlet_names + ["Spread", "Most", "Least"]
        st.dataframe(
            bias_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Spread": st.column_config.NumberColumn(format="%.1f pp"),
                **{name: st.column_config.NumberColumn(format="%.1f%%") for name in outlet_names},
            },
        )

st.divider()

# ============================================================================
# Section 3: Omission Detection
# ============================================================================

st.header("3 Omission Detection")
st.markdown(
    "Topics present in some outlets but absent from others "
    "(threshold: at least 5 chunks in one outlet)."
)

if not chunk_topics_version_id:
    st.info("Select a Chunk Topics version in ⚙️ Settings to view omission detection.")
else:
    topics = load_chunk_topics(chunk_topics_version_id)
    topic_source_data = load_chunk_topic_by_source(chunk_topics_version_id)
    outlet_totals = load_chunk_outlet_totals(chunk_topics_version_id)

    if not topics or not topic_source_data:
        st.warning("No chunk topic data found for this version.")
    else:
        ts_df = pd.DataFrame(topic_source_data)
        all_outlets = [o for o in OUTLETS if o in (outlet_totals or {})]

        omission_rows = []
        for t in topics:
            tid = t["topic_id"]
            tname = t["name"]
            counts = {}
            for o in all_outlets:
                match = ts_df[(ts_df["topic_id"] == tid) & (ts_df["source_id"] == o)]
                counts[o] = int(match["count"].sum()) if len(match) > 0 else 0

            present = [o for o, c in counts.items() if c > 0]
            absent = [o for o, c in counts.items() if c == 0]

            if absent and max(counts.values()) >= 5:
                claim_text = None
                desc = t.get("description")
                if desc:
                    try:
                        parsed = json.loads(desc) if isinstance(desc, str) else desc
                        claim_text = parsed.get("claim")
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                if not claim_text:
                    keywords = t.get("keywords") or []
                    claim_text = ", ".join(keywords[:5])

                omission_rows.append({
                    "Topic": f"T{tid}",
                    "Claim": claim_text,
                    "Total": sum(counts.values()),
                    **{SOURCE_NAMES[o]: counts[o] for o in all_outlets},
                    "Present In": ", ".join(SOURCE_NAMES[o] for o in present),
                    "Absent From": ", ".join(SOURCE_NAMES[o] for o in absent),
                })

        if omission_rows:
            omission_df = pd.DataFrame(omission_rows).sort_values("Total", ascending=False)
            st.caption(f"Found {len(omission_df)} topics with omissions")
            st.dataframe(omission_df, use_container_width=True, hide_index=True)
        else:
            st.info("No omissions detected with current thresholds.")

st.divider()

# ============================================================================
# Section 4: Sentiment Analysis
# ============================================================================

st.header("4 Sentiment Analysis")

st.info(
    "**Sentiment score** ranges from **−5 (very negative)** to **+5 (very positive)**, "
    "with 0 as neutral. Scores reflect the average emotional tone of article headlines and "
    "body text as measured by the selected sentiment model.\n\n"
    "**Examples:** *'Devastating floods destroy homes'* → −4 · "
    "*'Government announces infrastructure plan'* → 0 · "
    "*'Economy recovers as exports rise'* → +3"
)

if not topics_version_id or not sentiment_model:
    st.info("Select a Topics version and Sentiment Model in ⚙️ Settings to view this section.")
else:
    topic_sentiment = load_topic_sentiment(sentiment_model, topics_version_id)
    if topic_sentiment:
        ts_df = pd.DataFrame(topic_sentiment)
        ts_df["source_name"] = ts_df["source_id"].map(SOURCE_NAMES)

        pivot = ts_df.pivot_table(
            values="avg_sentiment",
            index="topic",
            columns="source_name",
            aggfunc="mean",
        )

        # Select top 8 topics by sentiment spread across sources
        pivot_multi = pivot.dropna(thresh=2)
        spread = pivot_multi.max(axis=1) - pivot_multi.min(axis=1)
        top_topics = spread.nlargest(8).index.tolist()

        st.caption("Top 8 topics by sentiment divergence across sources")

        for i in range(0, len(top_topics), 2):
            col_left, col_right = st.columns(2)
            for col, topic in zip([col_left, col_right], top_topics[i:i+2]):
                topic_data = ts_df[ts_df["topic"] == topic].copy()
                topic_title = topic if len(topic) <= 60 else topic[:57] + "..."
                topic_spread = spread[topic]

                fig = px.bar(
                    topic_data,
                    x="source_name",
                    y="avg_sentiment",
                    color="source_name",
                    color_discrete_map=SOURCE_COLORS,
                    title=topic_title,
                    labels={"source_name": "Source", "avg_sentiment": "Avg Sentiment"},
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
                fig.update_layout(
                    yaxis=dict(range=[-5, 5], title="Avg Sentiment"),
                    xaxis_title="",
                    showlegend=False,
                    height=300,
                    margin=dict(t=40, b=20),
                )
                with col:
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No topic-sentiment data found for this model and version.")

st.divider()

# ============================================================================
# Section 5: Stance Analysis
# ============================================================================

st.header("5 Stance Analysis")

col_stance1, col_stance2 = st.columns(2)

with col_stance1:
    st.subheader("By Entity")
    if not entity_stance_version_id:
        st.info("Select an Entity Stance version in ⚙️ Settings.")
    else:
        df_summary = load_entity_stance_summary(entity_stance_version_id)
        if df_summary is not None and not df_summary.empty:
            df_summary["source_name"] = (
                df_summary["source_id"].map(SOURCE_NAMES).fillna(df_summary["source_id"])
            )
            # Top 10 most polarizing entities by abs avg stance
            top_entities = (
                df_summary.groupby("entity_text")["avg_stance"]
                .apply(lambda x: x.abs().mean())
                .nlargest(10)
                .index.tolist()
            )
            df_top = df_summary[df_summary["entity_text"].isin(top_entities)]
            df_agg = (
                df_top.groupby(["entity_text", "source_name"], as_index=False)
                .apply(
                    lambda g: pd.Series({
                        "avg_stance": (g["avg_stance"] * g["stance_count"]).sum() / g["stance_count"].sum(),
                        "stance_count": g["stance_count"].sum(),
                    }),
                    include_groups=False,
                )
            )
            fig = px.bar(
                df_agg,
                x="entity_text",
                y="avg_stance",
                color="source_name",
                color_discrete_map=SOURCE_COLORS,
                barmode="group",
                labels={"avg_stance": "Avg Stance", "entity_text": "Entity", "source_name": "Source"},
            )
            fig.update_layout(yaxis=dict(range=[-1, 1]), height=450)
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entity stance data found for this version.")

with col_stance2:
    st.subheader("By Claim")
    if not claims_version_id:
        st.info("Select a Ditwah Claims version in ⚙️ Settings.")
    else:
        df_polar = load_stance_polarization_matrix(claims_version_id)
        if df_polar is not None and not df_polar.empty:
            df_polar["source_name"] = df_polar["source_id"].map(SOURCE_NAMES).fillna(df_polar["source_id"])

            # Truncate claim text for display
            df_polar["claim_short"] = df_polar["claim_text"].str[:60] + "…"

            # Top 15 most controversial claims
            top_claims = (
                df_polar.groupby("claim_id")["controversy_index"]
                .mean()
                .nlargest(15)
                .index.tolist()
            )
            df_top = df_polar[df_polar["claim_id"].isin(top_claims)]

            pivot = df_top.pivot_table(
                values="avg_stance",
                index="claim_short",
                columns="source_name",
                aggfunc="mean",
            )

            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Stance"),
            ))
            fig.update_layout(
                height=500,
                xaxis_title="Source",
                yaxis_title="Claim",
                yaxis=dict(tickfont=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No stance polarization data found for this version.")

st.divider()

# ============================================================================
# Section 6: Polarity
# ============================================================================

st.header("6 Polarity")

if not sentiment_model:
    st.info("Select a Sentiment Model in ⚙️ Settings to view polarity analysis.")
else:
    # 100% stacked bar (neg/neutral/pos)
    pct_data = load_sentiment_percentage_by_source(sentiment_model)
    if pct_data:
        pct_df = pd.DataFrame(pct_data)
        pct_df["source_name"] = pct_df["source_id"].map(SOURCE_NAMES)
        pct_df["negative_pct"] = pct_df["negative_count"] / pct_df["total_count"] * 100
        pct_df["neutral_pct"] = pct_df["neutral_count"] / pct_df["total_count"] * 100
        pct_df["positive_pct"] = pct_df["positive_count"] / pct_df["total_count"] * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Negative (< -0.5)",
            x=pct_df["source_name"],
            y=pct_df["negative_pct"],
            marker_color="#d62728",
            text=pct_df["negative_pct"].round(1).astype(str) + "%",
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="Neutral (-0.5 to 0.5)",
            x=pct_df["source_name"],
            y=pct_df["neutral_pct"],
            marker_color="#7f7f7f",
            text=pct_df["neutral_pct"].round(1).astype(str) + "%",
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="Positive (> 0.5)",
            x=pct_df["source_name"],
            y=pct_df["positive_pct"],
            marker_color="#2ca02c",
            text=pct_df["positive_pct"].round(1).astype(str) + "%",
            textposition="inside",
        ))
        fig.update_layout(
            barmode="stack",
            yaxis_title="Percentage (%)",
            xaxis_title="News Source",
            height=380,
            yaxis_range=[0, 100],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Average sentiment with error bars
    sentiment_data = load_sentiment_by_source(sentiment_model)
    if sentiment_data:
        src_df = pd.DataFrame(sentiment_data)
        src_df["source_name"] = src_df["source_id"].map(SOURCE_NAMES)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=src_df["source_name"],
            y=src_df["avg_sentiment"],
            error_y=dict(type="data", array=src_df["stddev_sentiment"]),
            marker_color=[SOURCE_COLORS.get(name, "#999") for name in src_df["source_name"]],
            text=src_df["avg_sentiment"].round(2),
            textposition="outside",
        ))
        fig.update_layout(
            yaxis_title="Average Sentiment Score (-5 to +5)",
            xaxis_title="News Source",
            height=380,
            yaxis_range=[-5, 5],
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        st.plotly_chart(fig, use_container_width=True)

    if not pct_data and not sentiment_data:
        st.info("No sentiment data found. Run the sentiment pipeline first.")
