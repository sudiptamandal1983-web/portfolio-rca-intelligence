"""
dashboard.py — Portfolio RCA Insights Dashboard

Reads the latest JSON report from the reports/ directory and renders
an interactive Streamlit dashboard for senior stakeholders.

Usage:
    pip install streamlit plotly
    streamlit run dashboard.py

    # Point at a specific report:
    streamlit run dashboard.py -- --report reports/audit_20260522_013726.json
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title  = "Portfolio RCA Dashboard",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.metric-card {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.severity-critical {
    border-left: 3px solid #e24b4a;
    background: linear-gradient(135deg, #1a0a0a, #0f0f0f);
}

.severity-warning {
    border-left: 3px solid #ef9f27;
    background: linear-gradient(135deg, #1a1200, #0f0f0f);
}

.severity-info {
    border-left: 3px solid #378add;
    background: linear-gradient(135deg, #0a1220, #0f0f0f);
}

.dimension-tag {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: #1e1e1e;
    padding: 3px 8px;
    border-radius: 4px;
    color: #888;
}

.z-score-positive { color: #e24b4a; font-family: 'DM Mono', monospace; }
.z-score-negative { color: #1d9e75; font-family: 'DM Mono', monospace; }

.narrative-text {
    font-size: 14px;
    line-height: 1.7;
    color: #c8c8c8;
    border-left: 2px solid #2a2a2a;
    padding-left: 16px;
    margin: 12px 0;
}

.co-mover-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #1a1a1a;
    font-size: 13px;
}

.risk-deteriorating { color: #e24b4a; }
.risk-improving     { color: #1d9e75; }
.risk-mixed         { color: #ef9f27; }
.risk-isolated      { color: #888;    }

.stMetric > div > div { font-family: 'DM Mono', monospace !important; }

div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load report
# ---------------------------------------------------------------------------

@st.cache_data
def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_latest_report(report_dir: str = "reports") -> "str | None":
    files = glob.glob(f"{report_dir}/audit_*.json")
    return max(files, key=os.path.getmtime) if files else None


def get_report_path() -> "str | None":
    # CLI override
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=None)
    try:
        args, _ = parser.parse_known_args()
        if args.report and os.path.exists(args.report):
            return args.report
    except Exception:
        pass
    return find_latest_report()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEVERITY_COLOR = {
    "critical": "#e24b4a",
    "warning":  "#ef9f27",
    "info":     "#378add",
}

RISK_COLOR = {
    "deteriorating": "#e24b4a",
    "improving":     "#1d9e75",
    "mixed":         "#ef9f27",
    "isolated":      "#888780",
    "unknown":       "#888780",
}

RISK_ICON = {
    "deteriorating": "📉",
    "improving":     "📈",
    "mixed":         "↔️",
    "isolated":      "🔍",
    "unknown":       "❓",
}

DIMENSION_LABELS = {
    "regional":           "Geographic DTI",
    "vintage_risk":       "Vintage Delinquency",
    "credit_quality":     "Credit Quality",
    "yield_analysis":     "Yield Analysis",
    "utilisation_stress": "Utilisation Stress",
    "income_verification":"Income Verification",
}


def fmt_segment(segment: dict) -> str:
    return " · ".join(f"{k.replace('_',' ')}={v}" for k, v in segment.items())


def fmt_z(z: float) -> str:
    return f"{z:+.2f}σ"


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

report_path = get_report_path()

if not report_path:
    st.error("No report found in reports/ directory. Run `python3 run.py` first.")
    st.stop()

report   = load_report(report_path)
insights = report.get("insights", [])
df       = pd.DataFrame(insights) if insights else pd.DataFrame()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🏦 Portfolio RCA")
    st.markdown("---")

    # Report picker
    report_files = sorted(
        glob.glob("reports/audit_*.json"),
        key=os.path.getmtime,
        reverse=True,
    )
    report_names = [os.path.basename(f) for f in report_files]
    selected_idx = 0
    if report_names:
        selected_name = st.selectbox(
            "Report", report_names, index=0,
            help="Select a historical report"
        )
        selected_path = f"reports/{selected_name}"
        if selected_path != report_path:
            report      = load_report(selected_path)
            insights    = report.get("insights", [])
            df          = pd.DataFrame(insights) if insights else pd.DataFrame()
            report_path = selected_path

    st.markdown("---")

    # Filters
    st.markdown("### Filters")

    all_dims = sorted(df["dimension"].unique().tolist()) if not df.empty else []
    selected_dims = st.multiselect(
        "Dimensions", all_dims, default=all_dims,
    )

    all_severities = ["critical", "warning", "info"]
    selected_sevs = st.multiselect(
        "Severity", all_severities, default=["critical", "warning"],
    )

    z_min = st.slider(
        "Min |z-score|", 0.0, 5.0, 2.0, 0.1,
    )

    st.markdown("---")
    st.markdown("### Run info")
    st.markdown(f"**Run ID:** `{report.get('run_id', '—')}`")
    gen_at = report.get("generated_at", "")
    if gen_at:
        try:
            dt = datetime.fromisoformat(gen_at)
            st.markdown(f"**Generated:** {dt.strftime('%d %b %Y %H:%M')}")
        except Exception:
            st.markdown(f"**Generated:** {gen_at[:16]}")
    st.markdown(f"**Duration:** {report.get('duration_s', 0):.1f}s")
    st.markdown(f"**LLM mode:** {'✅ on' if report.get('llm_mode') else '⚪ off'}")

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

filtered = df.copy()
if not filtered.empty:
    if selected_dims:
        filtered = filtered[filtered["dimension"].isin(selected_dims)]
    if selected_sevs:
        filtered = filtered[filtered["severity"].isin(selected_sevs)]
    filtered = filtered[filtered["top_z"].abs() >= z_min]
    filtered = filtered.sort_values("top_z", ascending=False)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='margin-bottom:4px'>Portfolio RCA Dashboard</h1>",
    unsafe_allow_html=True,
)
run_date = report.get("generated_at", "")[:10]
st.markdown(
    f"<p style='color:#888;font-size:14px;margin-top:0'>"
    f"Risk anomaly intelligence  ·  {run_date}</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

total     = len(filtered)
critical  = len(filtered[filtered["severity"] == "critical"]) if not filtered.empty else 0
warning   = len(filtered[filtered["severity"] == "warning"])  if not filtered.empty else 0
deterior  = len(filtered[filtered["risk_direction"] == "deteriorating"]) if not filtered.empty else 0
top_z     = filtered["top_z"].max() if not filtered.empty else 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total insights", total)
with c2:
    st.metric("🔴 Critical", critical,
              delta="action required" if critical > 0 else None,
              delta_color="inverse")
with c3:
    st.metric("🟡 Warnings", warning)
with c4:
    st.metric("📉 Deteriorating", deterior,
              delta="segments" if deterior > 0 else None,
              delta_color="inverse")
with c5:
    st.metric("Peak z-score", f"{top_z:+.2f}σ" if top_z else "—")

st.markdown("---")

# ---------------------------------------------------------------------------
# Main content — two columns
# ---------------------------------------------------------------------------

left, right = st.columns([3, 2], gap="large")

# ── LEFT: Insight cards ────────────────────────────────────────────────────
with left:
    st.markdown("### Anomaly insights")

    if filtered.empty:
        st.info("No insights match the current filters.")
    else:
        for _, row in filtered.iterrows():
            sev_color = SEVERITY_COLOR.get(row["severity"], "#888")
            sev_class = f"severity-{row['severity']}"
            seg_str   = fmt_segment(row["segment"]) if isinstance(row["segment"], dict) else str(row["segment"])
            direction = row.get("risk_direction", "unknown")
            risk_icon = RISK_ICON.get(direction, "❓")
            risk_color= RISK_COLOR.get(direction, "#888")
            z_class   = "z-score-positive" if row["top_z"] > 0 else "z-score-negative"

            with st.expander(
                f"{risk_icon}  {DIMENSION_LABELS.get(row['dimension'], row['dimension'].upper())}  "
                f"·  {seg_str}  ·  z={row['top_z']:+.2f}σ",
                expanded=(row["severity"] == "critical"),
            ):
                # Severity + direction badges
                col_a, col_b, col_c = st.columns([1, 1, 2])
                with col_a:
                    st.markdown(
                        f"<span style='background:{sev_color}22;color:{sev_color};"
                        f"padding:3px 10px;border-radius:4px;font-size:12px;"
                        f"font-weight:500'>{row['severity'].upper()}</span>",
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.markdown(
                        f"<span style='background:{risk_color}22;color:{risk_color};"
                        f"padding:3px 10px;border-radius:4px;font-size:12px'>"
                        f"{risk_icon} {direction}</span>",
                        unsafe_allow_html=True,
                    )
                with col_c:
                    st.markdown(
                        f"<span class='dimension-tag'>{row['dimension']}</span>",
                        unsafe_allow_html=True,
                    )

                st.markdown("")

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                mv   = round(float(row.get("metric_value", 0)), 2)
                mean = round(float(row.get("metric_mean",  0)), 2)
                vol  = int(row.get("volume", 0))
                z    = float(row.get("top_z", 0))
                delta_pct = round((mv - mean) / mean * 100, 1) if mean else 0

                with m1:
                    st.metric("Segment value", mv, f"{delta_pct:+.1f}% vs mean",
                              delta_color="inverse" if z > 0 else "normal")
                with m2:
                    st.metric("Portfolio mean", mean)
                with m3:
                    st.metric("Z-score", f"{z:+.2f}σ")
                with m4:
                    st.metric("Loans", f"{vol:,}")

                # Narrative
                narrative = row.get("narrative", "")
                if narrative:
                    # Strip emoji prefix from template narratives for cleaner display
                    clean = narrative.replace("🚨 ", "").strip()
                    st.markdown(
                        f"<div class='narrative-text'>{clean}</div>",
                        unsafe_allow_html=True,
                    )

                # Co-movers
                co_movers = row.get("co_movers", [])
                if co_movers and isinstance(co_movers, list) and len(co_movers) > 0:
                    st.markdown("**Co-moving metrics**")
                    co_df = pd.DataFrame([
                        {
                            "Metric":  m["label"],
                            "Value":   round(m["value"], 3),
                            "Z-score": f"{m['z_score']:+.2f}",
                            "Signal":  m["signal"],
                        }
                        for m in co_movers
                        if isinstance(m, dict)
                    ])
                    if not co_df.empty:
                        st.dataframe(
                            co_df,
                            use_container_width=True,
                            hide_index=True,
                        )

# ── RIGHT: Charts ───────────────────────────────────────────────────────────
with right:
    st.markdown("### Visual summary")

    if filtered.empty:
        st.info("No data to visualise.")
    else:
        # Chart 1 — Z-score bar chart
        st.markdown("**Anomaly severity by dimension**")
        chart_df = filtered[["dimension", "top_z", "severity", "risk_direction"]].copy()
        chart_df["label"] = chart_df["dimension"].map(
            lambda d: DIMENSION_LABELS.get(d, d)
        )
        chart_df["color"] = chart_df["severity"].map(SEVERITY_COLOR)
        chart_df = chart_df.sort_values("top_z", ascending=True)

        fig_bar = go.Figure(go.Bar(
            x          = chart_df["top_z"],
            y          = chart_df["label"],
            orientation= "h",
            marker_color = chart_df["color"],
            text       = chart_df["top_z"].apply(lambda z: f"{z:+.2f}σ"),
            textposition = "outside",
        ))
        fig_bar.update_layout(
            height          = 280,
            margin          = dict(l=0, r=40, t=10, b=10),
            paper_bgcolor   = "rgba(0,0,0,0)",
            plot_bgcolor    = "rgba(0,0,0,0)",
            font            = dict(color="#c8c8c8", size=12),
            xaxis           = dict(
                showgrid     = True,
                gridcolor    = "#1e1e1e",
                zeroline     = True,
                zerolinecolor= "#333",
                tickfont     = dict(family="DM Mono", size=11),
            ),
            yaxis = dict(tickfont=dict(size=11)),
            showlegend = False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Chart 2 — Risk direction donut
        st.markdown("**Risk direction breakdown**")
        dir_counts = filtered["risk_direction"].value_counts().reset_index()
        dir_counts.columns = ["direction", "count"]
        dir_counts["color"] = dir_counts["direction"].map(RISK_COLOR)

        fig_donut = go.Figure(go.Pie(
            labels      = dir_counts["direction"],
            values      = dir_counts["count"],
            hole        = 0.65,
            marker      = dict(colors=dir_counts["color"].tolist()),
            textinfo    = "label+percent",
            textfont    = dict(size=11, color="#c8c8c8"),
            showlegend  = False,
        ))
        fig_donut.update_layout(
            height        = 240,
            margin        = dict(l=0, r=0, t=10, b=10),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = dict(color="#c8c8c8"),
            annotations   = [dict(
                text      = f"<b>{len(filtered)}</b><br>insights",
                x=0.5, y=0.5,
                font_size = 14,
                font_color= "#c8c8c8",
                showarrow = False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Chart 3 — Metric value vs mean scatter
        if len(filtered) >= 2:
            st.markdown("**Segment value vs portfolio mean**")
            scatter_df = filtered.copy()
            scatter_df["segment_str"] = scatter_df["segment"].apply(
                lambda s: fmt_segment(s) if isinstance(s, dict) else str(s)
            )
            scatter_df["color"] = scatter_df["severity"].map(SEVERITY_COLOR)

            fig_scatter = go.Figure()
            for sev, grp in scatter_df.groupby("severity"):
                fig_scatter.add_trace(go.Scatter(
                    x    = grp["metric_mean"],
                    y    = grp["metric_value"],
                    mode = "markers+text",
                    name = sev,
                    marker = dict(
                        color = SEVERITY_COLOR.get(sev, "#888"),
                        size  = grp["top_z"].abs() * 5,
                        line  = dict(width=1, color="#0f0f0f"),
                    ),
                    text     = grp["segment_str"].apply(lambda s: s[:20]),
                    textposition = "top center",
                    textfont = dict(size=9, color="#888"),
                    hovertemplate = (
                        "<b>%{text}</b><br>"
                        "Value: %{y:.2f}<br>"
                        "Mean: %{x:.2f}<br>"
                        "<extra></extra>"
                    ),
                ))

            # Diagonal reference line
            all_vals = pd.concat([scatter_df["metric_mean"], scatter_df["metric_value"]])
            mn, mx = all_vals.min(), all_vals.max()
            fig_scatter.add_trace(go.Scatter(
                x    = [mn, mx],
                y    = [mn, mx],
                mode = "lines",
                line = dict(color="#333", dash="dot", width=1),
                showlegend = False,
                hoverinfo  = "skip",
            ))
            fig_scatter.update_layout(
                height        = 280,
                margin        = dict(l=0, r=0, t=10, b=30),
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                font          = dict(color="#c8c8c8", size=11),
                xaxis         = dict(
                    title     = "Portfolio mean",
                    showgrid  = True,
                    gridcolor = "#1e1e1e",
                    tickfont  = dict(family="DM Mono", size=10),
                ),
                yaxis         = dict(
                    title     = "Segment value",
                    showgrid  = True,
                    gridcolor = "#1e1e1e",
                    tickfont  = dict(family="DM Mono", size=10),
                ),
                legend = dict(
                    font      = dict(size=10),
                    bgcolor   = "rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Bottom section — Agent status + raw data
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Pipeline audit trail")

a1, a2 = st.columns([1, 2])

with a1:
    st.markdown("**Agent statuses**")
    statuses = report.get("agent_statuses", {})
    for agent, status in statuses.items():
        icon = "✅" if status.startswith("ok") else "⏭️" if status.startswith("skip") else "❌"
        st.markdown(
            f"`{agent:<14}` {icon} {status}",
            unsafe_allow_html=False,
        )

with a2:
    st.markdown("**Email receipts**")
    receipts = report.get("email_receipts", [])
    if receipts:
        rec_df = pd.DataFrame(receipts)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.info("No emails sent in this run.")

# Raw JSON expander
with st.expander("📄 Raw report JSON"):
    st.json(report)
