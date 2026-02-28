"""
Customer Churn Intelligence Dashboard
======================================
Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="🔴",
    layout="wide",
)

# ── Brand palette ─────────────────────────────────────────────────────────────
COLOR_HIGH   = "#E63946"   # red   – high risk
COLOR_MED    = "#F4A261"   # amber – medium risk
COLOR_LOW    = "#2A9D8F"   # teal  – low risk
COLOR_ACCENT = "#264653"   # dark  – headers / borders

# ── Top banner ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {COLOR_ACCENT} 0%, #1b2e38 100%);
        border-radius: 10px;
        padding: 20px 32px 16px 32px;
        margin-bottom: 20px;
    ">
        <h1 style="color:#ffffff; margin:0; font-size:2rem;">
            🔴 Customer Churn Intelligence Dashboard
        </h1>
        <p style="color:#b0c4cc; margin:4px 0 0 0; font-size:1rem;">
            ML-powered churn risk scoring · Revenue-at-risk quantification · Retention ROI simulation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading champion model…")
def load_model():
    try:
        return joblib.load("champion_model.pkl")
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading test data…")
def load_data():
    try:
        X = pd.read_csv("test_features.csv")
        y = pd.read_csv("test_labels.csv")
        return X, y
    except FileNotFoundError:
        return None, None


@st.cache_data(show_spinner=False)
def load_feature_names():
    try:
        with open("feature_names.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


model = load_model()
X_test_raw, y_test_df = load_data()
feature_names = load_feature_names()

# ── Guard: missing files ──────────────────────────────────────────────────────
if model is None:
    st.error(
        "**champion_model.pkl not found.**  "
        "Please run the *Save Test Data & Champion Model* cell in `churn_analysis.ipynb` "
        "and refresh this page."
    )
    st.stop()

if X_test_raw is None or y_test_df is None:
    st.error(
        "**test_features.csv / test_labels.csv not found.**  "
        "Please run the *Save Test Data & Champion Model* cell in `churn_analysis.ipynb` "
        "and refresh this page."
    )
    st.stop()

# ── Score the test set ────────────────────────────────────────────────────────
with st.spinner("Scoring customers…"):
    churn_prob = model.predict_proba(X_test_raw)[:, 1]

actual_churn = y_test_df["actual_churn"].values

# ── Build master scored frame ─────────────────────────────────────────────────
df_scored = X_test_raw.copy()
df_scored["churn_probability"] = churn_prob
df_scored["actual_churn"]      = actual_churn

# customer_id
if "customer_id" in df_scored.columns:
    pass
elif "customerID" in df_scored.columns:
    df_scored.rename(columns={"customerID": "customer_id"}, inplace=True)
else:
    df_scored.insert(
        0, "customer_id",
        [f"CUST_{i+1:04d}" for i in range(len(df_scored))],
    )

# MRR detection
MRR_COL = None
for _c in ["mrr", "MonthlyCharges", "monthly_charges", "monthlycharges"]:
    if _c in df_scored.columns:
        MRR_COL = _c
        break
MRR_FALLBACK = 500.0

def get_mrr(row_series):
    return row_series[MRR_COL] if MRR_COL else MRR_FALLBACK

df_scored["_mrr_val"] = df_scored[MRR_COL] if MRR_COL else MRR_FALLBACK

# Risk tiers
def assign_tier(p, threshold):
    if p > threshold:
        return "High"
    elif p >= 0.40:
        return "Medium"
    return "Low"

# Revenue at risk (will be recalculated dynamically with threshold in sidebar)
df_scored["revenue_at_risk_mrr"] = df_scored["churn_probability"] * df_scored["_mrr_val"]
df_scored["revenue_at_risk_arr"] = df_scored["revenue_at_risk_mrr"] * 12

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{COLOR_ACCENT}; margin-bottom:2px;'>⚙️ Controls</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Set parameters, then click **Run Churn Analysis**.")

    run_analysis = st.button("🔍 Run Churn Analysis", type="primary", use_container_width=True)

    st.divider()

    risk_threshold = st.slider(
        "High Risk Threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05,
        format="%.2f",
        help="Customers with churn probability above this value are flagged as High Risk.",
    )

    top_n = st.slider(
        "Show Top N At-Risk Accounts",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
    )

    st.divider()

    st.markdown(
        f"<h3 style='color:{COLOR_ACCENT}; font-size:1rem;'>💡 Retention Intervention Simulator</h3>",
        unsafe_allow_html=True,
    )

    intervention_pct = st.slider(
        "Intervention Effectiveness",
        min_value=5,
        max_value=30,
        value=15,
        step=5,
        format="%d%%",
        help="Expected reduction in churn probability for targeted high-risk customers.",
    )

    campaign_cost_pct = st.slider(
        "Campaign Cost (% of ARR Saved)",
        min_value=10,
        max_value=30,
        value=20,
        step=5,
        format="%d%%",
    )

    st.divider()

    st.markdown(
        f"<h3 style='color:{COLOR_ACCENT}; font-size:1rem;'>🔎 Churn Driver Explorer</h3>",
        unsafe_allow_html=True,
    )

    # Only numeric feature columns
    numeric_feature_cols = [
        c for c in (feature_names if feature_names else X_test_raw.columns.tolist())
        if c in df_scored.columns and pd.api.types.is_numeric_dtype(df_scored[c])
    ]
    if not numeric_feature_cols:
        numeric_feature_cols = [
            c for c in df_scored.columns
            if c not in ("customer_id", "churn_probability", "actual_churn",
                         "risk_tier", "revenue_at_risk_mrr", "revenue_at_risk_arr", "_mrr_val")
            and pd.api.types.is_numeric_dtype(df_scored[c])
        ]

    selected_driver = st.selectbox(
        "Select a Churn Driver",
        options=numeric_feature_cols,
    )

# ── Apply threshold to risk tiers ─────────────────────────────────────────────
df_scored["risk_tier"] = df_scored["churn_probability"].apply(
    lambda p: assign_tier(p, risk_threshold)
)

TIER_COLOR = {"High": COLOR_HIGH, "Medium": COLOR_MED, "Low": COLOR_LOW}

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Overview",
    "💡 Intervention Simulator",
    "🔎 Churn Driver Explorer",
    "📈 Model Performance",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RISK OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if not run_analysis:
        st.info("👈 Click **🔍 Run Churn Analysis** in the sidebar to generate the risk report.")
    else:
        # ── KPI metrics ──────────────────────────────────────────────────────
        total_scored     = len(df_scored)
        high_risk_mask   = df_scored["risk_tier"] == "High"
        high_risk_count  = high_risk_mask.sum()
        high_risk_pct    = high_risk_count / total_scored * 100
        expected_arr_loss = df_scored.loc[high_risk_mask, "revenue_at_risk_arr"].sum()
        portfolio_churn  = df_scored["churn_probability"].mean() * 100

        # After intervention (using sidebar effectiveness)
        eff = intervention_pct / 100
        post_arr = df_scored["revenue_at_risk_arr"].copy()
        post_arr[high_risk_mask] *= (1 - eff)
        arr_saved_kpi  = expected_arr_loss - post_arr[high_risk_mask].sum()
        retained_kpi   = (df_scored.loc[high_risk_mask, "churn_probability"] * eff).sum()

        st.subheader("Portfolio Risk Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Customers Scored",
            f"{total_scored:,}",
        )
        c2.metric(
            "High Risk Customers",
            f"{high_risk_count:,}  ({high_risk_pct:.1f}%)",
            delta=f"−{retained_kpi:.0f} after intervention",
            delta_color="normal",
        )
        c3.metric(
            "Expected ARR Loss",
            f"${expected_arr_loss:,.0f}",
            delta=f"−${arr_saved_kpi:,.0f} saved w/ intervention",
            delta_color="normal",
        )
        c4.metric(
            "Portfolio Churn Rate",
            f"{portfolio_churn:.1f}%",
            delta=f"−{portfolio_churn * eff * high_risk_pct / 100:.1f}pp w/ intervention",
            delta_color="normal",
        )

        st.divider()

        # ── Row 2: Tier donut + Probability histogram ─────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            tier_counts = df_scored["risk_tier"].value_counts().reindex(
                ["High", "Medium", "Low"], fill_value=0
            )
            fig_pie = go.Figure(
                go.Pie(
                    labels=tier_counts.index.tolist(),
                    values=tier_counts.values.tolist(),
                    hole=0.45,
                    marker_colors=[COLOR_HIGH, COLOR_MED, COLOR_LOW],
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value:,} customers<extra></extra>",
                )
            )
            fig_pie.update_layout(
                title="Risk Tier Distribution",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                margin=dict(t=50, b=40, l=20, r=20),
                height=360,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            fig_hist = px.histogram(
                df_scored,
                x="churn_probability",
                nbins=40,
                color_discrete_sequence=["#4a90d9"],
                labels={"churn_probability": "Churn Probability"},
                title="Churn Probability Distribution",
            )
            fig_hist.add_vline(
                x=risk_threshold,
                line_dash="dash",
                line_color=COLOR_HIGH,
                annotation_text=f"Threshold: {risk_threshold:.2f}",
                annotation_position="top right",
            )
            fig_hist.update_layout(
                height=360,
                margin=dict(t=50, b=40, l=20, r=20),
                xaxis_title="Churn Probability",
                yaxis_title="# Customers",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # ── Row 3: Top-N at-risk table ────────────────────────────────────────
        st.subheader(f"Top {top_n} At-Risk Accounts")

        top_df = (
            df_scored[["customer_id", "churn_probability", "risk_tier",
                        "revenue_at_risk_mrr", "revenue_at_risk_arr"]]
            .nlargest(top_n, "churn_probability")
            .copy()
        )

        def _fmt_pct(v):
            return f"{v*100:.1f}%"

        def _fmt_usd(v):
            return f"${v:,.0f}"

        def _color_tier(v):
            colors = {"High": f"color: {COLOR_HIGH}; font-weight:bold",
                      "Medium": f"color: {COLOR_MED}; font-weight:bold",
                      "Low": f"color: {COLOR_LOW}; font-weight:bold"}
            return colors.get(v, "")

        display_df = top_df.rename(columns={
            "customer_id": "Customer ID",
            "churn_probability": "Churn Prob",
            "risk_tier": "Risk Tier",
            "revenue_at_risk_mrr": "MRR at Risk",
            "revenue_at_risk_arr": "ARR at Risk",
        })

        styled = (
            display_df.style
            .format({
                "Churn Prob": _fmt_pct,
                "MRR at Risk": _fmt_usd,
                "ARR at Risk": _fmt_usd,
            })
            .applymap(_color_tier, subset=["Risk Tier"])
        )
        st.dataframe(styled, use_container_width=True, height=380)

        # Download button
        csv_bytes = top_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Download At-Risk Customers CSV",
            data=csv_bytes,
            file_name="high_risk_customers.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — INTERVENTION SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("💡 Retention Intervention Simulator")
    st.caption(
        "Estimates the revenue impact of reducing predicted churn probabilities "
        "for high-risk customers through targeted retention campaigns."
    )

    high_risk_mask_sim = df_scored["churn_probability"] > risk_threshold
    baseline_arr_loss  = df_scored["revenue_at_risk_arr"].sum()
    baseline_churn_rt  = df_scored["churn_probability"].mean() * 100

    scenario_rows = []
    for eff_pct in [0, 0.10, 0.15, 0.20]:
        label = "Baseline" if eff_pct == 0 else f"{int(eff_pct*100)}% Reduction"
        new_proba = df_scored["churn_probability"].copy()
        new_proba[high_risk_mask_sim] *= (1 - eff_pct)

        new_arr_loss   = (new_proba * df_scored["_mrr_val"] * 12).sum()
        arr_saved      = baseline_arr_loss - new_arr_loss
        mrr_saved      = arr_saved / 12
        cust_retained  = (df_scored.loc[high_risk_mask_sim, "churn_probability"] * eff_pct).sum()
        new_churn_rt   = new_proba.mean() * 100
        campaign_cost  = arr_saved * (campaign_cost_pct / 100)
        net_benefit    = arr_saved - campaign_cost
        roi            = (net_benefit / campaign_cost) if campaign_cost > 0 else 0.0

        scenario_rows.append({
            "Scenario":            label,
            "ARR Loss ($)":        new_arr_loss,
            "ARR Saved ($)":       arr_saved,
            "MRR Saved ($)":       mrr_saved,
            "Customers Retained":  cust_retained,
            "Churn Rate":          new_churn_rt,
            "Campaign Cost ($)":   campaign_cost,
            "Net Benefit ($)":     net_benefit,
            "ROI Ratio":           roi,
            "_eff":                eff_pct,
        })

    sim_df = pd.DataFrame(scenario_rows)
    selected_label = "Baseline" if intervention_pct == 0 else f"{intervention_pct}% Reduction"

    def _highlight_selected(row):
        if row["Scenario"] == selected_label:
            return ["background-color: #d4f5e9"] * len(row)
        return [""] * len(row)

    display_sim = sim_df.drop(columns=["_eff"]).style.format({
        "ARR Loss ($)":       "${:,.0f}",
        "ARR Saved ($)":      "${:,.0f}",
        "MRR Saved ($)":      "${:,.0f}",
        "Customers Retained": "{:.1f}",
        "Churn Rate":         "{:.1f}%",
        "Campaign Cost ($)":  "${:,.0f}",
        "Net Benefit ($)":    "${:,.0f}",
        "ROI Ratio":          "{:.2f}x",
    }).apply(_highlight_selected, axis=1)

    st.dataframe(display_sim, use_container_width=True)

    st.divider()

    # ── Grouped bar: ARR Loss vs ARR Saved ────────────────────────────────────
    non_base = sim_df[sim_df["_eff"] > 0]
    col_a, col_b = st.columns(2)

    with col_a:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="ARR Loss",
            x=non_base["Scenario"],
            y=non_base["ARR Loss ($)"],
            marker_color=COLOR_HIGH,
        ))
        fig_bar.add_trace(go.Bar(
            name="ARR Saved",
            x=non_base["Scenario"],
            y=non_base["ARR Saved ($)"],
            marker_color=COLOR_LOW,
        ))
        fig_bar.update_layout(
            title="ARR Loss vs ARR Saved by Scenario",
            barmode="group",
            yaxis_title="ARR ($)",
            height=380,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=50, b=60),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_line = make_subplots(specs=[[{"secondary_y": True}]])
        eff_labels = [f"{int(r['_eff']*100)}%" for _, r in non_base.iterrows()]
        fig_line.add_trace(
            go.Scatter(
                x=eff_labels,
                y=non_base["ARR Saved ($)"].tolist(),
                mode="lines+markers",
                name="ARR Saved ($)",
                line=dict(color=COLOR_LOW, width=2.5),
                marker=dict(size=8),
            ),
            secondary_y=False,
        )
        fig_line.add_trace(
            go.Scatter(
                x=eff_labels,
                y=non_base["Customers Retained"].tolist(),
                mode="lines+markers",
                name="Customers Retained",
                line=dict(color=COLOR_ACCENT, width=2.5, dash="dot"),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )
        fig_line.update_layout(
            title="ARR Saved & Customers Retained vs Effectiveness",
            height=380,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=50, b=60),
        )
        fig_line.update_yaxes(title_text="ARR Saved ($)", secondary_y=False)
        fig_line.update_yaxes(title_text="Customers Retained", secondary_y=True)
        st.plotly_chart(fig_line, use_container_width=True)

    st.divider()

    # ── Executive Summary ─────────────────────────────────────────────────────
    sel_row = sim_df[sim_df["Scenario"] == selected_label].iloc[0]
    st.markdown(f"""
<div style="background-color:#e8f4f8; padding:20px; border-radius:10px; border-left:5px solid #1f77b4;">
    <h4 style="margin-top:0;">💼 Executive Summary</h4>
    <p>
    A <b>{intervention_pct}%</b> reduction in high-risk churn probability saves 
    <b>${sel_row['ARR Saved ($)']:,.0f} ARR</b> annually 
    (<b>${sel_row['MRR Saved ($)']:,.0f} MRR</b>), retaining approximately 
    <b>{sel_row['Customers Retained']:.0f} customers</b>.
    </p>
    <p>
    After a <b>{campaign_cost_pct}%</b> campaign budget assumption 
    (<b>${sel_row['Campaign Cost ($)']:,.0f}</b>), the net benefit is 
    <b>${sel_row['Net Benefit ($)']:,.0f}</b> — a <b>{sel_row['ROI Ratio']:.1f}x</b> 
    return on retention spend.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CHURN DRIVER EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader(f"Customers Most Affected by: {selected_driver}")

    if selected_driver not in df_scored.columns:
        st.warning(f"Feature **{selected_driver}** not found in the scored dataset.")
    elif not pd.api.types.is_numeric_dtype(df_scored[selected_driver]):
        st.warning(
            f"Feature **{selected_driver}** contains non-numeric values. "
            "Scatter plot is not available for categorical features."
        )
    else:
        # ── Scatter plot ──────────────────────────────────────────────────────
        scatter_df = df_scored[
            ["customer_id", selected_driver, "churn_probability",
             "risk_tier", "revenue_at_risk_arr"]
        ].copy()

        fig_scatter = px.scatter(
            scatter_df,
            x=selected_driver,
            y="churn_probability",
            color="risk_tier",
            size="revenue_at_risk_arr",
            size_max=20,
            hover_name="customer_id",
            color_discrete_map={
                "High":   COLOR_HIGH,
                "Medium": COLOR_MED,
                "Low":    COLOR_LOW,
            },
            category_orders={"risk_tier": ["High", "Medium", "Low"]},
            labels={
                selected_driver:    selected_driver,
                "churn_probability": "Churn Probability",
                "risk_tier":         "Risk Tier",
            },
            title=f"Churn Probability vs {selected_driver}",
        )
        fig_scatter.update_layout(height=440)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # ── Quartile impact stat box ──────────────────────────────────────────
        q75 = df_scored[selected_driver].quantile(0.75)
        q25 = df_scored[selected_driver].quantile(0.25)
        top_q_mean = df_scored.loc[df_scored[selected_driver] >= q75, "churn_probability"].mean()
        bot_q_mean = df_scored.loc[df_scored[selected_driver] <= q25, "churn_probability"].mean()

        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Top Quartile Mean Churn Prob", f"{top_q_mean*100:.1f}%")
        col_s2.metric("Bottom Quartile Mean Churn Prob", f"{bot_q_mean*100:.1f}%")
        col_s3.metric(
            "Driver Impact (Δ)",
            f"{(top_q_mean - bot_q_mean)*100:+.1f}pp",
            delta_color="inverse",
        )

        st.divider()

        # ── Top-20 table: high driver value customers ─────────────────────────
        top_quartile_df = (
            df_scored[df_scored[selected_driver] >= q75]
            [["customer_id", selected_driver, "churn_probability",
              "risk_tier", "revenue_at_risk_mrr", "revenue_at_risk_arr"]]
            .nlargest(20, "churn_probability")
            .copy()
        )

        st.caption(
            f"Top 20 customers by churn probability among those in the **top quartile** "
            f"of **{selected_driver}** (≥ {q75:.2f})"
        )

        def _tier_color(v):
            colors = {"High": f"color: {COLOR_HIGH}; font-weight:bold",
                      "Medium": f"color: {COLOR_MED}; font-weight:bold",
                      "Low": f"color: {COLOR_LOW}; font-weight:bold"}
            return colors.get(v, "")

        styled_top = (
            top_quartile_df.rename(columns={
                "customer_id": "Customer ID",
                "churn_probability": "Churn Prob",
                "risk_tier": "Risk Tier",
                "revenue_at_risk_mrr": "MRR at Risk",
                "revenue_at_risk_arr": "ARR at Risk",
            })
            .style
            .format({
                "Churn Prob":  lambda v: f"{v*100:.1f}%",
                "MRR at Risk": "${:,.0f}",
                "ARR at Risk": "${:,.0f}",
            })
            .applymap(_tier_color, subset=["Risk Tier"])
        )
        st.dataframe(styled_top, use_container_width=True)

        # Download button for driver-filtered list
        csv_driver = top_quartile_df.to_csv(index=False).encode()
        st.download_button(
            label=f"⬇️ Download Top-Quartile {selected_driver} Customers",
            data=csv_driver,
            file_name=f"high_{selected_driver}_customers.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📈 Model Performance Metrics")

    y_true  = df_scored["actual_churn"].values
    y_prob  = df_scored["churn_probability"].values
    y_pred  = (y_prob >= 0.5).astype(int)

    col1, col2 = st.columns(2)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    with col1:
        cm_vals = confusion_matrix(y_true, y_pred)
        labels  = ["Retained", "Churned"]
        total   = cm_vals.sum()
        annot   = [[f"{cm_vals[r][c]}<br>({cm_vals[r][c]/total:.1%})"
                    for c in range(2)] for r in range(2)]

        fig_cm = go.Figure(go.Heatmap(
            z=cm_vals,
            x=labels,
            y=labels,
            text=annot,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            title="Confusion Matrix (threshold = 0.50)",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=380,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    with col2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey", width=1),
            name="Random Chance (AUC = 0.50)",
            showlegend=True,
        ))
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"Champion Model (AUC = {roc_auc_val:.3f})",
            line=dict(color=COLOR_ACCENT, width=2.5),
        ))
        fig_roc.update_layout(
            title=f"ROC Curve  (AUC = {roc_auc_val:.3f})",
            xaxis=dict(title="False Positive Rate", range=[0, 1]),
            yaxis=dict(title="True Positive Rate", range=[0, 1]),
            height=380,
            legend=dict(x=0.5, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    col3, col4 = st.columns(2)

    # ── Precision-Recall Curve ────────────────────────────────────────────────
    with col3:
        pre_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
        baseline_pre = y_true.mean()

        fig_pr = go.Figure()
        fig_pr.add_hline(
            y=baseline_pre,
            line_dash="dash",
            line_color="grey",
            annotation_text=f"No-skill ({baseline_pre:.2f})",
            annotation_position="top right",
        )
        fig_pr.add_trace(go.Scatter(
            x=rec_vals, y=pre_vals,
            mode="lines",
            name="Champion Model",
            line=dict(color=COLOR_MED, width=2.5),
        ))
        fig_pr.update_layout(
            title="Precision-Recall Curve",
            xaxis=dict(title="Recall", range=[0, 1]),
            yaxis=dict(title="Precision", range=[0, 1]),
            height=380,
            legend=dict(x=0.5, y=0.9),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Classification Report ─────────────────────────────────────────────────
    with col4:
        report_dict = classification_report(
            y_true, y_pred,
            target_names=["Retained (0)", "Churned (1)"],
            output_dict=True,
        )
        report_df = pd.DataFrame(report_dict).T.drop(
            index=["accuracy"], errors="ignore"
        )
        # Keep only key rows
        keep_rows = ["Retained (0)", "Churned (1)", "macro avg", "weighted avg"]
        report_df = report_df.loc[[r for r in keep_rows if r in report_df.index]]

        st.markdown("**Classification Report**")
        st.dataframe(
            report_df.style.format({
                "precision": "{:.3f}",
                "recall":    "{:.3f}",
                "f1-score":  "{:.3f}",
                "support":   "{:.0f}",
            }).background_gradient(subset=["f1-score"], cmap="Greens"),
            use_container_width=True,
            height=200,
        )

        st.divider()

        # ── Quick summary stats ───────────────────────────────────────────────
        accuracy = (y_true == y_pred).mean()
        st.markdown(
            f"""
| Metric | Value |
|---|---|
| Test ROC-AUC | **{roc_auc_val:.4f}** |
| Accuracy (0.5 threshold) | **{accuracy*100:.1f}%** |
| Total Customers Scored | **{len(y_true):,}** |
| Actual Churn Rate | **{y_true.mean()*100:.1f}%** |
| Predicted Churn Rate | **{y_pred.mean()*100:.1f}%** |
            """
        )
