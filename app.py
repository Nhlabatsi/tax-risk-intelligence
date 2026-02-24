"""
Tax Compliance Risk Intelligence Platform
==========================================
Production Streamlit App — Clean Model, No Data Leakage
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tax Risk Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #111827 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p { color: #94a3b8 !important; }

/* ── Header ── */
.app-header {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #f8fafc;
    padding: 1rem 0 0.25rem;
}
.app-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1.5rem;
    font-weight: 300;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Metric Cards ── */
.metric-card {
    background: #0f172a;
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
}
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
}

/* ── Risk Badge ── */
.risk-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 0.6rem 1.6rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.risk-high   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1.5px solid rgba(239,68,68,0.4); }
.risk-medium { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1.5px solid rgba(245,158,11,0.4); }
.risk-low    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1.5px solid rgba(34,197,94,0.4); }

/* ── Section headers ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(59,130,246,0.2);
}

/* ── Input fields ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div {
    background: #0f172a !important;
    border-color: rgba(99,179,237,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(59,130,246,0.35) !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(99,179,237,0.1);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #64748b !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #1e40af !important;
    color: #e0f2fe !important;
}

/* ── Recommendation boxes ── */
.rec-high   { background: rgba(239,68,68,0.06);  border-left: 3px solid #ef4444; padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }
.rec-medium { background: rgba(245,158,11,0.06); border-left: 3px solid #f59e0b; padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }
.rec-low    { background: rgba(34,197,94,0.06);  border-left: 3px solid #22c55e; padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }

/* ── Divider ── */
hr { border-color: rgba(99,179,237,0.08) !important; margin: 1.5rem 0 !important; }

/* ── Info boxes ── */
.info-box {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #93c5fd;
    margin: 0.75rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("risk_model_clean.pkl")
    except FileNotFoundError:
        try:
            return joblib.load("risk_model.pkl")
        except FileNotFoundError:
            return None


# ─── Helpers ────────────────────────────────────────────────────────────────
def get_risk_class(level):
    mapping = {"High Risk": "high", "Medium Risk": "medium", "Low Risk": "low"}
    return mapping.get(level, "low")


def build_features(rev, exp, tax_liab, tax_paid, late, industry, profit, encoder):
    tax_gap          = tax_liab - tax_paid
    tax_payment_ratio = tax_paid / (tax_liab + 1)
    tax_underpaid    = int(tax_gap > 0)
    profit_margin    = profit / (rev + 1)
    expense_ratio    = exp / (rev + 1)
    late_filer       = int(late > 2)
    ind_encoded      = encoder.transform([industry])[0] if encoder else 0

    return {
        "Revenue": rev,
        "Expenses": exp,
        "Tax_Liability": tax_liab,
        "Tax_Paid": tax_paid,
        "Late_Filings": late,
        "Profit": profit,
        "Tax_Gap": tax_gap,
        "Tax_Payment_Ratio": tax_payment_ratio,
        "Tax_Underpaid": tax_underpaid,
        "Profit_Margin": profit_margin,
        "Expense_Ratio": expense_ratio,
        "Late_Filer": late_filer,
        "Industry_Encoded": ind_encoded,
    }


def predict(model_artifacts, feature_dict):
    model    = model_artifacts["model"]
    scaler   = model_artifacts["scaler"]
    feat_names = model_artifacts["feature_names"]

    row = pd.DataFrame([feature_dict])
    # align columns exactly as trained
    for col in feat_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feat_names]

    scaled = scaler.transform(row)
    label  = model.predict(scaled)[0]
    proba  = model.predict_proba(scaled)[0] if hasattr(model, "predict_proba") else None
    return label, proba


def gauge_chart(probability, risk_level):
    colors = {"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#22c55e"}
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9", "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569"}},
            "bar": {"color": colors.get(risk_level, "#3b82f6"), "thickness": 0.28},
            "bgcolor": "#0f172a",
            "bordercolor": "#1e293b",
            "steps": [
                {"range": [0, 33],  "color": "rgba(34,197,94,0.08)"},
                {"range": [33, 66], "color": "rgba(245,158,11,0.08)"},
                {"range": [66, 100],"color": "rgba(239,68,68,0.08)"},
            ],
        },
        title={"text": "Confidence", "font": {"size": 13, "color": "#475569", "family": "DM Sans"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#0f172a",
        font_color="#f1f5f9",
    )
    return fig


def importance_chart(model, feature_names, top_n=10):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    df_imp = pd.DataFrame({
        "Feature": [feature_names[i] for i in idx],
        "Importance": imp[idx],
    }).sort_values("Importance")

    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation="h",
        marker=dict(
            color=df_imp["Importance"],
            colorscale=[[0, "#1e3a5f"], [1, "#3b82f6"]],
            line=dict(width=0),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=350,
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", tickfont=dict(color="#64748b")),
        yaxis=dict(tickfont=dict(color="#cbd5e1")),
        font_color="#f1f5f9",
    )
    return fig


def recommendations(risk_level):
    items = {
        "High Risk": [
            ("🚨", "Schedule comprehensive audit immediately"),
            ("📞", "Contact taxpayer for formal explanation"),
            ("🔍", "Cross-reference all financial records"),
            ("⚠️", "Flag account for investigation team"),
            ("📋", "Review full filing history"),
        ],
        "Medium Risk": [
            ("📊", "Increase monitoring frequency"),
            ("📧", "Issue formal compliance reminder"),
            ("📈", "Track payment patterns next quarter"),
            ("📝", "Request supporting documentation"),
            ("🗓️", "Schedule review in 30 days"),
        ],
        "Low Risk": [
            ("✅", "Continue standard monitoring schedule"),
            ("📅", "Routine annual review sufficient"),
            ("📋", "Maintain current compliance status"),
        ],
    }
    css = get_risk_class(risk_level)
    st.markdown(f'<div class="section-title">Recommended Actions</div>', unsafe_allow_html=True)
    for icon, text in items.get(risk_level, []):
        st.markdown(f'<div class="rec-{css}">{icon} &nbsp; {text}</div>', unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.5rem;">
        <div style="font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    color:#f1f5f9;letter-spacing:-0.5px;">🏛️ TaxRisk AI</div>
        <div style="font-size:0.72rem;color:#475569;letter-spacing:1.5px;
                    text-transform:uppercase;margin-top:4px;">Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Single Prediction", "Batch Upload", "Model Insights"], label_visibility="collapsed")

    st.markdown('<div class="section-title">Risk Levels</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem;line-height:2;">
        🔴 &nbsp;<b>High Risk</b> — Audit Required<br>
        🟡 &nbsp;<b>Medium Risk</b> — Monitor Closely<br>
        🟢 &nbsp;<b>Low Risk</b> — Compliant
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Status</div>', unsafe_allow_html=True)
    artifacts = load_model()
    if artifacts:
        st.success("✓ Model Loaded")
        st.markdown(f"""
        <div style="font-size:0.78rem;color:#64748b;line-height:1.8;">
            Features: <b style="color:#94a3b8">{len(artifacts['feature_names'])}</b><br>
            Type: <b style="color:#94a3b8">{type(artifacts['model']).__name__}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Model not found")
        st.caption("Upload `risk_model_clean.pkl` to the app directory")

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#334155;text-align:center;">Amdocs Tax Compliance · 2025</div>', unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="app-header">Tax Compliance Risk Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">ML-Powered Taxpayer Risk Assessment System</div>', unsafe_allow_html=True)

if artifacts is None:
    st.error("Model file not found. Please upload `risk_model_clean.pkl` to the app directory.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
if page == "Single Prediction":

    st.markdown('<div class="section-title">Taxpayer Information</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**Financial Data**")
        taxpayer_id = st.text_input("Taxpayer ID", "TP-000001")

        industries = list(artifacts["industry_encoder"].classes_) if artifacts.get("industry_encoder") else ["Retail", "Finance", "Manufacturing", "Technology", "Healthcare"]
        industry = st.selectbox("Industry", industries)

        revenue  = st.number_input("Revenue (E)", min_value=0.0, value=850000.0, step=10000.0, format="%.2f")
        expenses = st.number_input("Expenses (E)", min_value=0.0, value=590000.0, step=10000.0, format="%.2f")
        profit   = revenue - expenses
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Calculated Profit</div>
            <div class="metric-value">E{profit:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("**Tax & Compliance**")
        tax_liability = st.number_input("Tax Liability (E)", min_value=0.0, value=85000.0, step=1000.0, format="%.2f")
        tax_paid      = st.number_input("Tax Paid (E)",      min_value=0.0, value=68000.0, step=1000.0, format="%.2f")

        tax_gap   = tax_liability - tax_paid
        pay_ratio = tax_paid / (tax_liability + 1) * 100

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tax Gap</div>
            <div class="metric-value" style="color:{'#f87171' if tax_gap > 0 else '#4ade80'}">
                E{tax_gap:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Payment Ratio</div>
            <div class="metric-value" style="color:{'#f87171' if pay_ratio < 90 else '#4ade80'}">
                {pay_ratio:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        late_filings = st.number_input("Late Filings (count)", min_value=0, value=2, step=1)

    st.markdown("---")
    predict_btn = st.button("⚡  Run Risk Assessment", use_container_width=True)

    if predict_btn:
        feat_dict = build_features(
            revenue, expenses, tax_liability, tax_paid,
            late_filings, industry, profit,
            artifacts.get("industry_encoder")
        )
        risk_level, proba = predict(artifacts, feat_dict)
        confidence = float(np.max(proba)) if proba is not None else 0.85

        st.markdown("---")
        st.markdown('<div class="section-title">Assessment Result</div>', unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1.8], gap="large")

        with res_col1:
            css = get_risk_class(risk_level)
            st.markdown(f"""
            <div style="padding:1rem 0;">
                <div class="metric-label">Risk Classification</div>
                <div class="risk-badge risk-{css}">{risk_level}</div>
                <div style="margin-top:1rem;font-size:0.8rem;color:#475569;">
                    Taxpayer: <span style="color:#94a3b8">{taxpayer_id}</span><br>
                    Industry: <span style="color:#94a3b8">{industry}</span><br>
                    Assessed: <span style="color:#94a3b8">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            fig_gauge = gauge_chart(confidence, risk_level)
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        with res_col3:
            recommendations(risk_level)

        # Key metrics row
        st.markdown('<div class="section-title">Key Risk Indicators</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Tax Gap", f"E{tax_gap:,.0f}", delta="Underpaid" if tax_gap > 0 else "Overpaid",
                      delta_color="inverse" if tax_gap > 0 else "normal")
        with m2:
            st.metric("Payment Ratio", f"{pay_ratio:.1f}%", delta="Below threshold" if pay_ratio < 90 else "Compliant",
                      delta_color="inverse" if pay_ratio < 90 else "normal")
        with m3:
            st.metric("Late Filings", late_filings, delta="High" if late_filings > 2 else "Normal",
                      delta_color="inverse" if late_filings > 2 else "normal")
        with m4:
            st.metric("Profit Margin", f"{(profit/revenue*100):.1f}%" if revenue > 0 else "N/A")

        # Download report
        st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
        report_dict = {
            "Taxpayer_ID": taxpayer_id, "Industry": industry, "Risk_Level": risk_level,
            "Confidence": f"{confidence:.1%}", "Revenue": revenue, "Expenses": expenses,
            "Profit": profit, "Tax_Liability": tax_liability, "Tax_Paid": tax_paid,
            "Tax_Gap": tax_gap, "Payment_Ratio": f"{pay_ratio:.1f}%",
            "Late_Filings": late_filings, "Assessment_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        report_df = pd.DataFrame([report_dict])
        csv_bytes = report_df.to_csv(index=False).encode()
        st.download_button("📥  Download Assessment Report", csv_bytes,
                           f"risk_assessment_{taxpayer_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                           "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Batch Upload":

    st.markdown('<div class="section-title">Batch Risk Assessment</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Upload a CSV containing taxpayer records. The model will assess each taxpayer's
        risk level in one pass and return a downloadable results file.
    </div>
    """, unsafe_allow_html=True)

    # Template download
    template_df = pd.DataFrame({
        "Taxpayer_ID": ["TP-000001", "TP-000002"],
        "Revenue": [850000, 1200000],
        "Expenses": [590000, 840000],
        "Tax_Liability": [85000, 120000],
        "Tax_Paid": [68000, 118000],
        "Late_Filings": [2, 0],
        "Industry": ["Retail", "Finance"],
        "Profit": [260000, 360000],
    })
    st.download_button("⬇️  Download CSV Template", template_df.to_csv(index=False).encode(),
                       "batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload Taxpayer CSV", type=["csv"])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(batch_df):,} records — preview:**")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("⚡  Assess All Taxpayers", use_container_width=True):
            with st.spinner(f"Processing {len(batch_df):,} taxpayer records…"):
                results_rows = []
                encoder = artifacts.get("industry_encoder")

                for _, row in batch_df.iterrows():
                    try:
                        rev  = float(row.get("Revenue", 0))
                        exp  = float(row.get("Expenses", 0))
                        liab = float(row.get("Tax_Liability", 0))
                        paid = float(row.get("Tax_Paid", 0))
                        late = int(row.get("Late_Filings", 0))
                        ind  = str(row.get("Industry", "Retail"))
                        prof = float(row.get("Profit", rev - exp))

                        if encoder and ind not in encoder.classes_:
                            ind = encoder.classes_[0]

                        feats = build_features(rev, exp, liab, paid, late, ind, prof, encoder)
                        level, proba = predict(artifacts, feats)
                        conf = float(np.max(proba)) if proba is not None else 0.85

                        results_rows.append({
                            "Taxpayer_ID": row.get("Taxpayer_ID", "N/A"),
                            "Industry": ind,
                            "Risk_Level": level,
                            "Confidence": f"{conf:.1%}",
                            "Tax_Gap": f"${liab - paid:,.0f}",
                            "Payment_Ratio": f"{paid/(liab+1)*100:.1f}%",
                        })
                    except Exception as e:
                        results_rows.append({
                            "Taxpayer_ID": row.get("Taxpayer_ID", "N/A"),
                            "Industry": "N/A", "Risk_Level": "Error",
                            "Confidence": "N/A", "Tax_Gap": "N/A", "Payment_Ratio": "N/A",
                        })

            results_df = pd.DataFrame(results_rows)

            # Summary metrics
            st.markdown('<div class="section-title">Batch Summary</div>', unsafe_allow_html=True)
            total = len(results_df)
            high  = (results_df["Risk_Level"] == "High Risk").sum()
            med   = (results_df["Risk_Level"] == "Medium Risk").sum()
            low   = (results_df["Risk_Level"] == "Low Risk").sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Assessed", total)
            c2.metric("🔴 High Risk",   high,  f"{high/total*100:.0f}%")
            c3.metric("🟡 Medium Risk", med,   f"{med/total*100:.0f}%")
            c4.metric("🟢 Low Risk",    low,   f"{low/total*100:.0f}%")

            # Distribution chart
            dist = results_df["Risk_Level"].value_counts().reset_index()
            dist.columns = ["Risk Level", "Count"]
            color_map = {"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#22c55e"}
            fig = px.bar(dist, x="Risk Level", y="Count", color="Risk Level",
                         color_discrete_map=color_map, template="plotly_dark",
                         title="Risk Level Distribution")
            fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                              showlegend=False, height=280,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Results table
            st.markdown('<div class="section-title">Full Results</div>', unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True)

            # Download
            st.download_button("📥  Download Full Results CSV",
                               results_df.to_csv(index=False).encode(),
                               f"batch_risk_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               "text/csv", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Model Insights":

    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    if "results" in artifacts:
        results = artifacts["results"]
        best_name = max(results, key=lambda x: results[x].get("cv_mean", results[x].get("f1_score", 0)))
        best = results[best_name]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model",  best_name.split()[0])
        c2.metric("Test Accuracy", f"{best.get('test_accuracy', best.get('accuracy', 0)):.1%}")
        c3.metric("F1 Score",    f"{best.get('f1_score', 0):.1%}")
        c4.metric("CV Score",    f"{best.get('cv_mean', 0):.1%}" if best.get("cv_mean") else "N/A")

        # Overfitting analysis
        if best.get("cv_mean") and best.get("test_accuracy"):
            st.markdown('<div class="section-title">Overfitting Analysis</div>', unsafe_allow_html=True)
            train_acc = best.get("train_accuracy", 0)
            test_acc  = best.get("test_accuracy", 0)
            cv_mean   = best.get("cv_mean", 0)
            gap       = train_acc - test_acc

            fig = go.Figure()
            model_names = list(results.keys())
            for metric, color, label in [
                ("train_accuracy", "#4ade80",  "Train"),
                ("test_accuracy",  "#3b82f6",  "Test"),
                ("cv_mean",        "#f59e0b",  "CV"),
            ]:
                vals = [results[n].get(metric, results[n].get("accuracy", 0)) for n in model_names]
                fig.add_trace(go.Bar(name=label, x=model_names, y=vals,
                                     marker_color=color, opacity=0.85))

            fig.update_layout(
                barmode="group", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                xaxis=dict(tickfont=dict(color="#94a3b8"), gridcolor="#1e293b"),
                yaxis=dict(tickfont=dict(color="#94a3b8"), gridcolor="#1e293b", range=[0, 1.1]),
                font_color="#f1f5f9",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            gap_color = "#f87171" if gap > 0.1 else "#fbbf24" if gap > 0.05 else "#4ade80"
            st.markdown(f"""
            <div class="info-box">
                Overfitting gap (Train−Test): 
                <b style="color:{gap_color}">{gap:.1%}</b> —
                {'⚠️ High' if gap > 0.1 else '⚡ Moderate' if gap > 0.05 else '✅ Well-controlled'}
            </div>
            """, unsafe_allow_html=True)

    # Feature importance
    model = artifacts["model"]
    feat_names = artifacts["feature_names"]

    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    fig_imp = importance_chart(model, feat_names)
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    # Features used / excluded
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Features Used (Leakage-Free)</div>', unsafe_allow_html=True)
        for f in feat_names:
            st.markdown(f"&nbsp; ✅ &nbsp; `{f}`")
    with col_b:
        excluded = artifacts.get("excluded_features", [])
        if excluded:
            st.markdown('<div class="section-title">Excluded (Leakage Risk)</div>', unsafe_allow_html=True)
            for f in excluded:
                st.markdown(f"&nbsp; ❌ &nbsp; `{f}`")

    st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <b>Data Leakage Prevention</b><br>
        This model was trained exclusively on pre-assessment features — data available
        <em>before</em> risk classification. Post-hoc features such as
        <code>Audit_Findings</code>, <code>Audit_to_Tax_Ratio</code>, and
        <code>Risk_Score</code> were excluded to prevent circular reasoning and
        ensure predictions generalise to unseen taxpayers.
    </div>
    """, unsafe_allow_html=True)
