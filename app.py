import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score, confusion_matrix
)
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Analytics Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border: 1px solid #2e3555;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label { color: #8892b0 !important; font-size: 0.8rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #64ffda !important; }

    .section-header {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1.5rem 0 0.2rem 0;
    }
    .solar-divider {
        height: 2px;
        background: linear-gradient(90deg, #f7971e33, #ffd20066, #f7971e33);
        border: none;
        margin: 0.3rem 0 1.2rem 0;
        border-radius: 2px;
    }
    .insight-card {
        background: linear-gradient(135deg, #1a1f35, #1e2440);
        border-left: 4px solid #ffd200;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #ccd6f6;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117, #161b27); border-right: 1px solid #2e3555; }
    [data-testid="stSidebar"] label { color: #8892b0 !important; font-size: 0.78rem; }

    .stTabs [data-baseweb="tab-list"] { background: #1a1f35; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #f7971e, #ffd200) !important; color: #0f1117 !important; font-weight: 700; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

THEME = dict(
    plot_bgcolor="#161b27",
    paper_bgcolor="#1a1f35",
    font_color="#ccd6f6",
    title_font_color="#e2e8f0",
    colorway=["#ffd200", "#64ffda", "#f7971e", "#a78bfa", "#f472b6", "#38bdf8"],
)
_AXIS = dict(gridcolor="#2e3555", linecolor="#2e3555")

def _t(fig, height=None, xaxis_extra=None, yaxis_extra=None):
    kw = {**THEME, "xaxis": {**_AXIS, **(xaxis_extra or {})}, "yaxis": {**_AXIS, **(yaxis_extra or {})}}
    if height: kw["height"] = height
    fig.update_layout(**kw)
    return fig

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

@st.cache_data
def encode(df):
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.columns:
        if df_enc[col].dtype == object:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
    return df_enc, encoders

@st.cache_data
def clf_model(df_enc):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    y = df_enc["AdoptionLikelihood"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    metrics = {
        "Accuracy":  round(accuracy_score(y_te, pred), 4),
        "Precision": round(precision_score(y_te, pred, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(y_te, pred, average="weighted", zero_division=0), 4),
        "F1 Score":  round(f1_score(y_te, pred, average="weighted", zero_division=0), 4),
    }
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    cm  = confusion_matrix(y_te, pred)
    return metrics, imp, cm, y_te, pred

@st.cache_data
def reg_model(df_enc):
    drop_cols = ["AdoptionLikelihood", "EMI_Willingness"]
    X = df_enc.drop(columns=[c for c in drop_cols if c in df_enc.columns])
    y = df_enc["EMI_Willingness"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    return {"R² Score": round(r2_score(y_te, pred), 4), "MAE": round(mean_absolute_error(y_te, pred), 4)}, y_te.values, pred

@st.cache_data
def cluster_model(df_enc, k):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
    inertia = [KMeans(n_clusters=i, n_init=10, random_state=42).fit(Xs).inertia_ for i in range(2, 9)]
    return labels, inertia

@st.cache_data
def assoc_rules(df):
    cols = [c for c in ["Income", "Location", "AdoptionLikelihood"] if c in df.columns]
    db = pd.get_dummies(df[cols].astype(str)).astype(bool)
    try:
        freq = apriori(db, min_support=0.1, use_colnames=True)
        if freq.empty:
            return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq, metric="lift", min_threshold=1.0, num_itemsets=len(freq))
        return freq, rules.sort_values("lift", ascending=False).head(20)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️  `dataset.csv` not found. Make sure it is in the same folder as `app.py`.")
    st.stop()

df_enc, encoders = encode(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Solar Analytics")
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Filters")

    cat_cols = [c for c in df.select_dtypes("object").columns if df[c].nunique() <= 20]
    filters = {}
    for col in cat_cols[:4]:
        sel = st.selectbox(col, ["All"] + sorted(df[col].dropna().unique().tolist()))
        filters[col] = sel

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Settings")
    k = st.slider("K-Means Clusters", 2, 8, 4)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
    st.caption("Solar Venture Analytics · 2026")

# Apply filters
dff = df.copy()
for col, val in filters.items():
    if val != "All":
        dff = dff[dff[col] == val]

dff_enc, _ = encode(dff)
use_full = len(dff_enc) < 50   # fall back to full dataset if filter makes it too small

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#f7971e22,#ffd20011);border:1px solid #f7971e44;
            border-radius:16px;padding:24px 32px;margin-bottom:20px;'>
  <h1 style='color:#ffd200;margin:0;font-size:2.2rem;font-weight:800;'>☀️ Solar Market Analytics</h1>
  <p style='color:#8892b0;margin:6px 0 0 0;font-size:1rem;'>
      Consumer adoption intelligence · ML-powered insights · Pan-India market
  </p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3 = st.columns(3)
k1.metric("📊 Respondents", f"{len(dff):,}", f"{'Full dataset' if not filters or all(v=='All' for v in filters.values()) else 'Filtered'}")
if "AdoptionLikelihood" in dff.columns:
    top = dff["AdoptionLikelihood"].value_counts()
    k2.metric("🏆 Top Adoption", str(top.idxmax()), f"{top.max()/len(dff):.1%} of responses")
if "EMI_Willingness" in dff.columns:
    top_emi = dff["EMI_Willingness"].value_counts().idxmax()
    k3.metric("💰 Top EMI Preference", str(top_emi))

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📊 Exploration", "🤖 Classification", "📈 Regression", "🔵 Clustering", "🔗 Association Rules"
])

# ══════════════════════ TAB 1 ══════════════════════
with t1:
    st.markdown('<p class="section-header">Data Overview</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(dff.head(10), use_container_width=True, height=280)
    with c2:
        st.markdown("**Quick Stats**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Rows", "Columns", "Categorical cols", "Numeric cols", "Missing values"],
            "Value":  [len(dff), dff.shape[1],
                       dff.select_dtypes("object").shape[1],
                       dff.select_dtypes(np.number).shape[1],
                       int(dff.isnull().sum().sum())]
        }), use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Distribution Analysis</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    all_cols = dff.columns.tolist()
    default_idx = all_cols.index("AdoptionLikelihood") if "AdoptionLikelihood" in all_cols else 0
    sel = st.selectbox("Column to visualise", all_cols, index=default_idx)

    c1, c2 = st.columns(2)
    with c1:
        if dff[sel].dtype == object or dff[sel].nunique() <= 15:
            vc = dff[sel].value_counts().reset_index()
            vc.columns = [sel, "Count"]
            fig = px.bar(vc, x=sel, y="Count", title=f"{sel} — Count",
                         color="Count", color_continuous_scale=["#1a1f35","#ffd200"])
        else:
            fig = px.histogram(dff, x=sel, title=f"{sel} — Histogram",
                               color_discrete_sequence=["#ffd200"], nbins=20)
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if dff[sel].dtype == object or dff[sel].nunique() <= 15:
            vc = dff[sel].value_counts().reset_index()
            vc.columns = [sel, "Count"]
            fig2 = px.pie(vc, names=sel, values="Count", title=f"{sel} — Share",
                          color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig2.update_traces(textinfo="percent+label")
        else:
            fig2 = px.box(dff, y=sel, title=f"{sel} — Box Plot",
                          color_discrete_sequence=["#f7971e"])
        _t(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    num_cols = dff_enc.select_dtypes(np.number).columns.tolist()
    if len(num_cols) >= 3:
        st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
        st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
        corr = dff_enc[num_cols].corr()
        fig_c = px.imshow(corr, text_auto=".2f",
                          color_continuous_scale=["#0d1117","#1a1f35","#f7971e","#ffd200"],
                          title="Feature Correlation Matrix", aspect="auto")
        _t(fig_c)
        st.plotly_chart(fig_c, use_container_width=True)

# ══════════════════════ TAB 2 ══════════════════════
with t2:
    st.markdown('<p class="section-header">Adoption Likelihood Classifier</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    data_clf = df_enc if use_full else dff_enc
    if use_full:
        st.info("ℹ️ Too few rows after filtering — showing results on the full dataset.")

    with st.spinner("Training Random Forest…"):
        metrics_c, imp, cm, y_te_c, pred_c = clf_model(data_clf)

    m1, m2, m3, m4 = st.columns(4)
    for col_m, (name, val), icon in zip([m1,m2,m3,m4], metrics_c.items(), ["🎯","⚡","🔁","🏅"]):
        col_m.metric(f"{icon} {name}", f"{val:.2%}")

    c1, c2 = st.columns(2)
    with c1:
        fi = imp.reset_index(); fi.columns = ["Feature","Importance"]
        fig_fi = px.bar(fi.head(12), x="Importance", y="Feature", orientation="h",
                        title="Top 12 Predictive Features",
                        color="Importance", color_continuous_scale=["#1a1f35","#f7971e","#ffd200"])
        _t(fig_fi, height=420, yaxis_extra=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_fi, use_container_width=True)

    with c2:
        classes = [str(c) for c in sorted(set(y_te_c))]
        fig_cm = px.imshow(cm, text_auto=True, x=classes, y=classes,
                           labels=dict(x="Predicted", y="Actual"),
                           color_continuous_scale=["#0d1117","#1a1f35","#ffd200"],
                           title="Confusion Matrix")
        _t(fig_cm, height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="insight-card">💡 <b>Insight:</b> Features at the top of the importance chart are the strongest drivers of solar adoption. Focus awareness and sales efforts on respondents with high scores in these dimensions.</div>', unsafe_allow_html=True)

# ══════════════════════ TAB 3 ══════════════════════
with t3:
    st.markdown('<p class="section-header">EMI Willingness Predictor</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    if "EMI_Willingness" not in dff_enc.columns:
        st.info("EMI_Willingness column not found.")
    else:
        data_reg = df_enc if use_full else dff_enc
        with st.spinner("Training Regression Model…"):
            metrics_r, y_act, y_pred_r = reg_model(data_reg)

        c1, c2 = st.columns(2)
        c1.metric("📐 R² Score", f"{metrics_r['R² Score']:.4f}")
        c2.metric("📏 MAE", f"{metrics_r['MAE']:.4f}")

        c1, c2 = st.columns(2)
        with c1:
            pdf = pd.DataFrame({"Actual": y_act, "Predicted": y_pred_r})
            fig_sc = px.scatter(pdf, x="Actual", y="Predicted",
                                title="Actual vs Predicted EMI Willingness",
                                opacity=0.7, color_discrete_sequence=["#64ffda"])
            mn, mx = pdf.min().min(), pdf.max().max()
            fig_sc.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                             line=dict(color="#ffd200", dash="dash", width=2))
            _t(fig_sc)
            st.plotly_chart(fig_sc, use_container_width=True)

        with c2:
            resid = y_act - y_pred_r
            fig_r = px.histogram(x=resid, title="Residual Distribution",
                                 nbins=25, color_discrete_sequence=["#a78bfa"])
            fig_r.add_vline(x=0, line_dash="dash", line_color="#ffd200")
            _t(fig_r)
            st.plotly_chart(fig_r, use_container_width=True)

        st.markdown('<div class="insight-card">💡 <b>Insight:</b> Points tightly clustered along the dashed diagonal = strong predictions. A symmetric residual histogram centred at 0 confirms the model has no systematic bias.</div>', unsafe_allow_html=True)

# ══════════════════════ TAB 4 ══════════════════════
with t4:
    st.markdown('<p class="section-header">Consumer Segment Discovery</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    data_cl = df_enc if use_full else dff_enc
    with st.spinner("Running K-Means Clustering…"):
        labels, inertia = cluster_model(data_cl, k)

    dfc = (df if use_full else dff).copy()
    dfc["Cluster"] = [f"Segment {i+1}" for i in labels]

    c1, c2 = st.columns(2)
    with c1:
        sc = dfc["Cluster"].value_counts().reset_index(); sc.columns = ["Segment","Count"]
        fig_pie = px.pie(sc, names="Segment", values="Count", title="Respondents per Segment",
                         color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8","#34d399","#fb923c"])
        fig_pie.update_traces(textinfo="percent+label")
        _t(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_el = go.Figure()
        fig_el.add_trace(go.Scatter(x=list(range(2,9)), y=inertia, mode="lines+markers",
                                    line=dict(color="#ffd200", width=2),
                                    marker=dict(color="#f7971e", size=8)))
        fig_el.add_vline(x=k, line_dash="dash", line_color="#64ffda",
                         annotation_text=f"k={k}", annotation_font_color="#64ffda")
        _t(fig_el)
        fig_el.update_layout(title="Elbow Curve — Optimal K", xaxis_title="Clusters", yaxis_title="Inertia")
        st.plotly_chart(fig_el, use_container_width=True)

    if "AdoptionLikelihood" in dfc.columns:
        st.markdown("#### 📋 Adoption Mix by Segment")
        sp = dfc.groupby("Cluster")["AdoptionLikelihood"].value_counts(normalize=True).unstack(fill_value=0)
        fig_sp = px.bar(sp.reset_index(), x="Cluster", y=sp.columns.tolist(), barmode="stack",
                        title="Adoption Likelihood Distribution per Segment",
                        color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6"])
        _t(fig_sp)
        st.plotly_chart(fig_sp, use_container_width=True)

    st.markdown('<div class="insight-card">💡 <b>Insight:</b> Each segment is a distinct consumer archetype. Use the elbow curve to pick the ideal K — look for the "kink" where adding more clusters gives diminishing returns.</div>', unsafe_allow_html=True)

# ══════════════════════ TAB 5 ══════════════════════
with t5:
    st.markdown('<p class="section-header">Market Basket — Association Rules</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    data_ar = df if use_full else dff
    with st.spinner("Mining association rules…"):
        freq_df, rules_df = assoc_rules(data_ar)

    if rules_df.empty:
        st.info("No significant rules found with current filter. Try removing filters.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("🔗 Rules Found", len(rules_df))
        c2.metric("⚡ Avg Lift", f"{rules_df['lift'].mean():.2f}")
        c3.metric("📦 Frequent Itemsets", len(freq_df))

        c1, c2 = st.columns(2)
        with c1:
            fig_ar = px.scatter(rules_df, x="support", y="confidence", size="lift",
                                color="lift", hover_data=["antecedents","consequents"],
                                title="Support vs Confidence  (bubble = Lift)",
                                color_continuous_scale=["#1a1f35","#f7971e","#ffd200"])
            _t(fig_ar)
            st.plotly_chart(fig_ar, use_container_width=True)

        with c2:
            top10 = rules_df.head(10).copy()
            top10["rule"] = top10["antecedents"].astype(str) + " → " + top10["consequents"].astype(str)
            fig_lift = px.bar(top10, x="lift", y="rule", orientation="h",
                              title="Top 10 Rules by Lift",
                              color="lift", color_continuous_scale=["#1a1f35","#ffd200"])
            _t(fig_lift, height=420, yaxis_extra=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_lift, use_container_width=True)

        st.markdown("#### 📋 All Rules")
        disp = rules_df[["antecedents","consequents","support","confidence","lift"]].copy()
        disp["antecedents"] = disp["antecedents"].astype(str)
        disp["consequents"] = disp["consequents"].astype(str)
        for c in ["support","confidence","lift"]:
            disp[c] = disp[c].round(4)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown('<div class="insight-card">💡 <b>Insight:</b> Rules with lift > 1.5 reveal strong non-obvious affinities. Use these to build targeted cross-sell bundles — e.g. a specific income + location combination that strongly predicts adoption readiness.</div>', unsafe_allow_html=True)
