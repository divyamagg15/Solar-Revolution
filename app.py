import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score, confusion_matrix
)
from scipy import stats as scipy_stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        border: 1px solid #2e3555; border-radius: 12px;
        padding: 16px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label { color: #8892b0 !important; font-size: 0.8rem !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #64ffda !important; }
    .section-header {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 1.4rem; font-weight: 700; margin: 1.5rem 0 0.2rem 0;
    }
    .solar-divider {
        height: 2px;
        background: linear-gradient(90deg, #f7971e33, #ffd20066, #f7971e33);
        border: none; margin: 0.3rem 0 1.2rem 0; border-radius: 2px;
    }
    .insight-card {
        background: linear-gradient(135deg, #1a1f35, #1e2440);
        border-left: 4px solid #ffd200; border-radius: 8px;
        padding: 14px 18px; margin: 12px 0; color: #ccd6f6;
        font-size: 0.9rem; line-height: 1.6;
    }
    .stat-box {
        background: linear-gradient(135deg, #1a1f35, #1e2440);
        border: 1px solid #2e3555; border-radius: 10px;
        padding: 12px 16px; margin: 6px 0; color: #ccd6f6; font-size: 0.88rem;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117, #161b27); border-right: 1px solid #2e3555; }
    [data-testid="stSidebar"] label { color: #8892b0 !important; font-size: 0.78rem; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1f35; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #f7971e, #ffd200) !important; color: #0f1117 !important; font-weight: 700; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# THEME HELPERS
# ─────────────────────────────────────────────
THEME = dict(
    plot_bgcolor="#161b27", paper_bgcolor="#1a1f35",
    font_color="#ccd6f6", title_font_color="#e2e8f0",
    colorway=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8","#34d399","#fb923c"],
)
_AXIS  = dict(gridcolor="#2e3555", linecolor="#2e3555")
SOLAR_SCALE = ["#0d1117","#1a1f35","#f7971e","#ffd200"]
COOL_SCALE  = ["#0d1117","#1a1f35","#38bdf8","#64ffda"]

def _t(fig, height=None, xaxis_kw=None, yaxis_kw=None, **extra):
    kw = {**THEME, "xaxis": {**_AXIS, **(xaxis_kw or {})},
          "yaxis": {**_AXIS, **(yaxis_kw or {})}, **extra}
    if height: kw["height"] = height
    fig.update_layout(**kw)
    return fig

def sh(txt): st.markdown(f'<p class="section-header">{txt}</p>', unsafe_allow_html=True)
def sdiv():  st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
def icard(txt): st.markdown(f'<div class="insight-card">💡 <b>Insight:</b> {txt}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CACHED MODELS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

@st.cache_data
def encode(df):
    out = df.copy(); enc = {}
    for c in out.columns:
        if out[c].dtype == object:
            le = LabelEncoder()
            out[c] = le.fit_transform(out[c].astype(str))
            enc[c] = le
    return out, enc

@st.cache_data
def clf_model(df_enc):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    y = df_enc["AdoptionLikelihood"]
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(Xt, yt); pred = m.predict(Xe)
    metrics = {
        "Accuracy":  round(accuracy_score(ye, pred), 4),
        "Precision": round(precision_score(ye, pred, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(ye, pred, average="weighted", zero_division=0), 4),
        "F1 Score":  round(f1_score(ye, pred, average="weighted", zero_division=0), 4),
    }
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return metrics, imp, confusion_matrix(ye, pred), ye, pred

@st.cache_data
def dtree_model(df_enc):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    y = df_enc["AdoptionLikelihood"]
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(Xt, yt)
    return dt, X.columns.tolist(), accuracy_score(ye, dt.predict(Xe))

@st.cache_data
def reg_model(df_enc):
    drop = [c for c in ["AdoptionLikelihood","EMI_Willingness"] if c in df_enc.columns]
    X = df_enc.drop(columns=drop); y = df_enc["EMI_Willingness"]
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(Xt, yt); pred = m.predict(Xe)
    return {"R²": round(r2_score(ye, pred),4), "MAE": round(mean_absolute_error(ye,pred),4)}, ye.values, pred

@st.cache_data
def cluster_model(df_enc, k):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
    inertia = [KMeans(n_clusters=i, n_init=10, random_state=42).fit(Xs).inertia_ for i in range(2,9)]
    return labels, inertia

@st.cache_data
def assoc_rules(df):
    cols = [c for c in ["Income","Location","AdoptionLikelihood"] if c in df.columns]
    db = pd.get_dummies(df[cols].astype(str)).astype(bool)
    try:
        freq = apriori(db, min_support=0.1, use_colnames=True)
        if freq.empty: return pd.DataFrame(), pd.DataFrame()
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
    st.error("⚠️ `dataset.csv` not found. Ensure it's in the same directory as `app.py`.")
    st.stop()

df_enc, encoders = encode(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Solar Analytics")
    sdiv()
    st.markdown("### 🎛️ Filters")
    cat_cols = [c for c in df.select_dtypes("object").columns if df[c].nunique() <= 20]
    filters = {}
    for col in cat_cols[:4]:
        sel = st.selectbox(col, ["All"] + sorted(df[col].dropna().unique().tolist()))
        filters[col] = sel
    sdiv()
    st.markdown("### ⚙️ Model Settings")
    k = st.slider("K-Means Clusters", 2, 8, 4)
    sdiv()
    st.caption("Solar Venture Analytics · 2026")

dff = df.copy()
for col, val in filters.items():
    if val != "All": dff = dff[dff[col] == val]
dff_enc, _ = encode(dff)
use_full = len(dff_enc) < 50

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#f7971e22,#ffd20011);border:1px solid #f7971e44;
            border-radius:16px;padding:24px 32px;margin-bottom:20px;'>
  <h1 style='color:#ffd200;margin:0;font-size:2.2rem;font-weight:800;'>☀️ Solar Market Analytics</h1>
  <p style='color:#8892b0;margin:6px 0 0 0;font-size:1rem;'>
    Consumer adoption intelligence · ML-powered · Pan-India market · 15+ chart types
  </p>
</div>
""", unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
k1.metric("📊 Respondents", f"{len(dff):,}")
if "AdoptionLikelihood" in dff.columns:
    top = dff["AdoptionLikelihood"].value_counts()
    k2.metric("🏆 Top Adoption", str(top.idxmax()), f"{top.max()/len(dff):.1%}")
if "EMI_Willingness" in dff.columns:
    k3.metric("💰 Top EMI Pref.", str(dff["EMI_Willingness"].value_counts().idxmax()))
k4.metric("📋 Features", str(df.shape[1]-1))
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
t1,t2,t3,t4,t5,t6 = st.tabs([
    "📊 EDA & Charts",
    "🤖 Classification",
    "📈 Regression",
    "🔵 Clustering",
    "🔗 Association Rules",
    "🔬 Statistical Tools"
])

# ══════════════════════════════════════════════
# TAB 1 — EDA & ALL CHART TYPES
# ══════════════════════════════════════════════
with t1:
    sh("Data Overview")
    sdiv()
    c1,c2 = st.columns([2,1])
    with c1:
        st.dataframe(dff.head(10), use_container_width=True, height=260)
    with c2:
        st.markdown("**Quick Stats**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Rows","Columns","Categorical","Numeric","Missing"],
            "Value":  [len(dff), dff.shape[1],
                       dff.select_dtypes("object").shape[1],
                       dff.select_dtypes(np.number).shape[1],
                       int(dff.isnull().sum().sum())]
        }), use_container_width=True, hide_index=True)

    # ── INTERACTIVE PIE CHART ───────────────────
    sh("🥧 Interactive Pie Chart — Click a Slice to Drill Down")
    sdiv()
    pie_col_opts = [c for c in dff.columns if dff[c].dtype==object or dff[c].nunique()<=10]
    pie_col = st.selectbox("Select column for pie chart", pie_col_opts,
                           index=pie_col_opts.index("AdoptionLikelihood") if "AdoptionLikelihood" in pie_col_opts else 0,
                           key="pie_sel")
    vc_pie = dff[pie_col].value_counts().reset_index(); vc_pie.columns=[pie_col,"Count"]
    fig_pie = px.pie(vc_pie, names=pie_col, values="Count",
                     title=f"{pie_col} — Click a slice to filter the bar chart below",
                     color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8","#34d399","#fb923c"],
                     hole=0.35)
    fig_pie.update_traces(textinfo="percent+label", hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>")
    fig_pie.update_layout(**THEME, height=420)

    pie_event = st.plotly_chart(fig_pie, on_select="rerun", key="pie_chart", use_container_width=True)

    # Drill-down based on pie click
    selected_slice = None
    if pie_event and hasattr(pie_event, "selection") and pie_event.selection and pie_event.selection.get("points"):
        selected_slice = pie_event.selection["points"][0].get("label")

    if selected_slice:
        st.markdown(f'<div class="insight-card">🎯 <b>Drilled into: {pie_col} = "{selected_slice}"</b> — showing breakdown below</div>', unsafe_allow_html=True)
        dff_drill = dff[dff[pie_col].astype(str) == str(selected_slice)]
        drill_col = [c for c in dff.columns if c != pie_col and (dff[c].dtype==object or dff[c].nunique()<=12)]
        if drill_col:
            drill_on = st.selectbox("Break down by:", drill_col, key="drill_col")
            vc_d = dff_drill[drill_on].value_counts().reset_index(); vc_d.columns=[drill_on,"Count"]
            fig_d = px.bar(vc_d, x=drill_on, y="Count", title=f'"{selected_slice}" breakdown by {drill_on}',
                           color="Count", color_continuous_scale=SOLAR_SCALE)
            _t(fig_d, height=320)
            st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.caption("☝️ Click any slice in the pie chart above to drill down into that segment.")

    # ── CHART GALLERY ──────────────────────────
    sh("📊 Chart Gallery — All Visualisation Types")
    sdiv()

    all_cols   = dff.columns.tolist()
    cat_only   = [c for c in dff.columns if dff[c].dtype==object or dff[c].nunique()<=12]
    num_cols   = dff_enc.select_dtypes(np.number).columns.tolist()

    default_cat = "AdoptionLikelihood" if "AdoptionLikelihood" in cat_only else cat_only[0]
    default_num = "EMI_Willingness"    if "EMI_Willingness"    in num_cols else num_cols[0]

    chart_col  = st.selectbox("Categorical column for charts", cat_only, index=cat_only.index(default_cat), key="gal_cat")
    num_col    = st.selectbox("Numeric column for charts",     num_cols,  index=num_cols.index(default_num) if default_num in num_cols else 0, key="gal_num")

    r1c1, r1c2, r1c3 = st.columns(3)

    # 1. Bar chart
    with r1c1:
        vc = dff[chart_col].value_counts().reset_index(); vc.columns=[chart_col,"Count"]
        fig = px.bar(vc, x=chart_col, y="Count", title="📊 Bar Chart",
                     color="Count", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    # 2. Horizontal bar
    with r1c2:
        fig = px.bar(vc, x="Count", y=chart_col, orientation="h", title="📊 Horizontal Bar",
                     color="Count", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=320, yaxis_kw=dict(categoryorder="total ascending", gridcolor="#2e3555", linecolor="#2e3555"))
        st.plotly_chart(fig, use_container_width=True)

    # 3. Treemap
    with r1c3:
        fig = px.treemap(vc, path=[chart_col], values="Count", title="🗺️ Treemap",
                         color="Count", color_continuous_scale=SOLAR_SCALE)
        fig.update_layout(**THEME, height=320); st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2, r2c3 = st.columns(3)

    # 4. Sunburst
    with r2c1:
        sb_cols = [c for c in cat_only if c != chart_col][:1]
        if sb_cols:
            fig = px.sunburst(dff, path=[chart_col, sb_cols[0]], title="🌞 Sunburst Chart",
                              color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8"])
        else:
            fig = px.sunburst(vc, path=[chart_col], values="Count", title="🌞 Sunburst Chart",
                              color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa"])
        fig.update_layout(**THEME, height=320); st.plotly_chart(fig, use_container_width=True)

    # 5. Histogram
    with r2c2:
        fig = px.histogram(dff_enc, x=num_col, nbins=20, title="📈 Histogram",
                           color_discrete_sequence=["#ffd200"], marginal="rug")
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    # 6. Box plot
    with r2c3:
        fig = px.box(dff_enc, x=dff_enc.columns[dff_enc.columns.get_loc(chart_col) if chart_col in dff_enc.columns else 0],
                     y=num_col, title="📦 Box Plot",
                     color_discrete_sequence=["#f7971e"])
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    r3c1, r3c2, r3c3 = st.columns(3)

    # 7. Violin
    with r3c1:
        fig = px.violin(dff_enc, y=num_col, box=True, points="outliers",
                        title="🎻 Violin Plot", color_discrete_sequence=["#a78bfa"])
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    # 8. Scatter plot
    with r3c2:
        num2 = [c for c in num_cols if c != num_col]
        if num2:
            fig = px.scatter(dff_enc, x=num_col, y=num2[0], title="⚬ Scatter Plot",
                             opacity=0.6, color_discrete_sequence=["#64ffda"],
                             trendline="ols", trendline_color_override="#ffd200")
        else:
            fig = px.scatter(dff_enc, x=num_col, y=num_col, title="⚬ Scatter Plot",
                             opacity=0.6, color_discrete_sequence=["#64ffda"])
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    # 9. Bubble chart
    with r3c3:
        if len(num_cols) >= 2:
            bubble_df = dff_enc[num_cols[:3]].copy() if len(num_cols)>=3 else dff_enc[num_cols[:2]].copy()
            bubble_df = bubble_df.sample(min(200, len(bubble_df)), random_state=42)
            sz = num_cols[2] if len(num_cols)>=3 else num_cols[0]
            fig = px.scatter(bubble_df, x=num_cols[0], y=num_cols[1],
                             size=sz, title="🫧 Bubble Chart",
                             color=num_cols[0], color_continuous_scale=SOLAR_SCALE, opacity=0.7)
            _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    r4c1, r4c2, r4c3 = st.columns(3)

    # 10. Funnel chart
    with r4c1:
        vc_f = dff[chart_col].value_counts().reset_index(); vc_f.columns=[chart_col,"Count"]
        fig = px.funnel(vc_f, x="Count", y=chart_col, title="🔽 Funnel Chart",
                        color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6"])
        fig.update_layout(**THEME, height=320); st.plotly_chart(fig, use_container_width=True)

    # 11. Radar / Spider chart
    with r4c2:
        if len(num_cols) >= 4:
            radar_vals = dff_enc[num_cols[:6]].mean().values.tolist()
            radar_cats = num_cols[:6]
            radar_vals += [radar_vals[0]]
            radar_cats += [radar_cats[0]]
            fig = go.Figure(go.Scatterpolar(r=radar_vals, theta=radar_cats,
                                            fill='toself', line_color="#ffd200",
                                            fillcolor="rgba(247,151,30,0.2)"))
            fig.update_layout(**THEME, height=320, title="🕸️ Radar Chart",
                              polar=dict(bgcolor="#161b27",
                                         radialaxis=dict(gridcolor="#2e3555", linecolor="#2e3555", color="#8892b0"),
                                         angularaxis=dict(gridcolor="#2e3555", linecolor="#2e3555", color="#ccd6f6")))
            st.plotly_chart(fig, use_container_width=True)

    # 12. Area / Line chart (sorted counts)
    with r4c3:
        vc_l = dff[chart_col].value_counts().reset_index(); vc_l.columns=[chart_col,"Count"]
        fig = px.area(vc_l.sort_values(chart_col), x=chart_col, y="Count",
                      title="📉 Area Chart", color_discrete_sequence=["#64ffda"])
        fig.update_traces(fillcolor="rgba(100,255,218,0.15)")
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    # ── HEATMAP + PARALLEL COORDS ──────────────
    r5c1, r5c2 = st.columns(2)

    # 13. Correlation heatmap
    with r5c1:
        sh("🌡️ Correlation Heatmap"); sdiv()
        corr = dff_enc[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", title="Feature Correlation Matrix",
                        color_continuous_scale=SOLAR_SCALE, aspect="auto")
        _t(fig, height=420); st.plotly_chart(fig, use_container_width=True)

    # 14. Parallel coordinates
    with r5c2:
        sh("〰️ Parallel Coordinates"); sdiv()
        pc_cols = num_cols[:6]
        fig = px.parallel_coordinates(dff_enc[pc_cols], title="Parallel Coordinates Plot",
                                      color=dff_enc[num_cols[0]],
                                      color_continuous_scale=SOLAR_SCALE)
        fig.update_layout(**THEME, height=420); st.plotly_chart(fig, use_container_width=True)

    # 15. Stacked bar — two categorical
    if len(cat_only) >= 2:
        sh("📊 Stacked Bar — Two Categories"); sdiv()
        cat2 = [c for c in cat_only if c != chart_col][0]
        cross = dff.groupby([chart_col, cat2]).size().reset_index(name="Count")
        fig = px.bar(cross, x=chart_col, y="Count", color=cat2, barmode="stack",
                     title=f"{chart_col} × {cat2} — Stacked Bar",
                     color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8"])
        _t(fig, height=380); st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════════
with t2:
    sh("Random Forest Classifier"); sdiv()
    data_clf = df_enc if use_full else dff_enc
    if use_full: st.info("ℹ️ Too few rows after filtering — using full dataset.")
    with st.spinner("Training…"):
        mets, imp, cm, y_te_c, pred_c = clf_model(data_clf)

    m1,m2,m3,m4 = st.columns(4)
    for col_m,(name,val),icon in zip([m1,m2,m3,m4],mets.items(),["🎯","⚡","🔁","🏅"]):
        col_m.metric(f"{icon} {name}", f"{val:.2%}")

    c1,c2 = st.columns(2)
    with c1:
        fi = imp.reset_index(); fi.columns=["Feature","Importance"]
        fig = px.bar(fi.head(12), x="Importance", y="Feature", orientation="h",
                     title="🌟 Feature Importance (Top 12)",
                     color="Importance", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=400, yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        classes = [str(c) for c in sorted(set(y_te_c))]
        fig = px.imshow(cm, text_auto=True, x=classes, y=classes,
                        labels=dict(x="Predicted",y="Actual"),
                        color_continuous_scale=SOLAR_SCALE, title="🗺️ Confusion Matrix")
        _t(fig, height=400); st.plotly_chart(fig, use_container_width=True)

    # Decision Tree
    sh("🌳 Decision Tree Visualisation"); sdiv()
    with st.spinner("Training Decision Tree…"):
        dt, feat_names, dt_acc = dtree_model(data_clf)

    st.metric("🌳 Decision Tree Accuracy", f"{dt_acc:.2%}")

    # Plot Decision Tree as sankey-style using plotly
    tree_text = export_text(dt, feature_names=feat_names, max_depth=3)
    st.markdown(f"""<div class="stat-box"><pre style="color:#ccd6f6;font-size:0.75rem;overflow-x:auto;">{tree_text}</pre></div>""",
                unsafe_allow_html=True)

    # Feature importance comparison: RF vs DT
    dt_imp = pd.Series(dt.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
    rf_imp = imp.head(10)
    comp = pd.DataFrame({"Feature": dt_imp.index, "Decision Tree": dt_imp.values,
                          "Random Forest": rf_imp.reindex(dt_imp.index, fill_value=0).values})
    fig = px.bar(comp, x="Feature", y=["Decision Tree","Random Forest"], barmode="group",
                 title="🌳 vs 🌲 Decision Tree vs Random Forest — Feature Importance",
                 color_discrete_map={"Decision Tree":"#64ffda","Random Forest":"#ffd200"})
    _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)

    icard("The Random Forest aggregates 100 decision trees — its feature importance is more stable than a single tree. Compare both to see which features are consistently important.")

# ══════════════════════════════════════════════
# TAB 3 — REGRESSION
# ══════════════════════════════════════════════
with t3:
    sh("EMI Willingness Predictor"); sdiv()
    if "EMI_Willingness" not in dff_enc.columns:
        st.info("EMI_Willingness column not found.")
    else:
        data_reg = df_enc if use_full else dff_enc
        with st.spinner("Training…"):
            mets_r, y_act, y_pr = reg_model(data_reg)

        c1,c2 = st.columns(2)
        c1.metric("📐 R² Score", f"{mets_r['R²']:.4f}")
        c2.metric("📏 MAE", f"{mets_r['MAE']:.4f}")

        c1,c2 = st.columns(2)
        with c1:
            pdf = pd.DataFrame({"Actual":y_act,"Predicted":y_pr})
            fig = px.scatter(pdf, x="Actual", y="Predicted",
                             title="Actual vs Predicted", opacity=0.7,
                             color_discrete_sequence=["#64ffda"],
                             trendline="ols", trendline_color_override="#ffd200")
            mn,mx = pdf.min().min(), pdf.max().max()
            fig.add_shape(type="line", x0=mn,y0=mn,x1=mx,y1=mx,
                          line=dict(color="#f7971e",dash="dash",width=2))
            _t(fig); st.plotly_chart(fig, use_container_width=True)

        with c2:
            resid = y_act - y_pr
            fig = px.histogram(x=resid, nbins=25, title="Residual Distribution",
                               color_discrete_sequence=["#a78bfa"], marginal="box")
            fig.add_vline(x=0, line_dash="dash", line_color="#ffd200")
            _t(fig); st.plotly_chart(fig, use_container_width=True)

        # Residual vs Predicted
        c1,c2 = st.columns(2)
        with c1:
            fig = px.scatter(x=y_pr, y=resid, title="Residuals vs Predicted",
                             labels={"x":"Predicted","y":"Residual"},
                             opacity=0.6, color_discrete_sequence=["#f472b6"])
            fig.add_hline(y=0, line_dash="dash", line_color="#ffd200")
            _t(fig); st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Q-Q plot
            (osm, osr), (slope, intercept, r) = scipy_stats.probplot(resid)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                     marker=dict(color="#64ffda", size=5, opacity=0.6), name="Residuals"))
            fig.add_trace(go.Scatter(x=[min(osm),max(osm)],
                                     y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
                                     mode="lines", line=dict(color="#ffd200", dash="dash"), name="Normal line"))
            _t(fig, title="📐 Q-Q Plot (Normality Check)"); st.plotly_chart(fig, use_container_width=True)

        icard("The Q-Q plot checks if residuals are normally distributed — points close to the dashed line confirm the regression model's assumptions hold.")

# ══════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ══════════════════════════════════════════════
with t4:
    sh("Consumer Segment Discovery"); sdiv()
    data_cl = df_enc if use_full else dff_enc
    with st.spinner("Running K-Means…"):
        labels, inertia = cluster_model(data_cl, k)

    dfc = (df if use_full else dff).copy()
    dfc["Cluster"] = [f"Seg {i+1}" for i in labels]

    c1,c2 = st.columns(2)
    with c1:
        sc = dfc["Cluster"].value_counts().reset_index(); sc.columns=["Segment","Count"]
        fig = px.pie(sc, names="Segment", values="Count", title="Respondents per Segment",
                     color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8"],
                     hole=0.35)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(**THEME, height=380); st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,9)), y=inertia, mode="lines+markers",
                                 line=dict(color="#ffd200",width=2), marker=dict(color="#f7971e",size=8)))
        fig.add_vline(x=k, line_dash="dash", line_color="#64ffda",
                      annotation_text=f"k={k}", annotation_font_color="#64ffda")
        _t(fig, height=380, title="Elbow Curve")
        fig.update_layout(xaxis_title="Clusters", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)

    # Segment profile heatmap
    sh("Segment Feature Profiles"); sdiv()
    num_c = [c for c in df_enc.columns if c in dfc.columns and dfc[c].dtype in [np.int64, np.float64, int, float]]
    if num_c:
        seg_profile = dfc.groupby("Cluster")[num_c[:8]].mean()
        # Normalise for heatmap
        seg_norm = (seg_profile - seg_profile.min()) / (seg_profile.max() - seg_profile.min() + 1e-9)
        fig = px.imshow(seg_norm.T, text_auto=".2f",
                        color_continuous_scale=SOLAR_SCALE,
                        title="Segment Feature Heatmap (Normalised Mean Values)",
                        labels=dict(x="Segment", y="Feature", color="Norm. Value"))
        _t(fig, height=380); st.plotly_chart(fig, use_container_width=True)

    # Adoption mix per segment
    if "AdoptionLikelihood" in dfc.columns:
        sh("Adoption Mix by Segment"); sdiv()
        sp = dfc.groupby("Cluster")["AdoptionLikelihood"].value_counts(normalize=True).unstack(fill_value=0)
        fig = px.bar(sp.reset_index(), x="Cluster", y=sp.columns.tolist(), barmode="stack",
                     title="Adoption Likelihood Distribution per Segment",
                     color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6"])
        _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)

    icard("The segment heatmap reveals which features define each cluster. Bright cells = high relative values. Use this to build targeted product and pricing strategies per segment.")

# ══════════════════════════════════════════════
# TAB 5 — ASSOCIATION RULES
# ══════════════════════════════════════════════
with t5:
    sh("Market Basket — Association Rules"); sdiv()
    data_ar = df if use_full else dff
    with st.spinner("Mining rules…"):
        freq_df, rules_df = assoc_rules(data_ar)

    if rules_df.empty:
        st.info("No significant rules found. Try removing filters.")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("🔗 Rules Found", len(rules_df))
        c2.metric("⚡ Avg Lift", f"{rules_df['lift'].mean():.2f}")
        c3.metric("📦 Frequent Itemsets", len(freq_df))

        c1,c2 = st.columns(2)
        with c1:
            fig = px.scatter(rules_df, x="support", y="confidence", size="lift",
                             color="lift", hover_data=["antecedents","consequents"],
                             title="Support vs Confidence (size = Lift)",
                             color_continuous_scale=SOLAR_SCALE)
            _t(fig); st.plotly_chart(fig, use_container_width=True)

        with c2:
            top10 = rules_df.head(10).copy()
            top10["rule"] = top10["antecedents"].astype(str)+" → "+top10["consequents"].astype(str)
            fig = px.bar(top10, x="lift", y="rule", orientation="h",
                         title="Top 10 Rules by Lift",
                         color="lift", color_continuous_scale=SOLAR_SCALE)
            _t(fig, height=380, yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap: antecedent vs consequent lift
        sh("Lift Heatmap"); sdiv()
        h = rules_df.copy()
        h["ant_str"] = h["antecedents"].astype(str)
        h["con_str"] = h["consequents"].astype(str)
        pivot = h.pivot_table(index="ant_str", columns="con_str", values="lift", aggfunc="max").fillna(0)
        if not pivot.empty:
            fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale=SOLAR_SCALE,
                            title="Lift Heatmap — Antecedent × Consequent")
            _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📋 All Rules")
        disp = rules_df[["antecedents","consequents","support","confidence","lift"]].copy()
        disp["antecedents"] = disp["antecedents"].astype(str)
        disp["consequents"] = disp["consequents"].astype(str)
        for c in ["support","confidence","lift"]: disp[c] = disp[c].round(4)
        st.dataframe(disp, use_container_width=True, hide_index=True)

        icard("Rules with lift > 1.5 reveal strong purchasing affinities. The lift heatmap makes it easy to spot the strongest antecedent-consequent pairs at a glance.")

# ══════════════════════════════════════════════
# TAB 6 — STATISTICAL TOOLS
# ══════════════════════════════════════════════
with t6:
    sh("📐 Descriptive Statistics"); sdiv()
    num_enc_cols = dff_enc.select_dtypes(np.number).columns.tolist()
    desc = dff_enc[num_enc_cols].describe().T
    desc["skewness"] = dff_enc[num_enc_cols].skew().round(3)
    desc["kurtosis"] = dff_enc[num_enc_cols].kurtosis().round(3)
    desc = desc.round(3)
    st.dataframe(desc, use_container_width=True)

    # Skewness & Kurtosis bar charts
    c1,c2 = st.columns(2)
    with c1:
        skew_df = desc["skewness"].reset_index(); skew_df.columns=["Feature","Skewness"]
        fig = px.bar(skew_df, x="Feature", y="Skewness", title="📊 Skewness per Feature",
                     color="Skewness", color_continuous_scale=["#38bdf8","#1a1f35","#ffd200"])
        fig.add_hline(y=0, line_dash="dash", line_color="#64ffda")
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    with c2:
        kurt_df = desc["kurtosis"].reset_index(); kurt_df.columns=["Feature","Kurtosis"]
        fig = px.bar(kurt_df, x="Feature", y="Kurtosis", title="📊 Kurtosis per Feature",
                     color="Kurtosis", color_continuous_scale=["#a78bfa","#1a1f35","#f472b6"])
        fig.add_hline(y=3, line_dash="dash", line_color="#64ffda",
                      annotation_text="Normal=3", annotation_font_color="#64ffda")
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    # Outlier detection
    sh("🚨 Outlier Detection (IQR Method)"); sdiv()
    outlier_col = st.selectbox("Select feature for outlier analysis", num_enc_cols, key="out_col")
    col_data = dff_enc[outlier_col].dropna()
    Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = col_data[(col_data < lower) | (col_data > upper)]
    normal   = col_data[(col_data >= lower) & (col_data <= upper)]

    o1,o2,o3,o4 = st.columns(4)
    o1.metric("📊 Total Points", len(col_data))
    o2.metric("🚨 Outliers", len(outliers), f"{len(outliers)/len(col_data):.1%}")
    o3.metric("📉 Lower Fence", f"{lower:.2f}")
    o4.metric("📈 Upper Fence", f"{upper:.2f}")

    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(normal))),   y=normal.values,   mode="markers",
                                 marker=dict(color="#64ffda",size=4,opacity=0.5), name="Normal"))
        fig.add_trace(go.Scatter(x=list(range(len(outliers))), y=outliers.values, mode="markers",
                                 marker=dict(color="#f472b6",size=8,symbol="x"), name="Outlier"))
        fig.add_hline(y=upper, line_dash="dash", line_color="#ffd200", annotation_text="Upper fence")
        fig.add_hline(y=lower, line_dash="dash", line_color="#ffd200", annotation_text="Lower fence")
        _t(fig, height=350, title=f"Outlier Scatter — {outlier_col}")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(dff_enc, y=outlier_col, title=f"Box Plot — {outlier_col}",
                     color_discrete_sequence=["#f7971e"], points="outliers")
        _t(fig, height=350); st.plotly_chart(fig, use_container_width=True)

    # Chi-square test
    sh("📊 Chi-Square Independence Test"); sdiv()
    st.markdown("Test whether two categorical variables are statistically independent.")
    c1,c2 = st.columns(2)
    with c1:
        cat_a = st.selectbox("Variable A", cat_only, key="chi_a")
    with c2:
        cat_b_opts = [c for c in cat_only if c != cat_a]
        cat_b = st.selectbox("Variable B", cat_b_opts, key="chi_b")

    if cat_a and cat_b:
        ct = pd.crosstab(dff[cat_a], dff[cat_b])
        chi2, p, dof, expected = scipy_stats.chi2_contingency(ct)
        sig = "✅ Statistically significant (dependent)" if p < 0.05 else "❌ Not significant (independent)"
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("χ² Statistic", f"{chi2:.3f}")
        r2.metric("p-value", f"{p:.4f}")
        r3.metric("Degrees of Freedom", dof)
        r4.metric("Result (α=0.05)", sig)

        fig = px.imshow(ct, text_auto=True, title=f"Crosstab: {cat_a} × {cat_b}",
                        color_continuous_scale=SOLAR_SCALE, aspect="auto")
        _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)

        icard(f"p-value = {p:.4f}. {'The variables are NOT independent — they have a statistically significant relationship.' if p < 0.05 else 'No significant relationship found between these variables at the 5% level.'}")

    # Distribution comparison across categories
    sh("📈 Distribution Comparison"); sdiv()
    c1,c2 = st.columns(2)
    with c1:
        grp_col  = st.selectbox("Group by (categorical)", cat_only, key="dist_grp")
    with c2:
        val_col  = st.selectbox("Value column (numeric)",  num_enc_cols, key="dist_val")

    fig = px.violin(dff_enc if grp_col not in dff_enc.columns else dff,
                    x=grp_col, y=val_col if val_col in dff.columns else dff.columns[0],
                    box=True, points="outliers",
                    title=f"{val_col} distribution by {grp_col}",
                    color=grp_col,
                    color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8"])
    _t(fig, height=400); st.plotly_chart(fig, use_container_width=True)

    icard("Violin plots show the full distribution shape. A wider body = more data density there. Compare widths across groups to spot which segments show high variability in EMI willingness or adoption.")
