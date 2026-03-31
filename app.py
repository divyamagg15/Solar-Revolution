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
        padding: 14px 18px; margin: 10px 0; color: #ccd6f6;
        font-size: 0.88rem; line-height: 1.6;
    }
    .action-card {
        background: linear-gradient(135deg, #0d1f1a, #122a1e);
        border-left: 4px solid #64ffda; border-radius: 8px;
        padding: 14px 18px; margin: 10px 0; color: #ccd6f6;
        font-size: 0.88rem; line-height: 1.6;
    }
    .academic-card {
        background: linear-gradient(135deg, #1a1528, #201535);
        border-left: 4px solid #a78bfa; border-radius: 8px;
        padding: 14px 18px; margin: 10px 0; color: #ccd6f6;
        font-size: 0.88rem; line-height: 1.6;
    }
    .badge {
        display:inline-block; padding:2px 10px; border-radius:20px;
        font-size:0.72rem; font-weight:700; margin-right:6px;
    }
    .badge-biz  { background:#ffd20022; color:#ffd200; border:1px solid #ffd20055; }
    .badge-acad { background:#a78bfa22; color:#a78bfa; border:1px solid #a78bfa55; }
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
_AXIS       = dict(gridcolor="#2e3555", linecolor="#2e3555")
SOLAR_SCALE = ["#0d1117","#1a1f35","#f7971e","#ffd200"]
COOL_SCALE  = ["#0d1117","#1a1f35","#38bdf8","#64ffda"]
CAT_COLORS  = ["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8","#34d399","#fb923c"]

def _t(fig, height=None, xaxis_kw=None, yaxis_kw=None, **extra):
    kw = {**THEME, "xaxis":{**_AXIS,**(xaxis_kw or {})},
          "yaxis":{**_AXIS,**(yaxis_kw or {})}, **extra}
    if height: kw["height"] = height
    fig.update_layout(**kw); return fig

def sh(txt):    st.markdown(f'<p class="section-header">{txt}</p>', unsafe_allow_html=True)
def sdiv():     st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
def icard(t):   st.markdown(f'<div class="insight-card">💡 <b>Insight:</b> {t}</div>', unsafe_allow_html=True)
def acard(t):   st.markdown(f'<div class="action-card">🎯 <b>Action:</b> {t}</div>', unsafe_allow_html=True)
def acadcard(t):st.markdown(f'<div class="academic-card">📚 <b>Academic Note:</b> {t}</div>', unsafe_allow_html=True)
def badges(biz=False, acad=False):
    html = ""
    if biz:  html += '<span class="badge badge-biz">Business Relevant</span>'
    if acad: html += '<span class="badge badge-acad">Academic / Statistical</span>'
    if html: st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CACHED DATA & MODELS
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
            out[c] = le.fit_transform(out[c].astype(str)); enc[c] = le
    return out, enc

@st.cache_data
def clf_model(df_enc):
    X = df_enc.drop("AdoptionLikelihood", axis=1); y = df_enc["AdoptionLikelihood"]
    Xt,Xe,yt,ye = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(Xt,yt); pred = m.predict(Xe)
    mets = {"Accuracy":round(accuracy_score(ye,pred),4),
            "Precision":round(precision_score(ye,pred,average="weighted",zero_division=0),4),
            "Recall":round(recall_score(ye,pred,average="weighted",zero_division=0),4),
            "F1 Score":round(f1_score(ye,pred,average="weighted",zero_division=0),4)}
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return mets, imp, confusion_matrix(ye,pred), ye, pred

@st.cache_data
def dtree_model(df_enc):
    X = df_enc.drop("AdoptionLikelihood", axis=1); y = df_enc["AdoptionLikelihood"]
    Xt,Xe,yt,ye = train_test_split(X, y, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(Xt,yt)
    return dt, X.columns.tolist(), round(accuracy_score(ye,dt.predict(Xe)),4)

@st.cache_data
def reg_model(df_enc):
    drop = [c for c in ["AdoptionLikelihood","EMI_Willingness"] if c in df_enc.columns]
    X = df_enc.drop(columns=drop); y = df_enc["EMI_Willingness"]
    Xt,Xe,yt,ye = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestRegressor(n_estimators=100, random_state=42)
    m.fit(Xt,yt); pred = m.predict(Xe)
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"R²":round(r2_score(ye,pred),4),"MAE":round(mean_absolute_error(ye,pred),4)}, ye.values, pred, imp

@st.cache_data
def cluster_model(df_enc, k):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
    inertia = [KMeans(n_clusters=i,n_init=10,random_state=42).fit(Xs).inertia_ for i in range(2,9)]
    return labels, inertia

@st.cache_data
def assoc_rules_fn(df):
    cols = [c for c in ["Income","Location","AdoptionLikelihood"] if c in df.columns]
    db = pd.get_dummies(df[cols].astype(str)).astype(bool)
    try:
        freq = apriori(db, min_support=0.1, use_colnames=True)
        if freq.empty: return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq, metric="lift", min_threshold=1.0, num_itemsets=len(freq))
        return freq, rules.sort_values("lift", ascending=False).head(20)
    except Exception: return pd.DataFrame(), pd.DataFrame()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ `dataset.csv` not found. Ensure it is in the same directory as `app.py`.")
    st.stop()

df_enc, encoders = encode(df)
cat_cols = [c for c in df.select_dtypes("object").columns if df[c].nunique() <= 20]
num_enc_cols = df_enc.select_dtypes(np.number).columns.tolist()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Solar Analytics")
    sdiv()
    st.markdown("### 🎛️ Filters")
    filters = {}
    for col in cat_cols[:4]:
        sel = st.selectbox(col, ["All"] + sorted(df[col].dropna().unique().tolist()))
        filters[col] = sel
    sdiv()
    st.markdown("### ⚙️ Model Settings")
    k = st.slider("Customer Segments (K)", 2, 8, 4)
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
  <h1 style='color:#ffd200;margin:0;font-size:2.2rem;font-weight:800;'>☀️ Solar Market Analytics Dashboard</h1>
  <p style='color:#8892b0;margin:6px 0 0 0;font-size:1rem;'>
    Consumer adoption intelligence · ML-powered insights · Pan-India market
  </p>
</div>
""", unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
k1.metric("📊 Respondents", f"{len(dff):,}")
if "AdoptionLikelihood" in dff.columns:
    top = dff["AdoptionLikelihood"].value_counts()
    k2.metric("🏆 Top Adoption Intent", str(top.idxmax()), f"{top.max()/len(dff):.1%}")
if "EMI_Willingness" in dff.columns:
    k3.metric("💰 Most Common EMI", str(dff["EMI_Willingness"].value_counts().idxmax()))
if "Awareness" in dff.columns:
    hi = (dff["Awareness"]=="High").mean() if dff["Awareness"].dtype==object else dff["Awareness"].mean()
    k4.metric("📢 High Awareness", f"{hi:.1%}")
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
# TAB 1 — EDA & CHARTS
# ══════════════════════════════════════════════
with t1:
    sh("Data Overview")
    sdiv()
    c1,c2 = st.columns([2,1])
    with c1:
        st.dataframe(dff.head(10), use_container_width=True, height=260)
    with c2:
        st.markdown("**Dataset Stats**")
        st.dataframe(pd.DataFrame({
            "Metric":["Rows","Columns","Categorical","Numeric","Missing"],
            "Value":[len(dff), dff.shape[1],
                     dff.select_dtypes("object").shape[1],
                     dff.select_dtypes(np.number).shape[1],
                     int(dff.isnull().sum().sum())]
        }), use_container_width=True, hide_index=True)

    # ── INTERACTIVE PIE ──────────────────────
    sh("🥧 Interactive Pie Chart — Click a Slice to Drill Down")
    sdiv()
    badges(biz=True, acad=True)
    pie_opts = [c for c in cat_cols if c in dff.columns]
    default_pie = "AdoptionLikelihood" if "AdoptionLikelihood" in pie_opts else pie_opts[0]
    pie_col = st.selectbox("Select dimension", pie_opts, index=pie_opts.index(default_pie), key="pie_sel")
    vc_pie = dff[pie_col].value_counts().reset_index(); vc_pie.columns=[pie_col,"Count"]
    fig_pie = px.pie(vc_pie, names=pie_col, values="Count",
                     title=f"{pie_col} — Click a slice to drill into that group",
                     color_discrete_sequence=CAT_COLORS, hole=0.38)
    fig_pie.update_traces(textinfo="percent+label",
                          hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>")
    fig_pie.update_layout(**THEME, height=420)
    pie_event = st.plotly_chart(fig_pie, on_select="rerun", key="pie_main", use_container_width=True)

    selected_slice = None
    if pie_event and hasattr(pie_event,"selection") and pie_event.selection.get("points"):
        selected_slice = pie_event.selection["points"][0].get("label")
    if selected_slice:
        dff_drill = dff[dff[pie_col].astype(str)==str(selected_slice)]
        st.markdown(f'<div class="insight-card">🎯 <b>{len(dff_drill)} respondents</b> where <b>{pie_col} = "{selected_slice}"</b></div>', unsafe_allow_html=True)
        drill_opts = [c for c in cat_cols if c!=pie_col and c in dff.columns]
        if drill_opts:
            drill_on = st.selectbox("Break this group down by:", drill_opts, key="drill_col")
            vc_d = dff_drill[drill_on].value_counts().reset_index(); vc_d.columns=[drill_on,"Count"]
            col1,col2 = st.columns(2)
            with col1:
                fig_d = px.bar(vc_d, x=drill_on, y="Count", title=f'"{selected_slice}" → {drill_on}',
                               color="Count", color_continuous_scale=SOLAR_SCALE)
                _t(fig_d, height=300); st.plotly_chart(fig_d, use_container_width=True)
            with col2:
                fig_d2 = px.pie(vc_d, names=drill_on, values="Count",
                                color_discrete_sequence=CAT_COLORS, hole=0.35,
                                title=f'"{selected_slice}" → {drill_on} share')
                fig_d2.update_traces(textinfo="percent+label")
                fig_d2.update_layout(**THEME, height=300)
                st.plotly_chart(fig_d2, use_container_width=True)
    else:
        st.caption("☝️ Click any slice above to drill into that group.")

    # ── CHART GALLERY ───────────────────────
    sh("📊 Complete Chart Gallery")
    sdiv()
    st.markdown("*Select columns to explore all visualisation types:*")
    gc1,gc2 = st.columns(2)
    with gc1:
        gcat = st.selectbox("Categorical column", cat_cols,
                            index=cat_cols.index("AdoptionLikelihood") if "AdoptionLikelihood" in cat_cols else 0,
                            key="gal_cat")
    with gc2:
        gnum = st.selectbox("Numeric column (encoded)", num_enc_cols,
                            index=num_enc_cols.index("EMI_Willingness") if "EMI_Willingness" in num_enc_cols else 0,
                            key="gal_num")

    vc = dff[gcat].value_counts().reset_index(); vc.columns=[gcat,"Count"]

    # Row 1
    r1c1,r1c2,r1c3 = st.columns(3)
    with r1c1:
        badges(biz=True, acad=True)
        fig = px.bar(vc, x=gcat, y="Count", title="📊 Bar Chart",
                     color="Count", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
    with r1c2:
        badges(acad=True)
        fig = px.bar(vc, x="Count", y=gcat, orientation="h", title="📊 Horizontal Bar Chart",
                     color="Count", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=300, yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
        st.plotly_chart(fig, use_container_width=True)
    with r1c3:
        badges(biz=True, acad=True)
        fig = px.treemap(vc, path=[gcat], values="Count", title="🗺️ Treemap",
                         color="Count", color_continuous_scale=SOLAR_SCALE)
        fig.update_layout(**THEME, height=300); st.plotly_chart(fig, use_container_width=True)

    # Row 2
    r2c1,r2c2,r2c3 = st.columns(3)
    with r2c1:
        badges(biz=True, acad=True)
        sb2 = [c for c in cat_cols if c!=gcat]
        if sb2:
            fig = px.sunburst(dff, path=[gcat, sb2[0]], title="🌞 Sunburst Chart",
                              color_discrete_sequence=CAT_COLORS)
        else:
            fig = px.sunburst(vc, path=[gcat], values="Count", title="🌞 Sunburst Chart",
                              color_discrete_sequence=CAT_COLORS)
        fig.update_layout(**THEME, height=300); st.plotly_chart(fig, use_container_width=True)
    with r2c2:
        badges(acad=True)
        fig = px.histogram(dff_enc, x=gnum, nbins=20, title="📈 Histogram",
                           color_discrete_sequence=["#ffd200"], marginal="rug")
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
    with r2c3:
        badges(acad=True)
        fig = px.box(dff_enc, y=gnum, title="📦 Box Plot",
                     color_discrete_sequence=["#f7971e"], points="outliers")
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)

    # Row 3
    r3c1,r3c2,r3c3 = st.columns(3)
    with r3c1:
        badges(acad=True)
        fig = px.violin(dff_enc, y=gnum, box=True, points="outliers",
                        title="🎻 Violin Plot", color_discrete_sequence=["#a78bfa"])
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
    with r3c2:
        badges(acad=True)
        num2 = [c for c in num_enc_cols if c!=gnum]
        if num2:
            fig = px.scatter(dff_enc, x=gnum, y=num2[0], opacity=0.6,
                             title="⚬ Scatter Plot", color_discrete_sequence=["#64ffda"])
        else:
            fig = px.scatter(dff_enc, x=gnum, y=gnum, opacity=0.6,
                             title="⚬ Scatter Plot", color_discrete_sequence=["#64ffda"])
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
    with r3c3:
        badges(acad=True)
        if len(num_enc_cols) >= 2:
            bdf = dff_enc[num_enc_cols[:3]].sample(min(200,len(dff_enc)), random_state=42)
            sz  = num_enc_cols[2] if len(num_enc_cols)>=3 else num_enc_cols[0]
            fig = px.scatter(bdf, x=num_enc_cols[0], y=num_enc_cols[1], size=sz,
                             color=num_enc_cols[0], title="🫧 Bubble Chart",
                             color_continuous_scale=SOLAR_SCALE, opacity=0.7)
            _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)

    # Row 4
    r4c1,r4c2,r4c3 = st.columns(3)
    with r4c1:
        badges(acad=True)
        fig = px.funnel(vc, x="Count", y=gcat, title="🔽 Funnel Chart",
                        color_discrete_sequence=CAT_COLORS)
        fig.update_layout(**THEME, height=300); st.plotly_chart(fig, use_container_width=True)
        acadcard("Funnel charts show magnitude progressively — useful for visualising stages of adoption or sales pipeline drop-off.")
    with r4c2:
        badges(acad=True)
        if len(num_enc_cols) >= 4:
            rv = dff_enc[num_enc_cols[:6]].mean().values.tolist()
            rc = num_enc_cols[:6]; rv += [rv[0]]; rc += [rc[0]]
            fig = go.Figure(go.Scatterpolar(r=rv, theta=rc, fill='toself',
                                            line_color="#ffd200", fillcolor="rgba(247,151,30,0.18)"))
            fig.update_layout(**THEME, height=300, title="🕸️ Radar / Spider Chart",
                              polar=dict(bgcolor="#161b27",
                                         radialaxis=dict(gridcolor="#2e3555",linecolor="#2e3555",color="#8892b0"),
                                         angularaxis=dict(gridcolor="#2e3555",linecolor="#2e3555",color="#ccd6f6")))
            st.plotly_chart(fig, use_container_width=True)
        acadcard("Radar charts compare a profile across multiple dimensions simultaneously — useful for segment profiling.")
    with r4c3:
        badges(biz=True, acad=True)
        vc_l = dff[gcat].value_counts().reset_index(); vc_l.columns=[gcat,"Count"]
        fig = px.area(vc_l.sort_values(gcat), x=gcat, y="Count", title="📉 Area Chart",
                      color_discrete_sequence=["#64ffda"])
        fig.update_traces(fillcolor="rgba(100,255,218,0.12)")
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)

    # Row 5 — Stacked + Grouped
    sh("Cross-Category Analysis")
    sdiv()
    badges(biz=True, acad=True)
    xc1,xc2 = st.columns(2)
    with xc1: col_x = st.selectbox("X-axis", cat_cols, key="cx")
    with xc2:
        col_y_opts=[c for c in cat_cols if c!=col_x]
        col_y = st.selectbox("Colour / group by", col_y_opts, key="cy")
    cross = dff.groupby([col_x,col_y]).size().reset_index(name="Count")
    sc1,sc2 = st.columns(2)
    with sc1:
        fig = px.bar(cross, x=col_x, y="Count", color=col_y, barmode="stack",
                     title=f"{col_x} × {col_y} — Stacked Bar", color_discrete_sequence=CAT_COLORS)
        _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)
    with sc2:
        fig = px.bar(cross, x=col_x, y="Count", color=col_y, barmode="group",
                     title=f"{col_x} × {col_y} — Grouped Bar", color_discrete_sequence=CAT_COLORS)
        _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)

    # Row 6 — Sunburst (hierarchical) + Parallel Coords
    hc1,hc2 = st.columns(2)
    with hc1:
        sh("Hierarchical Sunburst"); sdiv(); badges(biz=True, acad=True)
        sb_opts = cat_cols
        sl1 = st.selectbox("Level 1", sb_opts, key="sb1",
                           index=sb_opts.index("Location") if "Location" in sb_opts else 0)
        sl2_opts = [c for c in sb_opts if c!=sl1]
        sl2 = st.selectbox("Level 2", sl2_opts, key="sb2",
                           index=sl2_opts.index("AdoptionLikelihood") if "AdoptionLikelihood" in sl2_opts else 0)
        fig = px.sunburst(dff, path=[sl1,sl2], title=f"{sl1} → {sl2}",
                          color_discrete_sequence=CAT_COLORS)
        fig.update_layout(**THEME, height=420)
        st.plotly_chart(fig, use_container_width=True)
    with hc2:
        sh("Parallel Coordinates"); sdiv(); badges(acad=True)
        pc_cols = num_enc_cols[:6]
        fig = px.parallel_coordinates(dff_enc[pc_cols],
                                      color=dff_enc[num_enc_cols[0]],
                                      color_continuous_scale=SOLAR_SCALE,
                                      title="All numeric features simultaneously")
        fig.update_layout(**THEME, height=420)
        st.plotly_chart(fig, use_container_width=True)
        acadcard("Parallel coordinates display every feature on its own vertical axis. Lines that cluster together represent similar respondents — a multivariate exploration tool.")

    # Correlation heatmap
    sh("Correlation Heatmap")
    sdiv(); badges(biz=True, acad=True)
    corr = dff_enc.corr()
    fig = px.imshow(corr, text_auto=".2f", title="Feature Correlation Matrix",
                    color_continuous_scale=SOLAR_SCALE, aspect="auto")
    _t(fig, height=460); st.plotly_chart(fig, use_container_width=True)
    icard("Values close to ±1 indicate strong linear relationships. Scan the AdoptionLikelihood row to see which features are most predictive.")

# ══════════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════════
with t2:
    sh("Random Forest — Adoption Likelihood Classifier")
    sdiv()
    data_clf = df_enc if use_full else dff_enc
    if use_full: st.info("ℹ️ Too few rows after filtering — using full dataset.")

    with st.spinner("Training Random Forest…"):
        mets, imp, cm, y_te, pred_c = clf_model(data_clf)

    m1,m2,m3,m4 = st.columns(4)
    for cm_,(name,val),icon in zip([m1,m2,m3,m4],mets.items(),["🎯","⚡","🔁","🏅"]):
        cm_.metric(f"{icon} {name}", f"{val:.2%}")

    c1,c2 = st.columns(2)
    with c1:
        badges(biz=True, acad=True)
        fi = imp.reset_index(); fi.columns=["Factor","Importance"]
        fig = px.bar(fi.head(10), x="Importance", y="Factor", orientation="h",
                     title="🔑 What drives adoption? (Feature Importance)",
                     color="Importance", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=400, yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
        st.plotly_chart(fig, use_container_width=True)
        acard("Target customers with high scores on the top 2-3 factors — they have the strongest propensity to adopt solar.")

    with c2:
        badges(acad=True)
        classes=[str(c) for c in sorted(set(y_te))]
        fig = px.imshow(cm, text_auto=True, x=classes, y=classes,
                        labels=dict(x="Predicted",y="Actual"),
                        color_continuous_scale=SOLAR_SCALE,
                        title="🗺️ Confusion Matrix")
        _t(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)
        acadcard("The confusion matrix shows classification performance per class. Diagonal = correct predictions. Off-diagonal = misclassifications.")

    # Customer profile by adoption group
    sh("Customer Profile by Adoption Group")
    sdiv(); badges(biz=True)
    if "AdoptionLikelihood" in dff.columns:
        prof_col = st.selectbox("Profile by:", [c for c in cat_cols if c!="AdoptionLikelihood"], key="prof_col")
        pct = dff.groupby("AdoptionLikelihood")[prof_col].value_counts(normalize=True).mul(100).round(1).reset_index(name="Percent")
        pc1,pc2 = st.columns(2)
        with pc1:
            fig = px.bar(pct, x="AdoptionLikelihood", y="Percent", color=prof_col,
                         barmode="stack", title=f"Adoption Group × {prof_col} (%)",
                         color_discrete_sequence=CAT_COLORS)
            _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)
        with pc2:
            fig = px.bar(pct, x=prof_col, y="Percent", color="AdoptionLikelihood",
                         barmode="group", title=f"{prof_col} within each Adoption Group",
                         color_discrete_sequence=CAT_COLORS)
            _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)
        acard("Use this to build a buyer persona. The group with the highest 'Yes' or '3m' share is your primary target market for initial rollout.")

    # EMI × Adoption
    if "EMI_Willingness" in dff.columns and "AdoptionLikelihood" in dff.columns:
        sh("EMI Willingness by Adoption Intent")
        sdiv(); badges(biz=True)
        emi_pct = dff.groupby("AdoptionLikelihood")["EMI_Willingness"].value_counts(normalize=True).mul(100).round(1).reset_index(name="Percent")
        fig = px.bar(emi_pct, x="AdoptionLikelihood", y="Percent", color="EMI_Willingness",
                     barmode="stack", title="EMI Willingness within each Adoption Group",
                     color_discrete_sequence=CAT_COLORS)
        _t(fig, height=380); st.plotly_chart(fig, use_container_width=True)
        acard("Design loan products around the EMI range preferred by your highest-adoption groups. If they prefer <1k EMI, offer smaller panel packages or longer tenors.")

    # Decision Tree
    sh("Decision Tree (Academic — Session 3)")
    sdiv(); badges(acad=True)
    acadcard("Decision Trees are covered in Session 3 of the course. They provide a transparent, rule-based classification that is easy to interpret and explain to non-technical stakeholders.")
    with st.spinner("Training Decision Tree…"):
        dt, feat_names, dt_acc = dtree_model(data_clf)

    dc1,dc2 = st.columns(2)
    dc1.metric("🌳 Decision Tree Accuracy", f"{dt_acc:.2%}")
    dc2.metric("🌲 Random Forest Accuracy", f"{mets['Accuracy']:.2%}", f"+{(mets['Accuracy']-dt_acc):.2%} vs Decision Tree")

    tree_text = export_text(dt, feature_names=feat_names, max_depth=3)
    st.markdown(f"""<div class="academic-card"><pre style="color:#ccd6f6;font-size:0.73rem;overflow-x:auto;white-space:pre-wrap;">{tree_text}</pre></div>""", unsafe_allow_html=True)

    # RF vs DT importance
    dt_imp = pd.Series(dt.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
    comp = pd.DataFrame({"Feature":dt_imp.index,
                          "Decision Tree":dt_imp.values,
                          "Random Forest":imp.reindex(dt_imp.index,fill_value=0).values})
    fig = px.bar(comp, x="Feature", y=["Decision Tree","Random Forest"], barmode="group",
                 title="Feature Importance: Decision Tree vs Random Forest",
                 color_discrete_map={"Decision Tree":"#64ffda","Random Forest":"#ffd200"})
    _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)
    acadcard("Random Forest averages 100 trees, giving more stable importance scores. A single Decision Tree can overfit — compare both to find genuinely robust predictors.")

# ══════════════════════════════════════════════
# TAB 3 — REGRESSION (Academic — Session 7)
# ══════════════════════════════════════════════
with t3:
    sh("Regression Analysis — EMI Willingness Predictor")
    sdiv()
    acadcard("Regression analysis is covered in Session 7 of the course (Business Forecasting using Multilinear Regression). Note: EMI_Willingness is an ordinal categorical variable encoded as integers (0,1,2,3). The regression model treats these as numeric — results are academically valid for demonstrating regression concepts, though an ordinal classifier would be more technically precise.")
    badges(acad=True)

    if "EMI_Willingness" not in dff_enc.columns:
        st.info("EMI_Willingness column not found.")
    else:
        data_reg = df_enc if use_full else dff_enc
        with st.spinner("Training Regression Model…"):
            mets_r, y_act, y_pr, reg_imp = reg_model(data_reg)

        rc1,rc2,rc3 = st.columns(3)
        rc1.metric("📐 R² Score", f"{mets_r['R²']:.4f}", help="1.0 = perfect prediction")
        rc2.metric("📏 MAE", f"{mets_r['MAE']:.4f}", help="Average prediction error")
        rc3.metric("📊 Observations", f"{len(y_act):,}")

        c1,c2 = st.columns(2)
        with c1:
            badges(acad=True)
            pdf = pd.DataFrame({"Actual":y_act,"Predicted":y_pr})
            fig = px.scatter(pdf, x="Actual", y="Predicted",
                             title="Actual vs Predicted EMI Willingness",
                             opacity=0.65, color_discrete_sequence=["#64ffda"])
            mn,mx = pdf.min().min(), pdf.max().max()
            fig.add_shape(type="line",x0=mn,y0=mn,x1=mx,y1=mx,
                          line=dict(color="#ffd200",dash="dash",width=2))
            _t(fig); st.plotly_chart(fig, use_container_width=True)
            acadcard("The dashed line = perfect prediction. Points close to it = accurate model. Scatter above/below = over/under-prediction.")

        with c2:
            badges(acad=True)
            resid = y_act - y_pr
            fig = px.histogram(x=resid, nbins=25, title="Residual Distribution",
                               color_discrete_sequence=["#a78bfa"], marginal="box")
            fig.add_vline(x=0, line_dash="dash", line_color="#ffd200",
                          annotation_text="Zero error", annotation_font_color="#ffd200")
            _t(fig); st.plotly_chart(fig, use_container_width=True)
            acadcard("Residuals = actual − predicted. A bell-shaped distribution centred at 0 means errors are random and unbiased — a healthy regression model.")

        c1,c2 = st.columns(2)
        with c1:
            badges(acad=True)
            fig = px.scatter(x=y_pr, y=resid, opacity=0.55,
                             labels={"x":"Predicted","y":"Residual"},
                             title="Residuals vs Fitted Values",
                             color_discrete_sequence=["#f472b6"])
            fig.add_hline(y=0, line_dash="dash", line_color="#ffd200")
            _t(fig); st.plotly_chart(fig, use_container_width=True)
            acadcard("If residuals are randomly scattered around 0 with no pattern, the model assumptions hold. A funnel shape would indicate heteroscedasticity.")

        with c2:
            badges(acad=True)
            (osm,osr),(slope,intercept,r) = scipy_stats.probplot(resid)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                     marker=dict(color="#64ffda",size=5,opacity=0.6), name="Residuals"))
            fig.add_trace(go.Scatter(x=[min(osm),max(osm)],
                                     y=[slope*min(osm)+intercept,slope*max(osm)+intercept],
                                     mode="lines", line=dict(color="#ffd200",dash="dash"), name="Normal line"))
            _t(fig, title="📐 Q-Q Plot — Normality of Residuals")
            st.plotly_chart(fig, use_container_width=True)
            acadcard("Q-Q Plot (Quantile-Quantile): if residuals are normally distributed, points will lie on the dashed line. Deviations at the tails indicate non-normality — a standard regression diagnostic check.")

        # Feature importance for regression
        sh("What Drives EMI Willingness?")
        sdiv(); badges(biz=True, acad=True)
        ri = reg_imp.reset_index(); ri.columns=["Factor","Importance"]
        fig = px.bar(ri.head(10), x="Importance", y="Factor", orientation="h",
                     title="Top predictors of EMI Willingness",
                     color="Importance", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=380, yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
        st.plotly_chart(fig, use_container_width=True)
        acard("Design your EMI product tiers around the top predictors. If Income is #1, offer income-linked EMI slabs. If ElectricityBill is #2, offer higher EMI options to high-bill households.")

# ══════════════════════════════════════════════
# TAB 4 — CLUSTERING (Academic — Session 5)
# ══════════════════════════════════════════════
with t4:
    sh("K-Means Clustering — Customer Segmentation")
    sdiv()
    acadcard("Clustering is covered in Session 5 of the course. K-Means partitions respondents into K groups based on feature similarity — no predefined labels. This is unsupervised learning.")
    badges(biz=True, acad=True)

    data_cl = df_enc if use_full else dff_enc
    with st.spinner("Segmenting customers…"):
        labels, inertia = cluster_model(data_cl, k)

    dfc = (df if use_full else dff).copy()
    dfc["Segment"] = [f"Segment {i+1}" for i in labels]

    c1,c2 = st.columns(2)
    with c1:
        sc = dfc["Segment"].value_counts().reset_index(); sc.columns=["Segment","Count"]
        fig = px.pie(sc, names="Segment", values="Count",
                     title="How many customers per segment?",
                     color_discrete_sequence=CAT_COLORS, hole=0.38)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(**THEME, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,9)), y=inertia, mode="lines+markers",
                                 line=dict(color="#ffd200",width=2),
                                 marker=dict(color="#f7971e",size=8)))
        fig.add_vline(x=k, line_dash="dash", line_color="#64ffda",
                      annotation_text=f"k={k}", annotation_font_color="#64ffda")
        _t(fig, height=380, title="Elbow Curve — Optimal K Selection")
        fig.update_layout(xaxis_title="Number of Segments", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
        acadcard("The elbow point is where adding more clusters gives diminishing returns in reducing inertia. This guides the choice of K.")

    # Segment profiles
    sh("What Makes Each Segment Different?")
    sdiv(); badges(biz=True)
    for seg in sorted(dfc["Segment"].unique()):
        seg_df = dfc[dfc["Segment"]==seg]
        with st.expander(f"📋 {seg} — {len(seg_df)} customers ({len(seg_df)/len(dfc):.1%})", expanded=False):
            profile_cols=[c for c in cat_cols if c in seg_df.columns]
            cols=st.columns(min(3,len(profile_cols)))
            for i,col in enumerate(profile_cols[:6]):
                tv=seg_df[col].value_counts().idxmax()
                tp=seg_df[col].value_counts(normalize=True).max()
                cols[i%3].metric(col, tv, f"{tp:.0%}")

    # Adoption mix per segment
    sh("Adoption Intent by Segment"); sdiv(); badges(biz=True)
    if "AdoptionLikelihood" in dfc.columns:
        sp = dfc.groupby("Segment")["AdoptionLikelihood"].value_counts(normalize=True).mul(100).round(1).unstack(fill_value=0)
        fig = px.bar(sp.reset_index(), x="Segment", y=sp.columns.tolist(), barmode="stack",
                     title="Which segments are most ready to adopt?",
                     color_discrete_sequence=CAT_COLORS)
        _t(fig, height=380); st.plotly_chart(fig, use_container_width=True)
        acard("Prioritise segments with the highest share of immediate adoption intent. These are your easiest first sales.")

    # Segment × selected variable
    sh("Segment Composition Deep-Dive"); sdiv(); badges(biz=True)
    seg_var = st.selectbox("Break segments by:", [c for c in cat_cols if c in dfc.columns], key="seg_var")
    seg_cross = dfc.groupby(["Segment",seg_var]).size().reset_index(name="Count")
    sv1,sv2 = st.columns(2)
    with sv1:
        fig = px.bar(seg_cross, x="Segment", y="Count", color=seg_var, barmode="stack",
                     title=f"Segment × {seg_var}", color_discrete_sequence=CAT_COLORS)
        _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)
    with sv2:
        fig = px.bar(seg_cross, x="Segment", y="Count", color=seg_var, barmode="group",
                     title=f"Segment × {seg_var} (grouped)", color_discrete_sequence=CAT_COLORS)
        _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 — ASSOCIATION RULES (Academic — Session 4)
# ══════════════════════════════════════════════
with t5:
    sh("Market Basket Analysis — Association Rules")
    sdiv()
    acadcard("Association Rules / Market Basket Analysis is covered in Session 4. It finds which combinations of customer attributes frequently co-occur, measured by Support, Confidence, and Lift.")
    badges(biz=True, acad=True)

    data_ar = df if use_full else dff
    with st.spinner("Mining patterns…"):
        freq_df, rules_df = assoc_rules_fn(data_ar)

    if rules_df.empty:
        st.info("No significant rules found. Try removing filters to use more data.")
    else:
        rc1,rc2,rc3 = st.columns(3)
        rc1.metric("🔗 Rules Found", len(rules_df))
        rc2.metric("⚡ Strongest Lift", f"{rules_df['lift'].max():.2f}x")
        rc3.metric("📦 Frequent Itemsets", len(freq_df))

        # Plain-English rules table
        sh("Top Rules — Plain English"); sdiv()
        top = rules_df.head(8).copy()
        top["If customer has →"] = top["antecedents"].astype(str)
        top["They likely also have →"] = top["consequents"].astype(str)
        top["Lift"] = top["lift"].round(2)
        top["Confidence"] = top["confidence"].round(2)
        top["Support"] = top["support"].round(3)
        st.dataframe(top[["If customer has →","They likely also have →","Lift","Confidence","Support"]],
                     use_container_width=True, hide_index=True)
        icard("Lift > 1 means the combination occurs more often than by chance. Lift of 2 = twice as likely. Use these rules to build targeted customer profiles.")

        c1,c2 = st.columns(2)
        with c1:
            badges(acad=True)
            fig = px.scatter(rules_df, x="support", y="confidence", size="lift",
                             color="lift", hover_data=["antecedents","consequents"],
                             title="Support vs Confidence (bubble size = Lift)",
                             color_continuous_scale=SOLAR_SCALE)
            _t(fig); st.plotly_chart(fig, use_container_width=True)
            acadcard("Support = how common the rule is. Confidence = how reliable it is. Lift = how much stronger than random. Good rules have high values on all three.")

        with c2:
            badges(acad=True)
            top10 = rules_df.head(10).copy()
            top10["rule"] = top10["antecedents"].astype(str)+" → "+top10["consequents"].astype(str)
            fig = px.bar(top10, x="lift", y="rule", orientation="h",
                         title="Top 10 Rules by Lift",
                         color="lift", color_continuous_scale=SOLAR_SCALE)
            _t(fig, height=380, yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
            st.plotly_chart(fig, use_container_width=True)

        # Lift heatmap
        sh("Lift Heatmap"); sdiv(); badges(acad=True)
        h = rules_df.copy()
        h["ant"] = h["antecedents"].astype(str); h["con"] = h["consequents"].astype(str)
        pivot = h.pivot_table(index="ant", columns="con", values="lift", aggfunc="max").fillna(0)
        if not pivot.empty:
            fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale=SOLAR_SCALE,
                            title="Lift Heatmap — Antecedent × Consequent", aspect="auto")
            _t(fig, height=400); st.plotly_chart(fig, use_container_width=True)
        acard("Use the highest-lift rules to design hyper-targeted campaigns — e.g. if Urban + 1L-2L Income → High Adoption, run a focused digital campaign for that exact profile.")

# ══════════════════════════════════════════════
# TAB 6 — STATISTICAL TOOLS
# ══════════════════════════════════════════════
with t6:
    sh("📐 Descriptive Statistics")
    sdiv(); badges(acad=True)
    desc = dff_enc[num_enc_cols].describe().T
    desc["skewness"] = dff_enc[num_enc_cols].skew().round(3)
    desc["kurtosis"] = dff_enc[num_enc_cols].kurtosis().round(3)
    st.dataframe(desc.round(3), use_container_width=True)
    acadcard("Skewness measures asymmetry (0 = symmetric). Kurtosis measures tail heaviness (3 = normal distribution). These diagnostics inform whether parametric tests are appropriate.")

    # Skewness + Kurtosis charts
    sk1,sk2 = st.columns(2)
    with sk1:
        badges(acad=True)
        skdf = desc["skewness"].reset_index(); skdf.columns=["Feature","Skewness"]
        fig = px.bar(skdf, x="Feature", y="Skewness", title="Skewness per Feature",
                     color="Skewness", color_continuous_scale=["#38bdf8","#1a1f35","#ffd200"])
        fig.add_hline(y=0, line_dash="dash", line_color="#64ffda",
                      annotation_text="Symmetric", annotation_font_color="#64ffda")
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)
    with sk2:
        badges(acad=True)
        kudf = desc["kurtosis"].reset_index(); kudf.columns=["Feature","Kurtosis"]
        fig = px.bar(kudf, x="Feature", y="Kurtosis", title="Kurtosis per Feature",
                     color="Kurtosis", color_continuous_scale=["#a78bfa","#1a1f35","#f472b6"])
        fig.add_hline(y=3, line_dash="dash", line_color="#64ffda",
                      annotation_text="Normal distribution = 3", annotation_font_color="#64ffda")
        _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

    # Outlier detection
    sh("🚨 Outlier Detection — IQR Method")
    sdiv(); badges(acad=True)
    acadcard("The IQR (Interquartile Range) method flags values below Q1−1.5×IQR or above Q3+1.5×IQR as outliers. This is a standard non-parametric outlier detection technique.")
    out_col = st.selectbox("Select feature", num_enc_cols, key="out_col")
    col_data = dff_enc[out_col].dropna()
    Q1,Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
    IQR = Q3-Q1; lower,upper = Q1-1.5*IQR, Q3+1.5*IQR
    outliers = col_data[(col_data<lower)|(col_data>upper)]
    normal   = col_data[(col_data>=lower)&(col_data<=upper)]

    o1,o2,o3,o4 = st.columns(4)
    o1.metric("Total Points", len(col_data))
    o2.metric("🚨 Outliers", len(outliers), f"{len(outliers)/len(col_data):.1%}")
    o3.metric("Lower Fence", f"{lower:.2f}")
    o4.metric("Upper Fence", f"{upper:.2f}")

    oc1,oc2 = st.columns(2)
    with oc1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(normal))), y=normal.values, mode="markers",
                                 marker=dict(color="#64ffda",size=4,opacity=0.5), name="Normal"))
        fig.add_trace(go.Scatter(x=list(range(len(outliers))), y=outliers.values, mode="markers",
                                 marker=dict(color="#f472b6",size=9,symbol="x"), name="Outlier"))
        fig.add_hline(y=upper, line_dash="dash", line_color="#ffd200", annotation_text="Upper fence")
        fig.add_hline(y=lower, line_dash="dash", line_color="#ffd200", annotation_text="Lower fence")
        _t(fig, height=340, title=f"Outliers — {out_col}")
        st.plotly_chart(fig, use_container_width=True)
    with oc2:
        fig = px.box(dff_enc, y=out_col, title=f"Box Plot — {out_col}",
                     color_discrete_sequence=["#f7971e"], points="outliers")
        _t(fig, height=340); st.plotly_chart(fig, use_container_width=True)

    # Chi-Square test
    sh("📊 Chi-Square Test of Independence")
    sdiv(); badges(acad=True)
    acadcard("The Chi-Square test checks whether two categorical variables are statistically independent. A low p-value (< 0.05) rejects the null hypothesis of independence, meaning the relationship is real and not due to chance.")
    cc1,cc2 = st.columns(2)
    with cc1: cat_a = st.selectbox("Variable A", cat_cols, key="chi_a")
    with cc2:
        cat_b_opts=[c for c in cat_cols if c!=cat_a]
        cat_b = st.selectbox("Variable B", cat_b_opts, key="chi_b")
    ct = pd.crosstab(dff[cat_a], dff[cat_b])
    chi2,p,dof,_ = scipy_stats.chi2_contingency(ct)
    sig = "✅ Significant — variables ARE related (p < 0.05)" if p<0.05 else "❌ Not significant — variables appear independent (p ≥ 0.05)"
    ch1,ch2,ch3,ch4 = st.columns(4)
    ch1.metric("χ² Statistic", f"{chi2:.3f}")
    ch2.metric("p-value", f"{p:.4f}")
    ch3.metric("Degrees of Freedom", dof)
    ch4.metric("Result", "Significant ✅" if p<0.05 else "Not Significant ❌")
    st.markdown(f'<div class="{"action-card" if p<0.05 else "insight-card"}">{sig}</div>', unsafe_allow_html=True)
    fig = px.imshow(ct, text_auto=True, title=f"Crosstab Heatmap: {cat_a} × {cat_b}",
                    color_continuous_scale=SOLAR_SCALE, aspect="auto")
    _t(fig, height=380); st.plotly_chart(fig, use_container_width=True)

    # Distribution comparison by group (violin)
    sh("📈 Distribution Comparison by Group")
    sdiv(); badges(acad=True)
    acadcard("Violin plots combine a box plot with a kernel density estimate — showing not just the median and quartiles, but the full shape of the distribution for each group.")
    dc1,dc2 = st.columns(2)
    with dc1: grp_col = st.selectbox("Group by", cat_cols, key="dist_grp")
    with dc2: val_col = st.selectbox("Value", [c for c in cat_cols if c!=grp_col], key="dist_val")
    if grp_col in dff.columns and val_col in dff.columns:
        vc_v = dff[val_col].value_counts()
        dff_v = dff.copy()
        dff_v[val_col+"_num"] = dff_v[val_col].map({v:i for i,v in enumerate(vc_v.index)})
        fig = px.violin(dff_v, x=grp_col, y=val_col+"_num", box=True, points="outliers",
                        color=grp_col, title=f"{val_col} distribution by {grp_col}",
                        color_discrete_sequence=CAT_COLORS,
                        labels={val_col+"_num": val_col})
        _t(fig, height=420); st.plotly_chart(fig, use_container_width=True)
