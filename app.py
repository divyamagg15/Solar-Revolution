import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

st.set_page_config(page_title="Solar Analytics Dashboard", page_icon="☀️",
                   layout="wide", initial_sidebar_state="expanded")

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
        height: 2px; background: linear-gradient(90deg, #f7971e33, #ffd20066, #f7971e33);
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
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117, #161b27); border-right: 1px solid #2e3555; }
    [data-testid="stSidebar"] label { color: #8892b0 !important; font-size: 0.78rem; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1f35; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #f7971e, #ffd200) !important; color: #0f1117 !important; font-weight: 700; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

THEME       = dict(plot_bgcolor="#161b27", paper_bgcolor="#1a1f35", font_color="#ccd6f6",
                   title_font_color="#e2e8f0",
                   colorway=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8","#34d399","#fb923c"])
_AXIS       = dict(gridcolor="#2e3555", linecolor="#2e3555")
SOLAR_SCALE = ["#0d1117","#1a1f35","#f7971e","#ffd200"]
COOL_SCALE  = ["#0d1117","#1a1f35","#38bdf8","#64ffda"]
CAT_COLORS  = ["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6","#38bdf8","#34d399","#fb923c"]

def _t(fig, height=None, xaxis_kw=None, yaxis_kw=None, **extra):
    kw = {**THEME, "xaxis":{**_AXIS,**(xaxis_kw or {})},
          "yaxis":{**_AXIS,**(yaxis_kw or {})}, **extra}
    if height: kw["height"] = height
    fig.update_layout(**kw); return fig

def sh(t):   st.markdown(f'<p class="section-header">{t}</p>', unsafe_allow_html=True)
def sdiv():  st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
def icard(t):st.markdown(f'<div class="insight-card">💡 <b>Insight:</b> {t}</div>', unsafe_allow_html=True)
def acard(t):st.markdown(f'<div class="action-card">🎯 <b>Action:</b> {t}</div>', unsafe_allow_html=True)

# ── DATA ─────────────────────────────────────
@st.cache_data
def load_data(): return pd.read_csv("dataset.csv")

@st.cache_data
def encode(df):
    out = df.copy(); enc = {}
    for c in out.columns:
        if out[c].dtype == object:
            le = LabelEncoder()
            out[c] = le.fit_transform(out[c].astype(str)); enc[c] = le
    return out, enc

@st.cache_data
def run_all_classifiers(df_enc):
    X = df_enc.drop("AdoptionLikelihood", axis=1); y = df_enc["AdoptionLikelihood"]
    Xt,Xe,yt,ye = train_test_split(X, y, test_size=0.2, random_state=42)
    Xs = StandardScaler().fit_transform(X)
    Xts,Xes = train_test_split(Xs, test_size=0.2, random_state=42)

    models = {
        "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), Xt, Xe, yt, ye),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=5, random_state=42),      Xt, Xe, yt, ye),
        "Gradient Boosting":   (GradientBoostingClassifier(random_state=42),               Xt, Xe, yt, ye),
        "Logistic Regression": (LogisticRegression(max_iter=500, random_state=42),         Xts, Xes, yt, ye),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=5),                       Xts, Xes, yt, ye),
        "Naive Bayes":         (GaussianNB(),                                               Xts, Xes, yt, ye),
    }

    results = []
    best_acc, best_name, best_pred, best_cm, best_imp = 0, None, None, None, None

    for name,(m,Xtr,Xte,ytr,yte) in models.items():
        m.fit(Xtr, ytr); pred = m.predict(Xte)
        acc = accuracy_score(yte, pred)
        row = {"Model": name,
               "Accuracy":  round(acc, 4),
               "Precision": round(precision_score(yte,pred,average="weighted",zero_division=0),4),
               "Recall":    round(recall_score(yte,pred,average="weighted",zero_division=0),4),
               "F1 Score":  round(f1_score(yte,pred,average="weighted",zero_division=0),4)}
        results.append(row)
        if acc > best_acc:
            best_acc, best_name, best_pred, best_cm = acc, name, pred, confusion_matrix(yte,pred)
            if hasattr(m,"feature_importances_"):
                best_imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
            else:
                best_imp = pd.Series(np.zeros(X.shape[1]), index=X.columns)

    return pd.DataFrame(results), best_name, best_pred, best_cm, best_imp, ye

@st.cache_data
def run_regression(df_enc):
    drop = [c for c in ["AdoptionLikelihood","EMI_Willingness"] if c in df_enc.columns]
    X = df_enc.drop(columns=drop); y = df_enc["EMI_Willingness"]
    Xt,Xe,yt,ye = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler(); Xts = scaler.fit_transform(Xt); Xes = scaler.transform(Xe)

    models = {
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Ridge Regression":        Ridge(alpha=1.0),
        "Lasso Regression":        Lasso(alpha=0.1, max_iter=2000),
    }
    results = []
    rf_pred, rf_imp = None, None
    for name, m in models.items():
        if "Forest" in name:
            m.fit(Xt,yt); pred = m.predict(Xe)
            rf_pred = pred
            rf_imp  = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
        else:
            m.fit(Xts,yt); pred = m.predict(Xes)
        results.append({"Model":name,
                        "R² Score": round(r2_score(ye,pred),4),
                        "MAE":      round(mean_absolute_error(ye,pred),4)})
    return pd.DataFrame(results), ye.values, rf_pred, rf_imp

@st.cache_data
def run_clustering(df_enc, k):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
    inertia = [KMeans(n_clusters=i,n_init=10,random_state=42).fit(Xs).inertia_ for i in range(2,9)]
    return labels, inertia

@st.cache_data
def run_assoc(df):
    cols = [c for c in ["Income","Location","AdoptionLikelihood"] if c in df.columns]
    db = pd.get_dummies(df[cols].astype(str)).astype(bool)
    try:
        freq = apriori(db, min_support=0.1, use_colnames=True)
        if freq.empty: return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq, metric="lift", min_threshold=1.0, num_itemsets=len(freq))
        return freq, rules.sort_values("lift", ascending=False).head(20)
    except Exception: return pd.DataFrame(), pd.DataFrame()

# ── LOAD ─────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ `dataset.csv` not found. Ensure it is in the same directory as `app.py`."); st.stop()

df_enc, encoders = encode(df)
cat_cols     = [c for c in df.select_dtypes("object").columns if df[c].nunique() <= 20]
num_enc_cols = df_enc.select_dtypes(np.number).columns.tolist()

# ── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Solar Analytics"); sdiv()
    st.markdown("### 🎛️ Filters")
    filters = {}
    for col in cat_cols[:4]:
        sel = st.selectbox(col, ["All"] + sorted(df[col].dropna().unique().tolist()))
        filters[col] = sel
    sdiv()
    st.markdown("### ⚙️ Settings")
    k = st.slider("Customer Segments (K)", 2, 8, 4)
    sdiv(); st.caption("Solar Venture Analytics · 2026")

dff = df.copy()
for col,val in filters.items():
    if val != "All": dff = dff[dff[col]==val]
dff_enc, _ = encode(dff)
use_full = len(dff_enc) < 50

# ── HEADER ────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#f7971e22,#ffd20011);border:1px solid #f7971e44;
            border-radius:16px;padding:24px 32px;margin-bottom:20px;'>
  <h1 style='color:#ffd200;margin:0;font-size:2.2rem;font-weight:800;'>☀️ Solar Market Analytics Dashboard</h1>
  <p style='color:#8892b0;margin:6px 0 0 0;font-size:1rem;'>Consumer adoption intelligence · ML-powered insights · Pan-India market</p>
</div>""", unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
k1.metric("📊 Respondents", f"{len(dff):,}")
if "AdoptionLikelihood" in dff.columns:
    top=dff["AdoptionLikelihood"].value_counts()
    k2.metric("🏆 Top Adoption Intent", str(top.idxmax()), f"{top.max()/len(dff):.1%}")
if "EMI_Willingness" in dff.columns:
    k3.metric("💰 Most Common EMI", str(dff["EMI_Willingness"].value_counts().idxmax()))
if "Awareness" in dff.columns:
    hi=(dff["Awareness"]=="High").mean() if dff["Awareness"].dtype==object else dff["Awareness"].mean()
    k4.metric("📢 High Awareness", f"{hi:.1%}")
st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────
t1,t2,t3,t4,t5,t6 = st.tabs([
    "📊 EDA & Charts", "🤖 Classification", "📈 Regression",
    "🔵 Clustering", "🔗 Association Rules", "🔬 Statistical Tools"
])

# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with t1:
    sh("Data Overview"); sdiv()
    c1,c2 = st.columns([2,1])
    with c1: st.dataframe(dff.head(10), use_container_width=True, height=260)
    with c2:
        st.markdown("**Dataset Stats**")
        st.dataframe(pd.DataFrame({
            "Metric":["Rows","Columns","Categorical","Numeric","Missing"],
            "Value":[len(dff),dff.shape[1],dff.select_dtypes("object").shape[1],
                     dff.select_dtypes(np.number).shape[1],int(dff.isnull().sum().sum())]
        }), use_container_width=True, hide_index=True)

    # Interactive Pie
    sh("🥧 Interactive Pie Chart — Click a Slice to Drill Down"); sdiv()
    pie_opts = [c for c in cat_cols if c in dff.columns]
    pie_col  = st.selectbox("Select dimension", pie_opts,
                            index=pie_opts.index("AdoptionLikelihood") if "AdoptionLikelihood" in pie_opts else 0,
                            key="pie_sel")
    vc_pie   = dff[pie_col].value_counts().reset_index(); vc_pie.columns=[pie_col,"Count"]
    fig_pie  = px.pie(vc_pie, names=pie_col, values="Count",
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
            dc1,dc2 = st.columns(2)
            with dc1:
                fig = px.bar(vc_d, x=drill_on, y="Count", title=f'"{selected_slice}" → {drill_on}',
                             color="Count", color_continuous_scale=SOLAR_SCALE)
                _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
            with dc2:
                fig = px.pie(vc_d, names=drill_on, values="Count", color_discrete_sequence=CAT_COLORS,
                             hole=0.35, title=f'"{selected_slice}" → {drill_on} share')
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(**THEME, height=300); st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("☝️ Click any slice above to drill into that group.")

    # Chart gallery
    sh("📊 Chart Gallery"); sdiv()
    gc1,gc2 = st.columns(2)
    with gc1:
        gcat = st.selectbox("Categorical column", cat_cols,
                            index=cat_cols.index("AdoptionLikelihood") if "AdoptionLikelihood" in cat_cols else 0,
                            key="gal_cat")
    with gc2:
        gnum = st.selectbox("Numeric column", num_enc_cols,
                            index=num_enc_cols.index("EMI_Willingness") if "EMI_Willingness" in num_enc_cols else 0,
                            key="gal_num")
    vc = dff[gcat].value_counts().reset_index(); vc.columns=[gcat,"Count"]

    r1c1,r1c2,r1c3 = st.columns(3)
    with r1c1:
        fig = px.bar(vc, x=gcat, y="Count", title="📊 Bar Chart",
                     color="Count", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
    with r1c2:
        fig = px.bar(vc, x="Count", y=gcat, orientation="h", title="📊 Horizontal Bar",
                     color="Count", color_continuous_scale=SOLAR_SCALE)
        _t(fig, height=300, yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
        st.plotly_chart(fig, use_container_width=True)
    with r1c3:
        fig = px.treemap(vc, path=[gcat], values="Count", title="🗺️ Treemap",
                         color="Count", color_continuous_scale=SOLAR_SCALE)
        fig.update_layout(**THEME, height=300); st.plotly_chart(fig, use_container_width=True)

    r2c1,r2c2,r2c3 = st.columns(3)
    with r2c1:
        sb2=[c for c in cat_cols if c!=gcat]
        if sb2:
            fig = px.sunburst(dff, path=[gcat,sb2[0]], title="🌞 Sunburst",
                              color_discrete_sequence=CAT_COLORS)
        else:
            fig = px.sunburst(vc, path=[gcat], values="Count", title="🌞 Sunburst",
                              color_discrete_sequence=CAT_COLORS)
        fig.update_layout(**THEME, height=300); st.plotly_chart(fig, use_container_width=True)
    with r2c2:
        fig = px.histogram(dff_enc, x=gnum, nbins=20, title="📈 Histogram",
                           color_discrete_sequence=["#ffd200"], marginal="rug")
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
    with r2c3:
        fig = px.box(dff_enc, y=gnum, title="📦 Box Plot",
                     color_discrete_sequence=["#f7971e"], points="outliers")
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)

    r3c1,r3c2,r3c3 = st.columns(3)
    with r3c1:
        fig = px.violin(dff_enc, y=gnum, box=True, points="outliers",
                        title="🎻 Violin Plot", color_discrete_sequence=["#a78bfa"])
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
    with r3c2:
        num2=[c for c in num_enc_cols if c!=gnum]
        fig = px.scatter(dff_enc, x=gnum, y=num2[0] if num2 else gnum, opacity=0.6,
                         title="⚬ Scatter Plot", color_discrete_sequence=["#64ffda"])
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)
    with r3c3:
        if len(num_enc_cols)>=2:
            bdf=dff_enc[num_enc_cols[:3]].sample(min(200,len(dff_enc)),random_state=42)
            sz=num_enc_cols[2] if len(num_enc_cols)>=3 else num_enc_cols[0]
            fig=px.scatter(bdf,x=num_enc_cols[0],y=num_enc_cols[1],size=sz,
                           color=num_enc_cols[0],title="🫧 Bubble Chart",
                           color_continuous_scale=SOLAR_SCALE,opacity=0.7)
            _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)

    r4c1,r4c2,r4c3 = st.columns(3)
    with r4c1:
        fig=px.funnel(vc, x="Count", y=gcat, title="🔽 Funnel Chart",
                      color_discrete_sequence=CAT_COLORS)
        fig.update_layout(**THEME, height=300); st.plotly_chart(fig, use_container_width=True)
        icard("Funnel charts show values in descending order — useful for comparing magnitude across categories and spotting the largest and smallest groups at a glance.")
    with r4c2:
        if len(num_enc_cols)>=4:
            rv=dff_enc[num_enc_cols[:6]].mean().values.tolist()
            rc=num_enc_cols[:6]; rv+=[rv[0]]; rc+=[rc[0]]
            fig=go.Figure(go.Scatterpolar(r=rv,theta=rc,fill='toself',
                                          line_color="#ffd200",fillcolor="rgba(247,151,30,0.18)"))
            fig.update_layout(**THEME,height=300,title="🕸️ Radar Chart",
                              polar=dict(bgcolor="#161b27",
                                         radialaxis=dict(gridcolor="#2e3555",linecolor="#2e3555",color="#8892b0"),
                                         angularaxis=dict(gridcolor="#2e3555",linecolor="#2e3555",color="#ccd6f6")))
            st.plotly_chart(fig, use_container_width=True)
    with r4c3:
        vc_l=dff[gcat].value_counts().reset_index(); vc_l.columns=[gcat,"Count"]
        fig=px.area(vc_l.sort_values(gcat),x=gcat,y="Count",title="📉 Area Chart",
                    color_discrete_sequence=["#64ffda"])
        fig.update_traces(fillcolor="rgba(100,255,218,0.12)")
        _t(fig, height=300); st.plotly_chart(fig, use_container_width=True)

    # Stacked + Grouped
    sh("Cross-Category Analysis"); sdiv()
    xc1,xc2=st.columns(2)
    with xc1: col_x=st.selectbox("X-axis",cat_cols,key="cx")
    with xc2:
        col_y_opts=[c for c in cat_cols if c!=col_x]
        col_y=st.selectbox("Colour by",col_y_opts,key="cy")
    cross=dff.groupby([col_x,col_y]).size().reset_index(name="Count")
    sc1,sc2=st.columns(2)
    with sc1:
        fig=px.bar(cross,x=col_x,y="Count",color=col_y,barmode="stack",
                   title=f"{col_x} × {col_y} — Stacked",color_discrete_sequence=CAT_COLORS)
        _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)
    with sc2:
        fig=px.bar(cross,x=col_x,y="Count",color=col_y,barmode="group",
                   title=f"{col_x} × {col_y} — Grouped",color_discrete_sequence=CAT_COLORS)
        _t(fig, height=360); st.plotly_chart(fig, use_container_width=True)
    icard(f"The stacked view shows total volume per {col_x} group. The grouped view makes it easier to compare {col_y} sub-groups side by side within each {col_x} category.")

    # Sunburst + Parallel Coords
    hc1,hc2=st.columns(2)
    with hc1:
        sh("Hierarchical Sunburst"); sdiv()
        sl1=st.selectbox("Level 1",cat_cols,key="sb1",
                         index=cat_cols.index("Location") if "Location" in cat_cols else 0)
        sl2_opts=[c for c in cat_cols if c!=sl1]
        sl2=st.selectbox("Level 2",sl2_opts,key="sb2",
                         index=sl2_opts.index("AdoptionLikelihood") if "AdoptionLikelihood" in sl2_opts else 0)
        fig=px.sunburst(dff,path=[sl1,sl2],title=f"{sl1} → {sl2}",color_discrete_sequence=CAT_COLORS)
        fig.update_layout(**THEME, height=420); st.plotly_chart(fig, use_container_width=True)
        icard(f"Wider wedges in the outer ring = larger sub-groups. Use this to spot which {sl1} groups carry the most adoption opportunity.")
    with hc2:
        sh("Parallel Coordinates"); sdiv()
        pc_cols=num_enc_cols[:6]
        fig=px.parallel_coordinates(dff_enc[pc_cols],color=dff_enc[num_enc_cols[0]],
                                    color_continuous_scale=SOLAR_SCALE,
                                    title="All numeric features simultaneously")
        fig.update_layout(**THEME, height=420); st.plotly_chart(fig, use_container_width=True)
        icard("Lines that cluster together represent respondents with similar profiles. Crossing lines between two axes indicate an inverse relationship between those features.")

    # Correlation heatmap
    sh("Correlation Heatmap"); sdiv()
    corr=dff_enc.corr()
    fig=px.imshow(corr,text_auto=".2f",title="Feature Correlation Matrix",
                  color_continuous_scale=SOLAR_SCALE,aspect="auto")
    _t(fig, height=460); st.plotly_chart(fig, use_container_width=True)
    icard("Values close to ±1 indicate strong relationships. Check the AdoptionLikelihood row/column to see which features are most predictive of adoption intent.")

# ══════════════════════════════════════════════
# TAB 2 — CLASSIFICATION (ALL ALGORITHMS)
# ══════════════════════════════════════════════
with t2:
    sh("Classification — All Algorithms Compared"); sdiv()
    data_clf = df_enc if use_full else dff_enc
    if use_full: st.info("ℹ️ Too few rows after filtering — using full dataset.")

    with st.spinner("Training all 6 classifiers…"):
        clf_df, best_name, best_pred, best_cm, best_imp, y_te = run_all_classifiers(data_clf)

    # Comparison table
    sh("📊 Algorithm Performance Comparison"); sdiv()
    styled = clf_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    icard(f"**{best_name}** achieves the highest accuracy. Accuracy alone can be misleading — check Precision, Recall and F1 together for a complete picture, especially when class sizes are unequal.")

    # Visual comparison
    metrics_long = clf_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig=px.bar(metrics_long, x="Model", y="Score", color="Metric", barmode="group",
               title="All Classifiers — All Metrics Side by Side",
               color_discrete_sequence=CAT_COLORS)
    _t(fig, height=400); st.plotly_chart(fig, use_container_width=True)

    # Radar comparison of all models
    fig=go.Figure()
    metrics_list=["Accuracy","Precision","Recall","F1 Score"]
    for _,row in clf_df.iterrows():
        vals=[row[m] for m in metrics_list]+[row[metrics_list[0]]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=metrics_list+[metrics_list[0]],
                                     fill='toself', name=row["Model"], opacity=0.7))
    fig.update_layout(**THEME, height=420, title="Algorithm Comparison — Radar View",
                      polar=dict(bgcolor="#161b27",
                                  radialaxis=dict(range=[0,1],gridcolor="#2e3555",linecolor="#2e3555",color="#8892b0"),
                                  angularaxis=dict(gridcolor="#2e3555",linecolor="#2e3555",color="#ccd6f6")))
    st.plotly_chart(fig, use_container_width=True)
    icard("The radar view lets you spot trade-offs at a glance — a model with a large, balanced shape across all four metrics is the most reliable overall performer.")

    # Best model detail
    sh(f"🏆 Best Model: {best_name}"); sdiv()
    c1,c2=st.columns(2)
    with c1:
        if best_imp is not None and best_imp.sum()>0:
            fi=best_imp.reset_index(); fi.columns=["Factor","Importance"]
            fig=px.bar(fi.head(10),x="Importance",y="Factor",orientation="h",
                       title=f"🔑 Key Drivers of Adoption ({best_name})",
                       color="Importance",color_continuous_scale=SOLAR_SCALE)
            _t(fig,height=400,yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
            st.plotly_chart(fig, use_container_width=True)
            acard("Focus your sales and awareness strategy on the top-ranked factors. These are what most strongly determine whether a customer will adopt solar.")
    with c2:
        classes=[str(c) for c in sorted(set(y_te))]
        fig=px.imshow(best_cm,text_auto=True,x=classes,y=classes,
                      labels=dict(x="Predicted",y="Actual"),
                      color_continuous_scale=SOLAR_SCALE,
                      title=f"Confusion Matrix — {best_name}")
        _t(fig, height=400); st.plotly_chart(fig, use_container_width=True)
        icard("Each row represents the actual class; each column is the predicted class. Large numbers on the diagonal mean the model predicts that adoption group reliably.")

    # Customer profile by adoption group
    sh("Customer Profile by Adoption Group"); sdiv()
    if "AdoptionLikelihood" in dff.columns:
        prof_col=st.selectbox("Profile by:",[c for c in cat_cols if c!="AdoptionLikelihood"],key="prof_col")
        pct=dff.groupby("AdoptionLikelihood")[prof_col].value_counts(normalize=True).mul(100).round(1).reset_index(name="Percent")
        pc1,pc2=st.columns(2)
        with pc1:
            fig=px.bar(pct,x="AdoptionLikelihood",y="Percent",color=prof_col,barmode="stack",
                       title=f"Adoption Group × {prof_col} (%)",color_discrete_sequence=CAT_COLORS)
            _t(fig,height=360); st.plotly_chart(fig, use_container_width=True)
        with pc2:
            fig=px.bar(pct,x=prof_col,y="Percent",color="AdoptionLikelihood",barmode="group",
                       title=f"{prof_col} breakdown by Adoption Group",color_discrete_sequence=CAT_COLORS)
            _t(fig,height=360); st.plotly_chart(fig, use_container_width=True)
        acard("Build a buyer persona from the dominant profile of the highest-adoption group — that's your primary target market for the first sales push.")

    if "EMI_Willingness" in dff.columns and "AdoptionLikelihood" in dff.columns:
        sh("EMI Willingness by Adoption Intent"); sdiv()
        emi_pct=dff.groupby("AdoptionLikelihood")["EMI_Willingness"].value_counts(normalize=True).mul(100).round(1).reset_index(name="Percent")
        fig=px.bar(emi_pct,x="AdoptionLikelihood",y="Percent",color="EMI_Willingness",barmode="stack",
                   title="EMI Willingness within each Adoption Group",color_discrete_sequence=CAT_COLORS)
        _t(fig, height=380); st.plotly_chart(fig, use_container_width=True)
        acard("Design EMI products around the range preferred by the highest-adoption groups. Match EMI slabs to their income and willingness to pay.")

# ══════════════════════════════════════════════
# TAB 3 — REGRESSION
# ══════════════════════════════════════════════
with t3:
    sh("Regression Analysis — EMI Willingness Predictor"); sdiv()
    if "EMI_Willingness" not in dff_enc.columns:
        st.info("EMI_Willingness column not found.")
    else:
        data_reg = df_enc if use_full else dff_enc
        with st.spinner("Training regression models…"):
            reg_df, y_act, rf_pred, reg_imp = run_regression(data_reg)

        # Model comparison
        sh("📊 Regression Models Compared"); sdiv()
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
        icard("R² closer to 1.0 = better fit. Lower MAE = smaller average prediction error. Compare all three models to select the most accurate for EMI prediction.")

        c1,c2=st.columns(2)
        with c1:
            fig=px.bar(reg_df,x="Model",y="R² Score",title="R² Score Comparison",
                       color="R² Score",color_continuous_scale=SOLAR_SCALE)
            _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig=px.bar(reg_df,x="Model",y="MAE",title="MAE Comparison (lower is better)",
                       color="MAE",color_continuous_scale=COOL_SCALE)
            _t(fig, height=320); st.plotly_chart(fig, use_container_width=True)

        # RF residual diagnostics
        sh("Regression Diagnostics (Random Forest)"); sdiv()
        resid = y_act - rf_pred
        dc1,dc2=st.columns(2)
        with dc1:
            pdf=pd.DataFrame({"Actual":y_act,"Predicted":rf_pred})
            fig=px.scatter(pdf,x="Actual",y="Predicted",title="Actual vs Predicted",
                           opacity=0.65,color_discrete_sequence=["#64ffda"])
            mn,mx=pdf.min().min(),pdf.max().max()
            fig.add_shape(type="line",x0=mn,y0=mn,x1=mx,y1=mx,
                          line=dict(color="#ffd200",dash="dash",width=2))
            _t(fig); st.plotly_chart(fig, use_container_width=True)
            icard("Points close to the dashed perfect-prediction line indicate high accuracy. Scatter above/below shows systematic over- or under-prediction.")
        with dc2:
            fig=px.histogram(x=resid,nbins=25,title="Residual Distribution",
                             color_discrete_sequence=["#a78bfa"],marginal="box")
            fig.add_vline(x=0,line_dash="dash",line_color="#ffd200")
            _t(fig); st.plotly_chart(fig, use_container_width=True)
            icard("A bell-shaped residual distribution centred at 0 means prediction errors are random and unbiased — a sign of a healthy model.")

        dc1,dc2=st.columns(2)
        with dc1:
            fig=px.scatter(x=rf_pred,y=resid,opacity=0.55,
                           labels={"x":"Predicted","y":"Residual"},
                           title="Residuals vs Fitted Values",color_discrete_sequence=["#f472b6"])
            fig.add_hline(y=0,line_dash="dash",line_color="#ffd200")
            _t(fig); st.plotly_chart(fig, use_container_width=True)
            icard("A random scatter around the zero line confirms the model has no systematic bias across different prediction levels.")
        with dc2:
            (osm,osr),(slope,intercept,r)=scipy_stats.probplot(resid)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=osm,y=osr,mode="markers",
                                     marker=dict(color="#64ffda",size=5,opacity=0.6),name="Residuals"))
            fig.add_trace(go.Scatter(x=[min(osm),max(osm)],
                                     y=[slope*min(osm)+intercept,slope*max(osm)+intercept],
                                     mode="lines",line=dict(color="#ffd200",dash="dash"),name="Normal line"))
            _t(fig,title="📐 Q-Q Plot — Normality of Residuals")
            st.plotly_chart(fig, use_container_width=True)
            icard("If residuals are normally distributed, points follow the dashed line closely. Deviations at the tails indicate non-normality in the error distribution.")

        if reg_imp is not None:
            sh("What Drives EMI Willingness?"); sdiv()
            ri=reg_imp.reset_index(); ri.columns=["Factor","Importance"]
            fig=px.bar(ri.head(10),x="Importance",y="Factor",orientation="h",
                       title="Top predictors of EMI Willingness",
                       color="Importance",color_continuous_scale=SOLAR_SCALE)
            _t(fig,height=380,yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
            st.plotly_chart(fig, use_container_width=True)
            acard("Design EMI product tiers around the top predictors. If Income ranks highest, create income-linked EMI slabs. If ElectricityBill ranks high, target heavy-bill households with aggressive solar financing.")

# ══════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ══════════════════════════════════════════════
with t4:
    sh("K-Means Clustering — Customer Segmentation"); sdiv()
    data_cl = df_enc if use_full else dff_enc
    with st.spinner("Segmenting customers…"):
        labels, inertia = run_clustering(data_cl, k)

    dfc=(df if use_full else dff).copy()
    dfc["Segment"]=[f"Segment {i+1}" for i in labels]

    c1,c2=st.columns(2)
    with c1:
        sc=dfc["Segment"].value_counts().reset_index(); sc.columns=["Segment","Count"]
        fig=px.pie(sc,names="Segment",values="Count",title="How many customers per segment?",
                   color_discrete_sequence=CAT_COLORS,hole=0.38)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(**THEME,height=380); st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=list(range(2,9)),y=inertia,mode="lines+markers",
                                 line=dict(color="#ffd200",width=2),
                                 marker=dict(color="#f7971e",size=8)))
        fig.add_vline(x=k,line_dash="dash",line_color="#64ffda",
                      annotation_text=f"k={k}",annotation_font_color="#64ffda")
        _t(fig,height=380,title="Elbow Curve — Optimal K Selection")
        fig.update_layout(xaxis_title="Number of Segments",yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
        icard("The elbow point is where adding more clusters gives diminishing returns. This guides the optimal choice of K.")

    sh("Segment Profiles — What Makes Each Group Different?"); sdiv()
    for seg in sorted(dfc["Segment"].unique()):
        seg_df=dfc[dfc["Segment"]==seg]
        with st.expander(f"📋 {seg} — {len(seg_df)} customers ({len(seg_df)/len(dfc):.1%})", expanded=False):
            pcols=[c for c in cat_cols if c in seg_df.columns]
            cols=st.columns(min(3,len(pcols)))
            for i,col in enumerate(pcols[:6]):
                tv=seg_df[col].value_counts().idxmax()
                tp=seg_df[col].value_counts(normalize=True).max()
                cols[i%3].metric(col,tv,f"{tp:.0%}")

    sh("Adoption Intent by Segment"); sdiv()
    if "AdoptionLikelihood" in dfc.columns:
        sp=dfc.groupby("Segment")["AdoptionLikelihood"].value_counts(normalize=True).mul(100).round(1).unstack(fill_value=0)
        fig=px.bar(sp.reset_index(),x="Segment",y=sp.columns.tolist(),barmode="stack",
                   title="Which segments are most ready to adopt?",color_discrete_sequence=CAT_COLORS)
        _t(fig,height=380); st.plotly_chart(fig, use_container_width=True)
        acard("Prioritise segments with the highest share of immediate adoption intent. These customers are closest to a purchase decision and require the least convincing.")

    sh("Segment Deep-Dive"); sdiv()
    seg_var=st.selectbox("Break segments by:",[c for c in cat_cols if c in dfc.columns],key="seg_var")
    seg_cross=dfc.groupby(["Segment",seg_var]).size().reset_index(name="Count")
    sv1,sv2=st.columns(2)
    with sv1:
        fig=px.bar(seg_cross,x="Segment",y="Count",color=seg_var,barmode="stack",
                   title=f"Segment × {seg_var}",color_discrete_sequence=CAT_COLORS)
        _t(fig,height=360); st.plotly_chart(fig, use_container_width=True)
    with sv2:
        fig=px.bar(seg_cross,x="Segment",y="Count",color=seg_var,barmode="group",
                   title=f"Segment × {seg_var} (grouped)",color_discrete_sequence=CAT_COLORS)
        _t(fig,height=360); st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 — ASSOCIATION RULES
# ══════════════════════════════════════════════
with t5:
    sh("Market Basket Analysis — Association Rules"); sdiv()
    data_ar = df if use_full else dff
    with st.spinner("Mining patterns…"):
        freq_df, rules_df = run_assoc(data_ar)

    if rules_df.empty:
        st.info("No significant rules found. Try removing filters to use more data.")
    else:
        rc1,rc2,rc3=st.columns(3)
        rc1.metric("🔗 Rules Found",len(rules_df))
        rc2.metric("⚡ Strongest Lift",f"{rules_df['lift'].max():.2f}x")
        rc3.metric("📦 Frequent Itemsets",len(freq_df))

        sh("Top Rules — Plain English"); sdiv()
        top=rules_df.head(8).copy()
        top["If customer has →"]=top["antecedents"].astype(str)
        top["They likely also have →"]=top["consequents"].astype(str)
        top["Lift"]=top["lift"].round(2)
        top["Confidence"]=top["confidence"].round(2)
        top["Support"]=top["support"].round(3)
        st.dataframe(top[["If customer has →","They likely also have →","Lift","Confidence","Support"]],
                     use_container_width=True,hide_index=True)
        icard("Lift > 1 means the combination occurs more often than by chance. Lift of 2 means twice as likely. Use high-lift rules to build targeted customer profiles for campaigns.")

        c1,c2=st.columns(2)
        with c1:
            fig=px.scatter(rules_df,x="support",y="confidence",size="lift",
                           color="lift",hover_data=["antecedents","consequents"],
                           title="Support vs Confidence (bubble size = Lift)",
                           color_continuous_scale=SOLAR_SCALE)
            _t(fig); st.plotly_chart(fig, use_container_width=True)
            icard("Best rules sit in the top-right of this chart — high support (common) and high confidence (reliable). Bubble size shows additional lift strength.")
        with c2:
            top10=rules_df.head(10).copy()
            top10["rule"]=top10["antecedents"].astype(str)+" → "+top10["consequents"].astype(str)
            fig=px.bar(top10,x="lift",y="rule",orientation="h",title="Top 10 Rules by Lift",
                       color="lift",color_continuous_scale=SOLAR_SCALE)
            _t(fig,height=380,yaxis_kw=dict(categoryorder="total ascending",gridcolor="#2e3555",linecolor="#2e3555"))
            st.plotly_chart(fig, use_container_width=True)

        sh("Lift Heatmap"); sdiv()
        h=rules_df.copy(); h["ant"]=h["antecedents"].astype(str); h["con"]=h["consequents"].astype(str)
        pivot=h.pivot_table(index="ant",columns="con",values="lift",aggfunc="max").fillna(0)
        if not pivot.empty:
            fig=px.imshow(pivot,text_auto=".2f",color_continuous_scale=SOLAR_SCALE,
                          title="Lift Heatmap — Antecedent × Consequent",aspect="auto")
            _t(fig,height=400); st.plotly_chart(fig, use_container_width=True)
            acard("Use the brightest cells in this heatmap to design hyper-targeted campaigns. Each bright cell is a customer archetype with a proven tendency toward a specific adoption pattern.")

        st.markdown("#### 📋 All Rules")
        disp=rules_df[["antecedents","consequents","support","confidence","lift"]].copy()
        disp["antecedents"]=disp["antecedents"].astype(str); disp["consequents"]=disp["consequents"].astype(str)
        for c in ["support","confidence","lift"]: disp[c]=disp[c].round(4)
        st.dataframe(disp,use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════
# TAB 6 — STATISTICAL TOOLS
# ══════════════════════════════════════════════
with t6:
    sh("Descriptive Statistics"); sdiv()
    desc=dff_enc[num_enc_cols].describe().T
    desc["skewness"]=dff_enc[num_enc_cols].skew().round(3)
    desc["kurtosis"]=dff_enc[num_enc_cols].kurtosis().round(3)
    st.dataframe(desc.round(3), use_container_width=True)
    icard("Skewness measures asymmetry — values near 0 indicate a roughly symmetric distribution. Kurtosis measures tail heaviness — a value of 3 corresponds to a normal distribution.")

    sk1,sk2=st.columns(2)
    with sk1:
        skdf=desc["skewness"].reset_index(); skdf.columns=["Feature","Skewness"]
        fig=px.bar(skdf,x="Feature",y="Skewness",title="Skewness per Feature",
                   color="Skewness",color_continuous_scale=["#38bdf8","#1a1f35","#ffd200"])
        fig.add_hline(y=0,line_dash="dash",line_color="#64ffda",annotation_text="Symmetric",annotation_font_color="#64ffda")
        _t(fig,height=320); st.plotly_chart(fig, use_container_width=True)
    with sk2:
        kudf=desc["kurtosis"].reset_index(); kudf.columns=["Feature","Kurtosis"]
        fig=px.bar(kudf,x="Feature",y="Kurtosis",title="Kurtosis per Feature",
                   color="Kurtosis",color_continuous_scale=["#a78bfa","#1a1f35","#f472b6"])
        fig.add_hline(y=3,line_dash="dash",line_color="#64ffda",annotation_text="Normal = 3",annotation_font_color="#64ffda")
        _t(fig,height=320); st.plotly_chart(fig, use_container_width=True)

    sh("Outlier Detection — IQR Method"); sdiv()
    out_col=st.selectbox("Select feature",num_enc_cols,key="out_col")
    col_data=dff_enc[out_col].dropna()
    Q1,Q3=col_data.quantile(0.25),col_data.quantile(0.75)
    IQR=Q3-Q1; lower,upper=Q1-1.5*IQR,Q3+1.5*IQR
    outliers=col_data[(col_data<lower)|(col_data>upper)]
    normal=col_data[(col_data>=lower)&(col_data<=upper)]
    o1,o2,o3,o4=st.columns(4)
    o1.metric("Total Points",len(col_data)); o2.metric("🚨 Outliers",len(outliers),f"{len(outliers)/len(col_data):.1%}")
    o3.metric("Lower Fence",f"{lower:.2f}"); o4.metric("Upper Fence",f"{upper:.2f}")
    oc1,oc2=st.columns(2)
    with oc1:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(normal))),y=normal.values,mode="markers",
                                 marker=dict(color="#64ffda",size=4,opacity=0.5),name="Normal"))
        fig.add_trace(go.Scatter(x=list(range(len(outliers))),y=outliers.values,mode="markers",
                                 marker=dict(color="#f472b6",size=9,symbol="x"),name="Outlier"))
        fig.add_hline(y=upper,line_dash="dash",line_color="#ffd200",annotation_text="Upper fence")
        fig.add_hline(y=lower,line_dash="dash",line_color="#ffd200",annotation_text="Lower fence")
        _t(fig,height=340,title=f"Outlier Scatter — {out_col}"); st.plotly_chart(fig, use_container_width=True)
        icard(f"{len(outliers)} outliers ({len(outliers)/len(col_data):.1%}) detected in {out_col}. Pink X marks show values beyond the IQR fences — investigate these before using in models.")
    with oc2:
        fig=px.box(dff_enc,y=out_col,title=f"Box Plot — {out_col}",
                   color_discrete_sequence=["#f7971e"],points="outliers")
        _t(fig,height=340); st.plotly_chart(fig, use_container_width=True)

    sh("Chi-Square Test of Independence"); sdiv()
    cc1,cc2=st.columns(2)
    with cc1: cat_a=st.selectbox("Variable A",cat_cols,key="chi_a")
    with cc2:
        cat_b_opts=[c for c in cat_cols if c!=cat_a]
        cat_b=st.selectbox("Variable B",cat_b_opts,key="chi_b")
    ct=pd.crosstab(dff[cat_a],dff[cat_b])
    chi2,p,dof,_=scipy_stats.chi2_contingency(ct)
    ch1,ch2,ch3,ch4=st.columns(4)
    ch1.metric("χ² Statistic",f"{chi2:.3f}"); ch2.metric("p-value",f"{p:.4f}")
    ch3.metric("Degrees of Freedom",dof)
    ch4.metric("Result","Significant ✅" if p<0.05 else "Not Significant ❌")
    sig_msg=f"p = {p:.4f} — {'The relationship between {cat_a} and {cat_b} is statistically significant. These variables are NOT independent.' if p<0.05 else f'No significant relationship found between {cat_a} and {cat_b} at the 5% level.'}"
    icard(f"p = {p:.4f} — {'The relationship between these variables is statistically significant — they are NOT independent.' if p<0.05 else 'No significant relationship found at the 5% significance level.'}")
    fig=px.imshow(ct,text_auto=True,title=f"Crosstab: {cat_a} × {cat_b}",
                  color_continuous_scale=SOLAR_SCALE,aspect="auto")
    _t(fig,height=380); st.plotly_chart(fig, use_container_width=True)

    sh("Distribution Comparison by Group"); sdiv()
    dc1,dc2=st.columns(2)
    with dc1: grp_col=st.selectbox("Group by",cat_cols,key="dist_grp")
    with dc2: val_col=st.selectbox("Value",[c for c in cat_cols if c!=grp_col],key="dist_val")
    if grp_col in dff.columns and val_col in dff.columns:
        vc_v=dff[val_col].value_counts()
        dff_v=dff.copy()
        dff_v[val_col+"_num"]=dff_v[val_col].map({v:i for i,v in enumerate(vc_v.index)})
        fig=px.violin(dff_v,x=grp_col,y=val_col+"_num",box=True,points="outliers",
                      color=grp_col,title=f"{val_col} distribution by {grp_col}",
                      color_discrete_sequence=CAT_COLORS,
                      labels={val_col+"_num":val_col})
        _t(fig,height=420); st.plotly_chart(fig, use_container_width=True)
        icard(f"Wider violin bodies indicate more respondents concentrated at that value. Compare violin widths across {grp_col} groups to see where {val_col} preferences diverge most.")
