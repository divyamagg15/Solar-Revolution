import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score, confusion_matrix
)
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
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252b40);
        border: 1px solid #2e3555;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="metric-container"] label {
        color: #8892b0 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #64ffda !important;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0 0.5rem 0;
    }

    /* Divider */
    .solar-divider {
        height: 2px;
        background: linear-gradient(90deg, #f7971e33, #ffd20066, #f7971e33);
        border: none;
        margin: 1rem 0 1.5rem 0;
        border-radius: 2px;
    }

    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #1a1f35, #1e2440);
        border-left: 4px solid #ffd200;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #ccd6f6;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #161b27);
        border-right: 1px solid #2e3555;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #8892b0 !important;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f35;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8892b0;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f7971e, #ffd200) !important;
        color: #0f1117 !important;
        font-weight: 700;
    }

    /* Dataframes */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    plot_bgcolor="#161b27",
    paper_bgcolor="#1a1f35",
    font_color="#ccd6f6",
    title_font_color="#e2e8f0",
    colorway=["#ffd200", "#64ffda", "#f7971e", "#a78bfa", "#f472b6", "#38bdf8"],
    xaxis=dict(gridcolor="#2e3555", linecolor="#2e3555"),
    yaxis=dict(gridcolor="#2e3555", linecolor="#2e3555"),
)

# ─────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

@st.cache_data
def prepare_encoded(df):
    df_enc = df.copy()
    encoders = {}
    for col in df_enc.columns:
        if df_enc[col].dtype == object:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
    return df_enc, encoders

@st.cache_data
def run_classification(df_enc):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    y = df_enc["AdoptionLikelihood"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    metrics = {
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, pred, average="weighted", zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, pred, average="weighted", zero_division=0), 4),
    }
    importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    cm = confusion_matrix(y_test, pred)
    return metrics, importance, cm, y_test, pred

@st.cache_data
def run_regression(df_enc):
    X = df_enc.drop(columns=["AdoptionLikelihood", "EMI_Willingness"], errors="ignore")
    y = df_enc["EMI_Willingness"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    metrics = {
        "R² Score": round(r2_score(y_test, pred), 4),
        "MAE": round(mean_absolute_error(y_test, pred), 4),
    }
    return metrics, y_test.values, pred

@st.cache_data
def run_clustering(df_enc, n_clusters=4):
    X = df_enc.drop("AdoptionLikelihood", axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertia = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    return labels, inertia

@st.cache_data
def run_association(df):
    cols = [c for c in ["Income", "Location", "AdoptionLikelihood"] if c in df.columns]
    df_bin = pd.get_dummies(df[cols].astype(str))
    df_bin = df_bin.astype(bool)
    try:
        freq = apriori(df_bin, min_support=0.1, use_colnames=True)
        if len(freq) == 0:
            return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq, metric="lift", min_threshold=1.0, num_itemsets=len(freq))
        rules = rules.sort_values("lift", ascending=False).head(20)
        return freq, rules
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ `dataset.csv` not found. Please ensure it's in the same directory as `app.py`.")
    st.stop()

df_enc, encoders = prepare_encoded(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☀️ Solar Analytics")
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    st.markdown("### 🎛️ Filters")

    cat_cols = [c for c in df.select_dtypes(include="object").columns if df[c].nunique() <= 20]

    filters = {}
    for col in cat_cols[:4]:  # max 4 filters
        options = ["All"] + sorted(df[col].dropna().unique().tolist())
        sel = st.selectbox(col, options)
        filters[col] = sel

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Settings")
    n_clusters = st.slider("K-Means Clusters", 2, 8, 4)

    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
    st.caption("Built for Solar Venture Analytics · 2026")

# Apply filters
df_filtered = df.copy()
for col, val in filters.items():
    if val != "All":
        df_filtered = df_filtered[df_filtered[col] == val]

df_enc_filtered, _ = prepare_encoded(df_filtered)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #f7971e22, #ffd20011); border: 1px solid #f7971e44;
            border-radius: 16px; padding: 24px 32px; margin-bottom: 24px;'>
  <h1 style='color:#ffd200; margin:0; font-size:2.2rem; font-weight:800;'>
    ☀️ Solar Market Analytics Dashboard
  </h1>
  <p style='color:#8892b0; margin:8px 0 0 0; font-size:1rem;'>
    Consumer adoption intelligence · ML-powered insights · Pan-India market
  </p>
</div>
""", unsafe_allow_html=True)

# KPI Row
total, col_ado, col_emi = st.columns(3)
total.metric("📊 Total Respondents", f"{len(df_filtered):,}", delta=f"{len(df_filtered) - len(df)} vs unfiltered" if len(df_filtered) != len(df) else "Full dataset")

if "AdoptionLikelihood" in df_filtered.columns:
    top_cat = df_filtered["AdoptionLikelihood"].value_counts().idxmax()
    top_pct = df_filtered["AdoptionLikelihood"].value_counts(normalize=True).max()
    col_ado.metric("🏆 Top Adoption Category", str(top_cat), f"{top_pct:.1%} of responses")

if "EMI_Willingness" in df_filtered.columns:
    if df_filtered["EMI_Willingness"].dtype == object:
        top_emi = df_filtered["EMI_Willingness"].value_counts().idxmax()
        col_emi.metric("💰 Top EMI Preference", str(top_emi))
    else:
        avg_emi = df_filtered["EMI_Willingness"].mean()
        col_emi.metric("💰 Avg EMI Willingness", f"{avg_emi:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploration",
    "🤖 Classification",
    "📈 Regression",
    "🔵 Clustering",
    "🔗 Association Rules"
])

# ══════════════════════════════════════════════
# TAB 1 — EXPLORATION
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Data Overview</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df_filtered.head(10), use_container_width=True, height=280)
    with c2:
        st.markdown("**Dataset Stats**")
        stat_df = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Categorical", "Numerical", "Missing Values"],
            "Value": [
                len(df_filtered),
                df_filtered.shape[1],
                df_filtered.select_dtypes("object").shape[1],
                df_filtered.select_dtypes(np.number).shape[1],
                int(df_filtered.isnull().sum().sum())
            ]
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Distribution Analysis</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    all_cols = df_filtered.columns.tolist()
    sel_col = st.selectbox("Select column to visualise", all_cols, index=all_cols.index("AdoptionLikelihood") if "AdoptionLikelihood" in all_cols else 0)

    c1, c2 = st.columns(2)
    with c1:
        if df_filtered[sel_col].dtype == object or df_filtered[sel_col].nunique() <= 15:
            vc = df_filtered[sel_col].value_counts().reset_index()
            vc.columns = [sel_col, "Count"]
            fig = px.bar(vc, x=sel_col, y="Count", title=f"{sel_col} — Distribution",
                         color="Count", color_continuous_scale=["#1a1f35", "#ffd200"])
        else:
            fig = px.histogram(df_filtered, x=sel_col, title=f"{sel_col} — Histogram",
                               color_discrete_sequence=["#ffd200"], nbins=20)
        fig.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if df_filtered[sel_col].dtype == object or df_filtered[sel_col].nunique() <= 15:
            vc = df_filtered[sel_col].value_counts().reset_index()
            vc.columns = [sel_col, "Count"]
            fig2 = px.pie(vc, names=sel_col, values="Count", title=f"{sel_col} — Share",
                          color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig2.update_traces(textinfo="percent+label")
        else:
            fig2 = px.box(df_filtered, y=sel_col, title=f"{sel_col} — Box Plot",
                          color_discrete_sequence=["#f7971e"])
        fig2.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap for numeric cols
    num_cols = df_enc_filtered.select_dtypes(np.number).columns.tolist()
    if len(num_cols) >= 3:
        st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
        st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)
        corr = df_enc_filtered[num_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale=["#0d1117", "#1a1f35", "#f7971e", "#ffd200"],
            title="Feature Correlation Matrix",
            aspect="auto"
        )
        fig_corr.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Adoption Likelihood Classifier</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    if len(df_enc_filtered) < 50:
        st.warning("⚠️ Too few rows after filtering for reliable classification. Showing results on full dataset.")
        df_enc_clf = df_enc
    else:
        df_enc_clf = df_enc_filtered

    with st.spinner("Training Random Forest Classifier..."):
        clf_metrics, feat_imp, cm, y_test_c, pred_c = run_classification(df_enc_clf)

    m1, m2, m3, m4 = st.columns(4)
    icons = ["🎯", "⚡", "🔁", "🏅"]
    for col_m, (name, val), icon in zip([m1, m2, m3, m4], clf_metrics.items(), icons):
        col_m.metric(f"{icon} {name}", f"{val:.2%}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🌟 Feature Importance")
        fi_df = feat_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fig_fi = px.bar(fi_df.head(12), x="Importance", y="Feature", orientation="h",
                        title="Top 12 Predictive Features",
                        color="Importance", color_continuous_scale=["#1a1f35", "#f7971e", "#ffd200"])
        fig_fi.update_layout(**PLOTLY_THEME, height=400, yaxis=dict(categoryorder="total ascending", gridcolor="#2e3555", linecolor="#2e3555"))
        st.plotly_chart(fig_fi, use_container_width=True)

    with c2:
        st.markdown("#### 🗺️ Confusion Matrix")
        classes = sorted(set(y_test_c))
        fig_cm = px.imshow(cm, text_auto=True, x=[str(c) for c in classes], y=[str(c) for c in classes],
                           labels=dict(x="Predicted", y="Actual"),
                           color_continuous_scale=["#0d1117", "#1a1f35", "#ffd200"],
                           title="Predicted vs Actual")
        fig_cm.update_layout(**PLOTLY_THEME, height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="insight-card">💡 <strong>Insight:</strong> Features at the top of the importance chart are the strongest drivers of solar adoption likelihood. Focus sales and awareness efforts on respondents with high-scoring profiles in these dimensions.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — REGRESSION
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">EMI Willingness Predictor</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    if "EMI_Willingness" not in df_enc_filtered.columns:
        st.info("EMI_Willingness column not found in dataset.")
    else:
        if len(df_enc_filtered) < 50:
            df_enc_reg = df_enc
        else:
            df_enc_reg = df_enc_filtered

        with st.spinner("Training Regression Model..."):
            reg_metrics, y_act, y_pred_r = run_regression(df_enc_reg)

        r2, mae = st.columns(2)
        r2.metric("📐 R² Score", f"{reg_metrics['R² Score']:.4f}", help="Closer to 1.0 is better")
        mae.metric("📏 Mean Absolute Error", f"{reg_metrics['MAE']:.4f}", help="Lower is better")

        c1, c2 = st.columns(2)
        with c1:
            pred_df = pd.DataFrame({"Actual": y_act, "Predicted": y_pred_r})
            fig_pred = px.scatter(pred_df, x="Actual", y="Predicted",
                                  title="Actual vs Predicted EMI Willingness",
                                  opacity=0.7, color_discrete_sequence=["#64ffda"])
            min_v, max_v = pred_df.min().min(), pred_df.max().max()
            fig_pred.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                               line=dict(color="#ffd200", dash="dash", width=2))
            fig_pred.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig_pred, use_container_width=True)

        with c2:
            residuals = y_act - y_pred_r
            fig_res = px.histogram(x=residuals, title="Residual Distribution",
                                   nbins=25, color_discrete_sequence=["#a78bfa"])
            fig_res.add_vline(x=0, line_dash="dash", line_color="#ffd200")
            fig_res.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig_res, use_container_width=True)

        st.markdown('<div class="insight-card">💡 <strong>Insight:</strong> The diagonal line on the scatter plot represents a perfect prediction. Points clustered tightly around it indicate a strong model. A symmetric residual distribution centred at 0 confirms low systematic bias.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Consumer Segment Discovery</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    with st.spinner("Running K-Means Clustering..."):
        cluster_labels, inertia_vals = run_clustering(df_enc_filtered if len(df_enc_filtered) >= 50 else df_enc, n_clusters)

    df_clustered = df_filtered.copy() if len(df_filtered) >= 50 else df.copy()
    df_clustered["Cluster"] = [f"Segment {i+1}" for i in cluster_labels]

    c1, c2 = st.columns(2)
    with c1:
        seg_counts = df_clustered["Cluster"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_seg = px.pie(seg_counts, names="Segment", values="Count",
                         title="Respondents per Segment",
                         color_discrete_sequence=["#ffd200", "#64ffda", "#f7971e", "#a78bfa", "#f472b6", "#38bdf8", "#34d399", "#fb923c"])
        fig_seg.update_traces(textinfo="percent+label")
        fig_seg.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_seg, use_container_width=True)

    with c2:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(2, 9)), y=inertia_vals,
            mode="lines+markers",
            line=dict(color="#ffd200", width=2),
            marker=dict(color="#f7971e", size=8),
            name="Inertia"
        ))
        fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#64ffda",
                            annotation_text=f"k={n_clusters}", annotation_font_color="#64ffda")
        fig_elbow.update_layout(**PLOTLY_THEME, title="Elbow Curve — Optimal K Selection",
                                xaxis_title="Number of Clusters", yaxis_title="Inertia")
        st.plotly_chart(fig_elbow, use_container_width=True)

    # Segment profile
    if "AdoptionLikelihood" in df_clustered.columns:
        st.markdown("#### 📋 Segment Profiles")
        seg_profile = df_clustered.groupby("Cluster")["AdoptionLikelihood"].value_counts(normalize=True).unstack(fill_value=0)
        fig_seg2 = px.bar(seg_profile.reset_index(), x="Cluster",
                          y=seg_profile.columns.tolist(), barmode="stack",
                          title="Adoption Likelihood by Segment",
                          color_discrete_sequence=["#ffd200","#64ffda","#f7971e","#a78bfa","#f472b6"])
        fig_seg2.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_seg2, use_container_width=True)

    st.markdown('<div class="insight-card">💡 <strong>Insight:</strong> Each segment represents a distinct consumer archetype. The elbow curve helps you choose the ideal K — look for the "kink" where adding more clusters yields diminishing returns in inertia reduction.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 5 — ASSOCIATION RULES
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Market Basket — Association Rules</p>', unsafe_allow_html=True)
    st.markdown('<hr class="solar-divider">', unsafe_allow_html=True)

    with st.spinner("Mining association rules..."):
        freq_items, rules_df = run_association(df_filtered if len(df_filtered) >= 50 else df)

    if rules_df.empty:
        st.info("No significant association rules found with current filter. Try removing filters or adjusting thresholds.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("🔗 Rules Found", len(rules_df))
        c2.metric("⚡ Avg Lift", f"{rules_df['lift'].mean():.2f}")
        c3.metric("📦 Frequent Itemsets", len(freq_items))

        c1, c2 = st.columns(2)
        with c1:
            fig_rules = px.scatter(rules_df, x="support", y="confidence", size="lift",
                                   color="lift", hover_data=["antecedents", "consequents"],
                                   title="Support vs Confidence (size = Lift)",
                                   color_continuous_scale=["#1a1f35", "#f7971e", "#ffd200"])
            fig_rules.update_layout(**PLOTLY_THEME)
            st.plotly_chart(fig_rules, use_container_width=True)

        with c2:
            top_rules = rules_df.head(10).copy()
            top_rules["rule"] = top_rules["antecedents"].astype(str) + " → " + top_rules["consequents"].astype(str)
            fig_lift = px.bar(top_rules, x="lift", y="rule", orientation="h",
                              title="Top 10 Rules by Lift",
                              color="lift", color_continuous_scale=["#1a1f35", "#ffd200"])
            fig_lift.update_layout(**PLOTLY_THEME, height=400, yaxis=dict(categoryorder="total ascending", gridcolor="#2e3555", linecolor="#2e3555"))
            st.plotly_chart(fig_lift, use_container_width=True)

        st.markdown("#### 📋 Full Rules Table")
        display_rules = rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
        display_rules["antecedents"] = display_rules["antecedents"].astype(str)
        display_rules["consequents"] = display_rules["consequents"].astype(str)
        for col in ["support", "confidence", "lift"]:
            display_rules[col] = display_rules[col].round(4)
        st.dataframe(display_rules, use_container_width=True, hide_index=True)

        st.markdown('<div class="insight-card">💡 <strong>Insight:</strong> Rules with high lift (> 1.5) reveal non-obvious purchasing affinities. Use these to build targeted cross-sell bundles — e.g., customers in a certain income bracket + location show strong affinity for specific adoption categories.</div>', unsafe_allow_html=True)
