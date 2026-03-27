
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Solar Analytics Dashboard")

# Load data
df = pd.read_csv("dataset.csv")

st.write("App started successfully")

st.header("Overview")
st.dataframe(df.head())

fig = px.histogram(df, x="AdoptionLikelihood")
st.plotly_chart(fig)

# Encoding properly
df_encoded = df.copy()
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le

# Classification
X = df_encoded.drop("AdoptionLikelihood", axis=1)
y = df_encoded["AdoptionLikelihood"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

st.header("Classification Metrics")
st.write("Accuracy:", accuracy_score(y_test, pred))
st.write("Precision:", precision_score(y_test, pred, average='weighted'))
st.write("Recall:", recall_score(y_test, pred, average='weighted'))
st.write("F1 Score:", f1_score(y_test, pred, average='weighted'))

# Regression
y_reg = df_encoded["EMI_Willingness"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2)

reg = RandomForestRegressor()
reg.fit(X_train_r, y_train_r)

# Clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.header("Clustering")
st.write(df['Cluster'].value_counts())

# Association Rules (reduced columns)
df_bin = pd.get_dummies(df[['Income','Location','AdoptionLikelihood']])
freq = apriori(df_bin, min_support=0.1, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.0)

st.header("Association Rules")
st.dataframe(rules.head())

