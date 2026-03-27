
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

st.title("Solar Analytics Dashboard")

df = pd.read_csv("dataset.csv")

st.header("Overview")
st.write(df.head())

fig = px.histogram(df, x="AdoptionLikelihood")
st.plotly_chart(fig)

# Encoding
df_encoded = df.copy()
le = LabelEncoder()
for col in df.columns:
    df_encoded[col] = le.fit_transform(df[col])

# Classification
X = df_encoded.drop("AdoptionLikelihood", axis=1)
y = df_encoded["AdoptionLikelihood"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

st.header("Classification Metrics")
st.write("Accuracy:", accuracy_score(y_test,pred))
st.write("Precision:", precision_score(y_test,pred,average='weighted'))
st.write("Recall:", recall_score(y_test,pred,average='weighted'))
st.write("F1:", f1_score(y_test,pred,average='weighted'))

# Regression (EMI as proxy)
y_reg = df_encoded["EMI_Willingness"]
reg = RandomForestRegressor()
reg.fit(X_train, y_reg.loc[X_train.index])

st.header("Clustering")
kmeans = KMeans(n_clusters=4)
df['Cluster'] = kmeans.fit_predict(X)
st.write(df[['Cluster']].value_counts())

# Association rules (dummy one-hot)
df_bin = pd.get_dummies(df)
freq = apriori(df_bin, min_support=0.1, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=1.0)

st.header("Association Rules")
st.write(rules.head())

st.header("Prediction Tool")
input_data = st.text_input("Enter sample row (comma separated values)")
