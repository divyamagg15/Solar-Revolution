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

# FIX: Define missing color scale
COOL_SCALE = ["#0d1117","#1a1f35","#38bdf8","#64ffda"]
SOLAR_SCALE = ["#0d1117","#1a1f35","#f7971e","#ffd200"]

st.title("Solar Analytics Dashboard (Fixed Version)")
