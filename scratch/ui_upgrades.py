import re
import sys

file_path = 'c:/Users/Vijay/Desktop/Data-Preprocessing-and-Exploratory-Data-Analysis-Streamlit-Project/streamlit_app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add AgGrid and Lottie Imports
imports = """
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, VotingRegressor, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, mean_absolute_error, roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
import shap
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_lottie import st_lottie
import requests
"""

content = re.sub(r'import streamlit as st\n.*import shap\nfrom io import BytesIO\n', imports, content, flags=re.DOTALL)

# Add Lottie helper function
lottie_func = """
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
"""

# Insert lottie_func after set_page_style
content = content.replace("def set_page_style():", lottie_func + "\n\ndef set_page_style():")

# 2. Replace st.success with st.toast where appropriate
content = content.replace('st.success("File uploaded successfully!")', 'st.toast("✅ File uploaded successfully!")')
content = content.replace('st.success("Dataset loaded from URL!")', 'st.toast("✅ Dataset loaded from URL!")')
content = content.replace('st.success("✅ Transformation Applied Successfully!")', 'st.toast("✨ Transformation Applied Successfully!")')
content = content.replace('st.success("✅ Training complete!")', 'st.toast("🎯 Training complete!")')
content = content.replace('st.success("Hyperparameter search complete!")', 'st.toast("🔍 Hyperparameter search complete!")')
content = content.replace('st.success("SHAP calculation successful!")', 'st.toast("🧠 SHAP calculation successful!")')
content = content.replace('st.success("Model loaded successfully! You can now use the Prediction page.")', 'st.toast("💾 Model loaded successfully!")')
content = content.replace('st.success(f"Model saved as: {model_filename}")', 'st.toast(f"💾 Model saved as: {model_filename}")')

# 3. Replace st.dataframe with AgGrid in Overview and Transform
# Let's find st.dataframe(df.head(20))
aggrid_code = """
        gb = GridOptionsBuilder.from_dataframe(df.head(100))
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        gridOptions = gb.build()
        AgGrid(df.head(100), gridOptions=gridOptions, enable_enterprise_modules=True, theme='alpine')
"""
content = content.replace('st.dataframe(df.head(20), use_container_width=True)', aggrid_code)
content = content.replace('st.dataframe(df, use_container_width=True)', aggrid_code.replace("df.head(100)", "df"))
content = content.replace('st.dataframe(df.head(20))', aggrid_code)

# 4. Add Lottie animation to empty state
empty_state_old = """
        st.info("👋 Welcome! Please upload a dataset to begin your analysis.")
"""
empty_state_new = """
        st.info("👋 Welcome! Please upload a dataset to begin your analysis.")
        lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")
        if lottie_hello:
            st_lottie(lottie_hello, height=300, key="hello")
"""
content = content.replace(empty_state_old, empty_state_new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("UI Upgrades Script Executed Successfully!")
