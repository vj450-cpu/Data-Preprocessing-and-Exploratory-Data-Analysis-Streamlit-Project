import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, RandomizedSearchCV
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, VotingRegressor, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from scipy.stats import chi2_contingency, f_oneway, pearsonr

import plotly.io as pio
pio.templates.default = "plotly_white"

sns.set_theme(style="whitegrid", palette="pastel")
import traceback


# Helper function to add dtype labels to columns
def add_dtype_label(df, col):
    """Returns column name with dtype label"""
    dtype = str(df[col].dtype)
    return f"{col} ({dtype})"

def get_columns_with_dtype(df, columns):
    """Returns list of columns with dtype labels for display"""
    return [add_dtype_label(df, col) for col in columns]

def extract_column_name(label):
    """Extracts column name from label like 'age (int64)' -> 'age'"""
    return label.rsplit(' (', 1)[0]



@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def set_page_style():
    st.set_page_config(page_title="Modern EDA Studio", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            color: #111827;
            background: #eef2ff;
        }

        .stApp {
            background: radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 28%),
                        radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.14), transparent 24%),
                        linear-gradient(180deg, #f8fafc 0%, #eef2ff 40%, #e0f2fe 100%);
            min-height: 100vh;
        }

        .hero-banner {
            position: relative;
            padding: 48px 42px;
            border-radius: 32px;
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 28px 80px rgba(15, 23, 42, 0.22);
            overflow: hidden;
            margin-bottom: 36px;
        }
        .hero-banner::before,
        .hero-banner::after {
            content: '';
            position: absolute;
            border-radius: 50%;
            opacity: 0.45;
            filter: blur(32px);
            animation: float 14s ease-in-out infinite;
        }
        .hero-banner::before {
            width: 220px;
            height: 220px;
            top: -60px;
            right: -70px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.75), transparent 60%);
        }
        .hero-banner::after {
            width: 180px;
            height: 180px;
            bottom: -50px;
            left: -40px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.65), transparent 60%);
            animation-delay: 4s;
        }

        .hero-heading {
            font-size: clamp(3rem, 4vw, 4.4rem);
            font-weight: 900;
            line-height: 1.02;
            letter-spacing: -0.06em;
            margin: 0;
            color: #f8fafc;
            background: linear-gradient(135deg, #93c5fd 0%, #a78bfa 40%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 18px 40px rgba(15, 23, 42, 0.26);
        }

        .hero-subtitle {
            font-size: 1.05rem;
            max-width: 860px;
            margin-top: 18px;
            color: rgba(241, 245, 249, 0.92);
            line-height: 1.8;
        }

        .hero-pill-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 14px;
            margin-top: 28px;
        }

        .hero-pill {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 999px;
            padding: 12px 18px;
            color: #e2e8f0;
            font-weight: 600;
            backdrop-filter: blur(10px);
        }

        .main-heading {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.25rem;
            color: #0f172a;
        }

        .subheading {
            font-size: 1.05rem;
            color: #334155;
            margin-top: 0.2rem;
            margin-bottom: 1.8rem;
            max-width: 860px;
        }

        .section-card,
        .feature-panel,
        .model-panel,
        .predict-panel {
            background: rgba(255, 255, 255, 0.70);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border-radius: 32px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            box-shadow: 0 28px 70px rgba(15, 23, 42, 0.08);
            padding: 32px;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .section-card:hover,
        .feature-panel:hover,
        .model-panel:hover,
        .predict-panel:hover {
            transform: translateY(-4px);
            box-shadow: 0 32px 80px rgba(15, 23, 42, 0.12);
        }
        .section-card::before {
            content: '';
            position: absolute;
            width: 220px;
            height: 220px;
            top: -90px;
            right: -80px;
            background: radial-gradient(circle, rgba(59, 130, 246, 0.12), transparent 65%);
        }

        .info-banner {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(16, 185, 129, 0.16));
            border-radius: 24px;
            border: 1px solid rgba(59, 130, 246, 0.16);
            padding: 20px 26px;
            margin-bottom: 24px;
            color: #0f172a;
            box-shadow: 0 18px 34px rgba(15, 23, 42, 0.07);
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 18px 35px rgba(15, 23, 42, 0.06);
            position: relative;
            overflow: hidden;
            margin-bottom: 18px;
            transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px) scale(1.02);
            box-shadow: 0 24px 45px rgba(99, 102, 241, 0.15);
            background: rgba(255, 255, 255, 0.85);
        }
        .metric-card h3 {
            margin: 0;
            font-size: 0.95rem;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .metric-card p.highlight {
            margin: 14px 0 0;
            font-size: 2rem;
            font-weight: 800;
            color: #4338ca;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
            border-radius: 18px;
            padding: 0.8rem 1.6rem;
            font-weight: 700;
            transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.25s ease;
        }

        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #7c3aed 0%, #2563eb 60%, #38bdf8 100%);
            color: white;
            box-shadow: 0 20px 42px rgba(59, 130, 246, 0.25);
            border: none;
            text-shadow: 0 1px 8px rgba(15, 23, 42, 0.2);
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-1px) scale(1.01);
            background: linear-gradient(135deg, #5b21b6 0%, #2563eb 60%, #22d3ee 100%);
            box-shadow: 0 24px 50px rgba(59, 130, 246, 0.28);
        }

        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #14b8a6 0%, #0f766e 60%, #22c55e 100%);
            color: white;
            box-shadow: 0 20px 42px rgba(16, 185, 129, 0.24);
            border: none;
            text-shadow: 0 1px 8px rgba(15, 23, 42, 0.2);
        }

        div[data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-1px) scale(1.01);
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 60%, #22c55e 100%);
            box-shadow: 0 24px 50px rgba(16, 185, 129, 0.28);
        }

        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.93) !important;
            color: #e2e8f0 !important;
            backdrop-filter: blur(24px);
            box-shadow: inset 0 0 0 1px rgba(96, 165, 250, 0.14);
            border-right: 1px solid rgba(96, 165, 250, 0.18) !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div {
            color: #f1f5f9 !important;
        }

        section[data-testid="stSidebar"] [role="radio"] {
            color: #cbd5e1 !important;
        }

        .css-1aumxhk {
            background: transparent !important;
        }

        .stSelectbox div[data-baseweb="select"] > div,
        .stTextInput div[data-baseweb="input"] > div,
        .stNumberInput div[data-baseweb="input"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            border-radius: 16px !important;
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
            background-color: rgba(255, 255, 255, 0.9) !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06) !important;
        }
        .stSelectbox div[data-baseweb="select"] > div:hover,
        .stTextInput div[data-baseweb="input"] > div:hover,
        .stNumberInput div[data-baseweb="input"] > div:hover,
        .stMultiSelect div[data-baseweb="select"] > div:hover {
            border-color: rgba(99, 102, 241, 0.7) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
        }

        .streamlit-expanderHeader {
            font-family: 'Outfit', sans-serif;
            font-weight: 700 !important;
            border-radius: 18px !important;
            background: rgba(59, 130, 246, 0.08) !important;
            border: 1px solid rgba(59, 130, 246, 0.16) !important;
            color: #0f172a !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 24px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 20px 36px rgba(15, 23, 42, 0.08);
            background: rgba(255,255,255,0.98);
        }

        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #1e40af;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) scale(1); }
            50% { transform: translateY(-18px) scale(1.05); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Could not load file: {exc}")
        return None


def summary_cards(df):
    rows, cols = df.shape
    numeric = df.select_dtypes(include=np.number).shape[1]
    categorical = df.select_dtypes(include=["object", "category"]).shape[1]
    missing = int(df.isna().sum().sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("<div class='metric-card'><h3>Rows</h3><p class='highlight'>" + str(rows) + "</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-card'><h3>Columns</h3><p class='highlight'>" + str(cols) + "</p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-card'><h3>Numeric</h3><p class='highlight'>" + str(numeric) + "</p></div>", unsafe_allow_html=True)
    col4.markdown("<div class='metric-card'><h3>Missing</h3><p class='highlight'>" + str(missing) + "</p></div>", unsafe_allow_html=True)


def render_insights(df):
    missing_cols = df.isna().sum().sort_values(ascending=False)
    most_missing = missing_cols[missing_cols > 0]
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### Quick Insights")
    st.write(
        "This workspace helps you inspect structure, clean missing values, transform features, and build beautiful visualizations in one place. Use the controls on the right to focus on the analysis you need."
    )
    if len(most_missing) > 0:
        top = most_missing.index[0]
        st.info(f"Column **{top}** has the highest missing rate with {most_missing.iloc[0]} missing values.")
    else:
        st.success("No missing values detected in this dataset.")
    st.markdown("</div>", unsafe_allow_html=True)


def plot_missing(df):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=True)
    if miss.empty:
        st.success("This dataset has no missing values.")
        return

    fig = px.bar(
        x=miss.values,
        y=miss.index,
        orientation="h",
        labels={"x": "Missing count", "y": "Column"},
        color=miss.values,
        color_continuous_scale="blues",
    )
    st.plotly_chart(fig, use_container_width=True)


def detect_outliers(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    with st.expander("Outlier detection", expanded=False):
        if not numeric_columns:
            st.warning("No numeric columns available for outlier detection.")
            return df

        col = st.selectbox("Numeric column", numeric_columns, key="outlier-col")
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if st.button("Inspect outliers", key="outlier-inspect"):
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            st.write(f"{len(outliers)} outlier rows detected for {col}.")
            st.write(f"Lower bound: {lower:.3f}, upper bound: {upper:.3f}")
            st.dataframe(outliers.head(20))
            fig = px.box(df, y=col, title=f"Outlier detection for {col}")
            st.plotly_chart(fig, use_container_width=True)

        if st.button("Remove detected outliers", key="outlier-remove"):
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            st.success("Outliers removed from dataset.")
            st.dataframe(df.head())
    return df


def imbalance_analysis(df):
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    with st.expander("Data imbalance analysis", expanded=False):
        if not categorical_columns:
            st.warning("No categorical columns available for imbalance analysis.")
            return

        col = st.selectbox("Categorical column", categorical_columns, key="imbalance-col")
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        counts["percentage"] = (counts["count"] / counts["count"].sum()) * 100
        st.dataframe(counts)
        fig = px.bar(counts, x=col, y="count", title=f"Class distribution for {col}")
        st.plotly_chart(fig, use_container_width=True)

        if counts.shape[0] > 0:
            dominant = counts.iloc[0]
            st.info(f"Most frequent class is {dominant[col]} with {dominant['percentage']:.1f}% of rows.")


def statistical_tests_section(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_display = get_columns_with_dtype(df, numeric_columns)
    cat_display = get_columns_with_dtype(df, categorical_columns)
    
    with st.expander("Statistical tests", expanded=False):
        test_type = st.selectbox(
            "Choose test type",
            ["Univariate summary", "Numeric vs Numeric", "Categorical vs Numeric", "Categorical vs Categorical"],
            key="test-type",
        )

        if test_type == "Univariate summary":
            field_type = st.selectbox("Field type", ["Numeric", "Categorical"], key="univariate-type")
            if field_type == "Numeric":
                if not numeric_columns:
                    st.warning("No numeric columns available.")
                else:
                    col_label = st.selectbox("Numeric column", num_display, key="uni-num-col")
                    col = extract_column_name(col_label)
                    st.write(df[col].describe())
                    fig = px.histogram(df, x=col, nbins=40, title=f"Distribution for {col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                if not categorical_columns:
                    st.warning("No categorical columns available.")
                else:
                    col_label = st.selectbox("Categorical column", cat_display, key="uni-cat-col")
                    col = extract_column_name(col_label)
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, "count"]
                    st.dataframe(counts)
                    fig = px.bar(counts, x=col, y="count", title=f"Frequency of {col}")
                    st.plotly_chart(fig, use_container_width=True)

        elif test_type == "Numeric vs Numeric":
            if len(numeric_columns) < 2:
                st.warning("At least two numeric columns required.")
            else:
                x_col_label = st.selectbox("X numeric column", num_display, key="bi-x-col")
                x_col = extract_column_name(x_col_label)
                y_col_label = st.selectbox("Y numeric column", [l for l in num_display if extract_column_name(l) != x_col], key="bi-y-col")
                y_col = extract_column_name(y_col_label)
                data = df[[x_col, y_col]].dropna()
                corr, pvalue = pearsonr(data[x_col], data[y_col])
                st.write(f"Pearson correlation between {x_col} and {y_col}: {corr:.3f} (p={pvalue:.3f})")
                fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif test_type == "Categorical vs Numeric":
            if not categorical_columns or not numeric_columns:
                st.warning("Need at least one categorical and one numeric column.")
            else:
                cat_col_label = st.selectbox("Categorical column", cat_display, key="cat-num-cat")
                cat_col = extract_column_name(cat_col_label)
                num_col_label = st.selectbox("Numeric column", num_display, key="cat-num-num")
                num_col = extract_column_name(num_col_label)
                groups = [group[num_col].dropna() for _, group in df.groupby(cat_col)]
                if len(groups) > 1:
                    try:
                        stat, p = f_oneway(*groups)
                        st.write(f"ANOVA F={stat:.3f}, p={p:.3f}")
                    except Exception as exc:
                        st.warning(f"ANOVA not available: {exc}")
                st.write(df.groupby(cat_col)[num_col].describe())

        else:
            if len(categorical_columns) < 2:
                st.warning("Need at least two categorical columns.")
            else:
                cat1_label = st.selectbox("First categorical column", cat_display, key="cat-cat-1")
                cat1 = extract_column_name(cat1_label)
                cat2_label = st.selectbox("Second categorical column", [l for l in cat_display if extract_column_name(l) != cat1], key="cat-cat-2")
                cat2 = extract_column_name(cat2_label)
                contingency = pd.crosstab(df[cat1], df[cat2])
                st.dataframe(contingency)
                try:
                    chi2, p, dof, expected = chi2_contingency(contingency)
                    st.write(f"Chi-squared: {chi2:.3f}, p={p:.3f}")
                except Exception as exc:
                    st.warning(f"Chi-squared test could not be computed: {exc}")


def duplicate_detection(df):
    with st.expander("Duplicate row detection", expanded=False):
        dup = df.duplicated(keep=False)
        count = dup.sum()
        st.write(f"Duplicate rows found: {count}")
        if count > 0:
            st.dataframe(df[dup].head(20))
            if st.button("Remove duplicate rows", key="duplicate-remove"):
                df = df.drop_duplicates()
                st.success("Duplicate rows removed.")
                st.dataframe(df.head())
    return df


def plot_correlation(df):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        st.warning("At least two numeric columns are required for a correlation heatmap.")
        return

    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation matrix",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_distribution(df, column):
    fig = px.histogram(df, x=column, nbins=40, title=f"Distribution of {column}", marginal="box")
    st.plotly_chart(fig, use_container_width=True)


def plot_pairwise(df, columns):
    if len(columns) < 2:
        st.warning("Select at least two numeric columns for pairwise plotting.")
        return

    sample = df[columns].dropna().sample(min(200, len(df)), random_state=42)
    fig = px.scatter_matrix(sample, dimensions=columns, title="Pairwise relationships")
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def download_button(df, file_name="processed_data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download processed dataset", csv, file_name, "text/csv", key="download-csv")


def transform_missing(df):
    cols_display = get_columns_with_dtype(df, df.columns.tolist())
    col_label = st.selectbox("Missing value column", cols_display, key="missing-col")
    col = extract_column_name(col_label)
    method = st.radio("Fill strategy", ["Drop rows", "Mean", "Median", "Mode", "Constant"], key="missing-method")
    constant = None
    if method == "Constant":
        constant = st.text_input("Constant value", value="0", key="missing-constant")
    if st.button("Apply missing strategy", key="missing-apply"):
        if method == "Drop rows":
            df = df.dropna(subset=[col])
        elif method == "Mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "Median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "Mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        else:
            df[col] = df[col].fillna(constant)
        st.success("Missing values updated")
        st.dataframe(df.head())
    return df


def transform_encoding(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.info("No categorical columns detected for encoding.")
        return df

    with st.expander("Categorical encoding options", expanded=True):
        cat_cols_display = get_columns_with_dtype(df, cat_cols)
        col_label = st.selectbox("Categorical column", cat_cols_display, key="enc-col")
        col = extract_column_name(col_label)
        enc_type = st.selectbox("Encoding type", ["One-Hot", "Ordinal", "Label"], key="enc-type")
        if st.button("Apply encoding", key="enc-apply"):
            if enc_type == "One-Hot":
                df = pd.get_dummies(df, columns=[col], drop_first=True)
            elif enc_type == "Ordinal":
                encoder = OrdinalEncoder()
                df[col] = encoder.fit_transform(df[[col]])
            else:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
            st.success("Encoding applied")
            st.dataframe(df.head())
    return df


def transform_scaling(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.info("No numeric columns available for scaling.")
        return df

    with st.expander("Scaling options", expanded=True):
        num_cols_display = get_columns_with_dtype(df, num_cols)
        col_label = st.selectbox("Numeric column", num_cols_display, key="scale-col")
        col = extract_column_name(col_label)
        scale_type = st.selectbox("Scaling method", ["None", "StandardScaler", "MinMaxScaler"], key="scale-type")
        if st.button("Apply scaling", key="scale-apply"):
            if scale_type == "StandardScaler":
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[[col]])
            elif scale_type == "MinMaxScaler":
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
            st.success("Scaling applied")
            st.dataframe(df.head())
    return df


def transform_pca(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        st.info("No numeric columns available for PCA.")
        return df

    with st.expander("Apply PCA", expanded=True):
        mode = st.radio(
            "Choose PCA mode",
            ["Number of components", "Explained variance (%)"],
            key="pca-mode",
        )

        if mode == "Number of components":
            max_components = min(len(numeric_columns), len(df))
            n_components = st.number_input(
                "Number of components",
                min_value=1,
                max_value=max_components,
                value=min(3, max_components),
                step=1,
                key="pca-n-components",
            )
        else:
            variance_target = st.slider(
                "Target explained variance (%)",
                min_value=50,
                max_value=100,
                value=90,
                step=1,
                key="pca-variance-target",
            )
            scaled_data = StandardScaler().fit_transform(df[numeric_columns].dropna())
            pca_temp = PCA().fit(scaled_data)
            cumulative = np.cumsum(pca_temp.explained_variance_ratio_) * 100
            n_components = int(np.searchsorted(cumulative, variance_target) + 1)
            n_components = min(n_components, len(numeric_columns))
            st.markdown(f"**Using {n_components} components to explain {variance_target}% of variance.**")

        if st.button("Apply PCA", key="pca-apply"):
            try:
                scaled = StandardScaler().fit_transform(df[numeric_columns].fillna(0))
                pca = PCA(n_components=n_components)
                transformed = pca.fit_transform(scaled)
                component_names = [f"PC{i + 1}" for i in range(n_components)]
                pca_df = pd.DataFrame(transformed, columns=component_names, index=df.index)
                other_columns = [c for c in df.columns if c not in numeric_columns]
                df = pd.concat([df[other_columns].reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
                explained = round(pca.explained_variance_ratio_.sum() * 100, 2)
                st.success(f"PCA applied with {n_components} components ({explained}% variance explained).")
                st.dataframe(df.head())

                st.markdown("### PCA visualization")
                variance_fig = px.bar(
                    x=[f"PC{i + 1}" for i in range(n_components)],
                    y=pca.explained_variance_ratio_ * 100,
                    labels={"x": "Principal Component", "y": "Explained Variance (%)"},
                    title="PCA explained variance by component",
                )
                st.plotly_chart(variance_fig, use_container_width=True)

                variance_table = pd.DataFrame({
                    "component": component_names,
                    "explained_variance_ratio": np.round(pca.explained_variance_ratio_, 4),
                    "cumulative_variance": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2),
                })
                st.markdown("**PCA summary metrics**")
                st.dataframe(variance_table)

                if n_components >= 2:
                    color_option = None
                    if other_columns:
                        cat_cols = [c for c in other_columns if df[c].dtype == object or pd.api.types.is_categorical_dtype(df[c])]
                        if cat_cols:
                            color_option = st.selectbox("Color PCA scatter by", ["None"] + cat_cols, key="pca-color")
                    scatter_fig = px.scatter(
                        pca_df,
                        x="PC1",
                        y="PC2",
                        color=None if color_option in [None, "None"] else df[color_option].astype(str),
                        title="PCA component scatter plot",
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)
            except Exception as exc:
                st.error(f"PCA failed: {exc}")
    return df


def visualization_section(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_display = get_columns_with_dtype(df, numeric_columns)
    cat_display = get_columns_with_dtype(df, categorical_columns)
    
    chart_type = st.selectbox(
        "Select visualization type",
        ["Scatter", "Line", "Bar", "Histogram", "Box", "Pie", "Heatmap", "Countplot"],
        key="viz-type",
    )

    if chart_type == "Heatmap":
        st.write("### Correlation heatmap for numeric columns")
        plot_correlation(df)
        return

    if chart_type == "Countplot":
        if not categorical_columns:
            st.warning("No categorical columns available for countplots.")
            return
        col_label = st.selectbox("Categorical column", cat_display, key="viz-count-col")
        col = extract_column_name(col_label)
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", color=col, title=f"Countplot of {col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Pie":
        if not categorical_columns:
            st.warning("No categorical columns available for pie charts.")
            return
        col_label = st.selectbox("Categorical column", cat_display, key="viz-pie-col")
        col = extract_column_name(col_label)
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.pie(counts, names=col, values="count", title=f"Pie chart of {col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type in ["Scatter", "Line", "Bar"]:
        if len(numeric_columns) < 2:
            st.warning("At least two numeric columns are needed for this chart.")
            return
        x_col_label = st.selectbox("X axis", num_display, key="viz-x")
        x_col = extract_column_name(x_col_label)
        y_col_label = st.selectbox("Y axis", [l for l in num_display if extract_column_name(l) != x_col], key="viz-y")
        y_col = extract_column_name(y_col_label)
        if chart_type == "Scatter":
            color_col = None
            if categorical_columns:
                color_options = ["None"] + cat_display
                color_label = st.selectbox("Color by (optional)", color_options, key="viz-color")
                if color_label == "None":
                    color_arg = None
                else:
                    color_arg = extract_column_name(color_label)
                fig = px.scatter(df, x=x_col, y=y_col, color=color_arg, title=f"{x_col} vs {y_col}")
            else:
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        else:
            fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Histogram":
        if not numeric_columns:
            st.warning("No numeric columns available for histograms.")
            return
        col_label = st.selectbox("Numeric column", num_display, key="viz-hist-col")
        col = extract_column_name(col_label)
        fig = px.histogram(df, x=col, nbins=40, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Box":
        if not numeric_columns:
            st.warning("No numeric columns available for box plots.")
            return
        col_label = st.selectbox("Numeric column", num_display, key="viz-box-col")
        col = extract_column_name(col_label)
        fig = px.box(df, y=col, title=f"Box plot of {col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Cluster segmentation":
        cluster_segmentation(df)
        return


def cluster_segmentation(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    num_display = get_columns_with_dtype(df, numeric_columns)
    
    with st.expander("Cluster segmentation", expanded=False):
        if len(numeric_columns) < 2:
            st.warning("Select at least two numeric columns for clustering.")
            return df

        selected_labels = st.multiselect("Numeric columns", num_display, default=num_display[:2] if len(num_display) >= 2 else num_display, key="cluster-cols")
        selected = [extract_column_name(label) for label in selected_labels]
        clusters = st.slider("Number of clusters", 2, min(10, len(df)), value=3, key="cluster-count")
        if st.button("Run clustering", key="cluster-run"):
            try:
                scaled = StandardScaler().fit_transform(df[selected].fillna(0))
                model = KMeans(n_clusters=clusters, random_state=42)
                labels = model.fit_predict(scaled)
                df["cluster_label"] = labels
                st.write(df["cluster_label"].value_counts())
                if len(selected) >= 2:
                    fig = px.scatter(
                        df,
                        x=selected[0],
                        y=selected[1],
                        color="cluster_label",
                        title="Cluster segmentation",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                try:
                    score = silhouette_score(scaled, labels)
                    st.write(f"Silhouette score: {score:.3f}")
                except Exception:
                    pass
                st.success("Clustering completed and added the column 'cluster_label'.")
            except Exception as exc:
                st.error(f"Clustering failed: {exc}")
    return df


def transform_types(df):
    cols = df.columns.tolist()
    if not cols:
        st.info("No columns are available for type conversion.")
        return df

    with st.expander("Change column data type", expanded=True):
        cols_display = get_columns_with_dtype(df, cols)
        col_label = st.selectbox("Column to convert", cols_display, key="type-col")
        col = extract_column_name(col_label)
        target_type = st.selectbox(
            "Target type",
            ["int", "float", "string", "datetime", "category"],
            key="type-target",
        )
        if st.button("Apply type conversion", key="type-apply"):
            try:
                if target_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(target_type)
                st.success(f"Column '{col}' converted to {target_type}.")
                st.dataframe(df[[col]].head())
            except Exception as exc:
                st.error(f"Could not convert {col} to {target_type}: {exc}")
    return df


def transform_delete_columns(df):
    cols = df.columns.tolist()
    if not cols:
        st.info("No columns available to delete.")
        return df

    with st.expander("Delete columns", expanded=True):
        cols_display = get_columns_with_dtype(df, cols)
        delete_cols_labels = st.multiselect("Select columns to remove", cols_display, key="delete-cols")
        delete_cols = [extract_column_name(label) for label in delete_cols_labels]
        if delete_cols and st.button("Delete selected columns", key="delete-apply"):
            try:
                df = df.drop(columns=delete_cols)
                st.success(f"Deleted columns: {', '.join(delete_cols)}")
                st.dataframe(df.head())
            except Exception as exc:
                st.error(f"Could not delete columns: {exc}")
    return df


def transform_feature_engineering(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    datetime_columns = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    with st.expander("Feature engineering toolkit", expanded=True):
        st.markdown("### Create derived features")
        if numeric_columns:
            feature_mode = st.selectbox(
                "Select feature engineering operation",
                [
                    "None",
                    "Arithmetic combination",
                    "Polynomial features",
                    "Binning / discretization",
                    "Datetime extraction",
                ],
                key="feature-mode",
            )

            if feature_mode == "Arithmetic combination":
                col1 = st.selectbox("First numeric column", numeric_columns, key="arith-col1")
                col2 = st.selectbox("Second numeric column", [c for c in numeric_columns if c != col1], key="arith-col2")
                op = st.selectbox("Operation", ["Add", "Subtract", "Multiply", "Divide"], key="arith-op")
                new_name = st.text_input("New feature name", value=f"{col1}_{op.lower()}_{col2}", key="arith-name")
                if st.button("Create feature", key="arith-apply"):
                    try:
                        if op == "Add":
                            df[new_name] = df[col1] + df[col2]
                        elif op == "Subtract":
                            df[new_name] = df[col1] - df[col2]
                        elif op == "Multiply":
                            df[new_name] = df[col1] * df[col2]
                        else:
                            df[new_name] = df[col1] / df[col2].replace({0: np.nan})
                        st.success(f"Created feature '{new_name}'.")
                        st.dataframe(df[[new_name]].head())
                    except Exception as exc:
                        st.error(f"Could not create feature: {exc}")

            elif feature_mode == "Polynomial features":
                selected = st.multiselect(
                    "Numeric columns",
                    numeric_columns,
                    default=numeric_columns[:2],
                    key="poly-cols",
                )
                degree = st.slider("Polynomial degree", 2, 3, value=2, key="poly-degree")
                if st.button("Generate polynomial features", key="poly-apply"):
                    try:
                        if len(selected) < 2:
                            st.warning("Select at least two numeric columns.")
                        else:
                            transformer = PolynomialFeatures(degree=degree, include_bias=False)
                            poly_data = transformer.fit_transform(df[selected].fillna(0))
                            feature_names = transformer.get_feature_names_out(selected)
                            new_cols = [name for name in feature_names if name not in selected]
                            poly_df = pd.DataFrame(poly_data, columns=feature_names, index=df.index)
                            df = pd.concat([df, poly_df[new_cols]], axis=1)
                            st.success(f"Added {len(new_cols)} polynomial features.")
                            st.dataframe(df.head())
                    except Exception as exc:
                        st.error(f"Polynomial feature generation failed: {exc}")

            elif feature_mode == "Binning / discretization":
                col = st.selectbox("Numeric column to bin", numeric_columns, key="bin-col")
                method = st.selectbox("Method", ["Equal-width", "Quantile"], key="bin-method")
                bins = st.slider("Number of bins", 2, 10, value=4, key="bin-count")
                new_name = st.text_input("New binned column name", value=f"{col}_binned", key="bin-name")

                labels = None
                use_custom_labels = st.checkbox("Use manual bin labels", key="bin-custom-labels")
                if use_custom_labels:
                    custom_labels = []
                    for i in range(bins):
                        label_text = st.text_input(
                            f"Label for category {i + 1}",
                            value=f"Category {i + 1}",
                            key=f"bin-label-{i}",
                        )
                        custom_labels.append(label_text.strip())

                    if all(custom_labels):
                        labels = custom_labels
                    else:
                        st.warning("Please provide a label for every bin.")
                elif bins == 3:
                    label_option = st.selectbox(
                        "Bin label style",
                        ["Range labels", "Low / Medium / High"],
                        key="bin-label-option",
                    )
                    if label_option == "Low / Medium / High":
                        labels = ["Low", "Medium", "High"]

                if st.button("Create bins", key="bin-apply"):
                    try:
                        if method == "Quantile":
                            df[new_name] = pd.qcut(df[col], q=bins, labels=labels, duplicates="drop")
                        else:
                            df[new_name] = pd.cut(df[col], bins=bins, labels=labels)
                        st.success(f"Created binned column '{new_name}'.")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Dataset Preview (First 5 Rows)**")
                            st.dataframe(df[[new_name]].head())
                        with col2:
                            st.markdown("**Bin Summary (Total Counts)**")
                            st.dataframe(df[new_name].value_counts().reset_index())
                    except Exception as exc:
                        st.error(f"Binning failed: {exc}")

            elif feature_mode == "Datetime extraction":
                all_datetime = datetime_columns + [c for c in df.columns if df[c].dtype == object]
                if all_datetime:
                    dt_col = st.selectbox("Possible datetime column", all_datetime, key="dt-col")
                    if dt_col not in datetime_columns:
                        converted = pd.to_datetime(df[dt_col], errors="coerce")
                    else:
                        converted = df[dt_col]
                    date_parts = st.multiselect(
                        "Date parts to extract",
                        ["year", "month", "day", "hour", "minute", "weekday"],
                        default=["year", "month", "day"],
                        key="date-parts",
                    )
                    if st.button("Extract date features", key="dt-apply"):
                        try:
                            if converted.isna().all():
                                raise ValueError("Column could not be parsed as dates.")
                            for part in date_parts:
                                df[f"{dt_col}_{part}"] = getattr(converted.dt, part)
                            st.success("Datetime features extracted.")
                            st.dataframe(df[[f"{dt_col}_{part}" for part in date_parts]].head())
                        except Exception as exc:
                            st.error(f"Datetime extraction failed: {exc}")
                else:
                    st.warning("No datetime-like columns available for extraction.")

        else:
            st.warning("At least one numeric column is required for feature engineering.")

    return df


def create_model_by_name(model_name):
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Decision Tree Regressor":
        return DecisionTreeRegressor(random_state=42)
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(random_state=42)
    if model_name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(random_state=42)
    if model_name == "K-Nearest Neighbors Regressor":
        return KNeighborsRegressor()
    if model_name == "Support Vector Regressor":
        return SVR()
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1500, solver='lbfgs', random_state=42)
    if model_name == "Decision Tree Classifier":
        return DecisionTreeClassifier(random_state=42)
    if model_name == "Random Forest Classifier":
        return RandomForestClassifier(random_state=42)
    if model_name == "Gradient Boosting Classifier":
        return GradientBoostingClassifier(random_state=42)
    if model_name == "K-Nearest Neighbors Classifier":
        return KNeighborsClassifier()
    if model_name == "Support Vector Classifier":
        return SVC(probability=True, random_state=42)
    if model_name == "Voting Regressor":
        return VotingRegressor(estimators=[
            ('rf', RandomForestRegressor(random_state=42)),
            ('gb', GradientBoostingRegressor(random_state=42)),
            ('lr', LinearRegression())
        ])
    if model_name == "Voting Classifier":
        return VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('lr', LogisticRegression(max_iter=1500, solver='lbfgs', random_state=42))
        ], voting='soft')
    return None


def get_search_space(model_name):
    if model_name in ["Random Forest Regressor", "Random Forest Classifier"]:
        return {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 6, 10, 16],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 4],
        }
    if model_name in ["Gradient Boosting Regressor", "Gradient Boosting Classifier"]:
        return {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 8],
        }
    if model_name in ["Decision Tree Regressor", "Decision Tree Classifier"]:
        return {
            'max_depth': [None, 4, 6, 10],
            'min_samples_split': [2, 4, 8],
            'min_samples_leaf': [1, 2, 4],
        }
    if model_name in ["K-Nearest Neighbors Regressor", "K-Nearest Neighbors Classifier"]:
        return {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }
    if model_name == "Support Vector Regressor":
        return {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly']
        }
    if model_name == "Support Vector Classifier":
        return {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly']
        }
    if model_name == "Logistic Regression":
        return {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        }
    return {}


def predict_section(df):
    st.markdown("## 🚀 Model Prediction & Scoring")
    st.markdown("Use your trained model to score new examples, inspect sample predictions, or upload a CSV for batch scoring.")

    model = st.session_state.get('trained_model')
    model_features = st.session_state.get('trained_features')
    target_col = st.session_state.get('trained_target')
    task = st.session_state.get('trained_task')
    model_name = st.session_state.get('trained_model_name')
    scaler = st.session_state.get('trained_scaler')

    if model is None or model_features is None:
        st.warning("No trained model available yet. Train a model first on the Modeling page.")
        return df

    st.markdown(f"### Active model: **{model_name}**")
    st.info(f"Target: {target_col} | Task: {task} | Features: {len(model_features)}")

    with st.expander("🎛️ Interactive What-If Simulator", expanded=True):
        st.markdown("Adjust the sliders to instantly see how the prediction changes!")
        sample_inputs = {}
        cols = st.columns(3)
        for idx, feature in enumerate(model_features):
            with cols[idx % 3]:
                # Get stats for slider
                f_min = float(df[feature].min())
                f_max = float(df[feature].max())
                f_mean = float(df[feature].mean())
                # Handle edge case where min == max
                if f_min == f_max:
                    f_min = f_min - 1.0
                    f_max = f_max + 1.0
                step = (f_max - f_min) / 100.0 if (f_max - f_min) > 0 else 0.1
                sample_inputs[feature] = st.slider(feature, min_value=f_min, max_value=f_max, value=f_mean, step=step, key=f'pred-slider-{feature}')

        # Real-time Prediction
        try:
            sample_df = pd.DataFrame([sample_inputs])
            if scaler is not None:
                sample_df = pd.DataFrame(scaler.transform(sample_df), columns=sample_df.columns)
            prediction = model.predict(sample_df)
            
            st.markdown("---")
            if task == "Classification" and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(sample_df)[0]
                st.markdown(f"### 🎯 Predicted Class: **{prediction[0]}**")
                
                # Show probability bar chart
                prob_df = pd.DataFrame({'Class': model.classes_, 'Probability': proba})
                fig_prob = px.bar(prob_df, x='Probability', y='Class', orientation='h', title="Prediction Probabilities", text_auto='.2%', range_x=[0, 1])
                fig_prob.update_layout(height=200)
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.markdown(f"### 🎯 Predicted Value: **{float(prediction[0]):.4f}**")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    with st.expander("💻 Generate API Code Snippet", expanded=False):
        st.markdown("Want to host this model? Copy this `FastAPI` snippet to serve your `.joblib` model as a REST API:")
        features_str = ", ".join([f"'{f}'" for f in model_features])
        api_code = f'''from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="{model_name} API")

# Load your saved model (Make sure to update the path!)
model = joblib.load("models/your_saved_model.joblib")

class InputData(BaseModel):
    # Defining expected input fields based on your trained features
'''
        for f in model_features:
            api_code += f"    {f.replace(' ', '_')}: float\n"
            
        api_code += f'''
@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict(by_alias=True)])
    
    # Optional: Apply scaling if your model expects scaled data
    # Make sure to also load and apply your saved StandardScaler here!
    
    # Make prediction
    prediction = model.predict(input_df)
    return {{"prediction": prediction[0]}}

# Run with: uvicorn app:app --reload
'''
        st.code(api_code, language='python')

    with st.expander("Batch scoring from CSV", expanded=False):
        uploaded_csv = st.file_uploader("Upload a CSV with feature rows", type=['csv'], key="predict-upload")
        if uploaded_csv is not None:
            batch_df = pd.read_csv(uploaded_csv)
            missing_features = [f for f in model_features if f not in batch_df.columns]
            if missing_features:
                st.error(f"Missing required feature columns: {', '.join(missing_features)}")
            else:
                batch_data = batch_df[model_features].apply(pd.to_numeric, errors='coerce')
                if batch_data.isna().any().any():
                    st.warning("Uploaded data contains non-numeric values or missing cells. Clean the file and try again.")
                else:
                    if scaler is not None:
                        batch_data = pd.DataFrame(scaler.transform(batch_data), columns=batch_data.columns)
                    predictions = model.predict(batch_data)
                    batch_df['prediction'] = predictions
                    st.success("Batch scoring complete.")
                    st.dataframe(batch_df.head(10))
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download scored CSV", csv, file_name="scored_predictions.csv", mime='text/csv')

    return df


def automl_section(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not numeric_columns:
        st.info("No numeric columns available. Machine learning models require numeric features.")
        return df

    st.markdown("## 🏎️ AutoML: One-Click Model Comparison")
    st.markdown("Automatically train and compare multiple algorithms to find the best fit for your data.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        task = st.radio("🎯 Machine Learning Task", ["Regression", "Classification"], key="automl-task")

    target_options = numeric_columns if task == "Regression" else numeric_columns + categorical_columns
    target_options_display = get_columns_with_dtype(df, target_options)
    with col2:
        target_label = st.selectbox("🎯 Select Target Variable", target_options_display, key="automl-target")
    
    target_col = extract_column_name(target_label)

    features = [col for col in numeric_columns if col != target_col]
    if not features:
        st.warning("You must have at least one numeric feature column apart from the target.")
        return df

    features_display = get_columns_with_dtype(df, features)
    selected_features_labels = st.multiselect("Select Feature Variables", features_display, default=features_display[:min(5, len(features_display))], key="automl-features")
    selected_features = [extract_column_name(label) for label in selected_features_labels]
    if not selected_features:
        st.error("Please select at least one feature.")
        return df
        
    if st.button("🚀 Run AutoML", type="primary", key="automl-run"):
        X = df[selected_features].dropna()
        y = df.loc[X.index, target_col].dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 10:
            st.error("Not enough data remaining after dropping missing values.")
            return df
            
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        if task == "Regression":
            models_to_try = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42)
            }
            scoring = 'r2'
        else:
            models_to_try = {
                "Logistic Regression": LogisticRegression(max_iter=1500, solver='liblinear', random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }
            scoring = 'accuracy'
            
        results = []
        progress_text = "Running AutoML. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        best_model = None
        best_score = -float('inf')
        best_model_name = ""
        
        items = list(models_to_try.items())
        for i, (name, model) in enumerate(items):
            try:
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring=scoring)
                mean_score = scores.mean()
                results.append({"Model": name, "Score": mean_score})
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_model_name = name
            except Exception as e:
                results.append({"Model": name, "Score": 0.0})
            my_bar.progress((i + 1) / len(items), text=f"Evaluated {name}...")
            
        my_bar.empty()
        st.success("AutoML complete!")
        
        # Plot results
        res_df = pd.DataFrame(results).sort_values("Score", ascending=False)
        fig = px.bar(res_df, x="Model", y="Score", title="Model Comparison", color="Score", color_continuous_scale="Viridis")
        fig.update_layout(yaxis_title="R² Score" if task == "Regression" else "Accuracy")
        st.plotly_chart(fig, use_container_width=True)
        
        # Save best model
        if best_model is not None:
            st.markdown(f"### 🏆 Winner: **{best_model_name}**")
            st.info(f"The best model has been automatically saved to your session! You can now go to the **Prediction** tab to use it.")
            best_model.fit(X_scaled, y)
            st.session_state['trained_model'] = best_model
            st.session_state['trained_scaler'] = scaler
            st.session_state['trained_features'] = selected_features
            st.session_state['trained_target'] = target_col
            st.session_state['trained_task'] = task
            st.session_state['trained_model_name'] = best_model_name

    return df


def modeling_section(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not numeric_columns:
        st.info("No numeric columns available. Machine learning models require numeric features.")
        return df

    st.markdown("## 🤖 Advanced Machine Learning Studio")
    st.markdown("Train, evaluate, and compare ML models with intelligent suggestions and advanced features.")
    st.markdown("*(Note: Ensure your features are numeric and any categorical target is prepared before training.)*")
    
    # Model Suggestions Section
    with st.expander("🎯 Smart Model Suggestions", expanded=True):
        st.markdown("### Dataset Analysis & Recommendations")
        
        # Analyze dataset characteristics
        n_samples, n_features = df.shape
        possible_targets = numeric_columns + categorical_columns
        target_candidates = []
        
        # Suggest potential targets based on column names and characteristics
        for col in possible_targets:
            if any(keyword in col.lower() for keyword in ['target', 'label', 'class', 'outcome', 'response', 'price', 'cost', 'salary', 'income', 'sales']):
                target_candidates.append(col)
        
        # If no obvious targets, suggest based on variance (low variance might be target)
        if not target_candidates and numeric_columns:
            variances = df[numeric_columns].var()
            low_variance_cols = variances[variances < variances.quantile(0.3)].index.tolist()
            target_candidates = low_variance_cols[:3]  # Suggest up to 3
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", f"{n_samples:,} rows")
        with col2:
            st.metric("Features Available", len(numeric_columns))
        with col3:
            st.metric("Suggested Targets", len(target_candidates))
        
        if target_candidates:
            st.info(f"💡 **Suggested target variables:** {', '.join(target_candidates[:3])}")
        
        # Problem type detection
        if categorical_columns:
            st.info("🔄 **Detected mixed data types** - Consider encoding categorical variables if used as features")
        
        # Data quality checks
        missing_pct = df.isnull().sum().sum() / (n_samples * n_features) * 100
        if missing_pct > 10:
            st.warning(f"⚠️ **High missing data** ({missing_pct:.1f}%) - Consider imputation")
        elif missing_pct > 0:
            st.info(f"ℹ️ **Some missing data** ({missing_pct:.1f}%) - Models will handle automatically")
    
    # Model Configuration
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        task = st.radio("🎯 Machine Learning Task", ["Regression", "Classification"], key="ml-task")

    target_options = numeric_columns if task == "Regression" else numeric_columns + categorical_columns
    target_options_display = get_columns_with_dtype(df, target_options)
    with col2:
        target_label = st.selectbox("🎯 Select Target Variable (What to predict)", target_options_display, key="ml-target")
    
    target_col = extract_column_name(target_label)

    features = [col for col in numeric_columns if col != target_col]
    
    if not features:
        st.warning("You must have at least one numeric feature column apart from the target.")
        return df
    
    # Advanced Feature Selection
    with st.expander("🔍 Feature Selection & Engineering", expanded=False):
        st.markdown("### Select & Transform Features")
        
        # Correlation-based suggestions
        if len(features) > 1:
            try:
                temp_df = df[features + [target_col]].copy()
                if temp_df[target_col].dtype == object or temp_df[target_col].dtype.name == 'category':
                    temp_df[target_col] = temp_df[target_col].astype('category').cat.codes
                    
                corr_with_target = temp_df.corr()[target_col].abs().sort_values(ascending=False)
                top_features = corr_with_target.head(6).index.tolist()
                if target_col in top_features:
                    top_features.remove(target_col)
                
                st.markdown(f"**Top correlated features with {target_col}:**")
                corr_df = pd.DataFrame({
                    'Feature': top_features[:5],
                    'Correlation': corr_with_target[top_features[:5]].round(3)
                })
                st.dataframe(corr_df, use_container_width=True)
            except Exception as e:
                st.info("Feature correlation could not be automatically calculated for this target variable.")
        
        selected_features = st.multiselect(
            "Select Feature Variables (Predictors)", 
            get_columns_with_dtype(df, features),
            default=get_columns_with_dtype(df, features[:min(5, len(features))]),
            key="ml-features"
        )
        selected_features = [extract_column_name(label) for label in selected_features]
        
        # Feature scaling option
        if selected_features:
            scale_features = st.checkbox("Apply Standard Scaling to features", value=True, key="scale-features")
    
    selected_features_raw = st.session_state.get('ml-features', get_columns_with_dtype(df, features[:min(5, len(features))]))
    # stored values may be labels like "col (int64)"; convert to actual column names
    if isinstance(selected_features_raw, list):
        selected_features = [extract_column_name(s) for s in selected_features_raw]
    else:
        selected_features = []

    if not selected_features:
        st.error("Please select at least one feature to train on.")
        return df
    
    # Model Selection with Intelligence
    with st.expander("🧠 Model Selection", expanded=True):
        if task == "Regression":
            models_info = {
                "Linear Regression": {"description": "Simple, interpretable, good baseline", "best_for": "Linear relationships"},
                "Decision Tree Regressor": {"description": "Handles non-linear data, interpretable", "best_for": "Complex patterns, small datasets"},
                "Random Forest Regressor": {"description": "Ensemble, robust, handles overfitting", "best_for": "Most regression tasks"},
                "Gradient Boosting Regressor": {"description": "Powerful, often best performance", "best_for": "Competitions, complex data"},
                "K-Nearest Neighbors Regressor": {"description": "Instance-based, no training", "best_for": "Small datasets"},
                "Support Vector Regressor": {"description": "Effective in high dimensions", "best_for": "High-dimensional data"},
                "Voting Regressor": {"description": "Ensemble averaging multiple strong models", "best_for": "Maximum performance"}
            }
            
            # Smart suggestions based on data size
            if n_samples < 1000:
                suggested_model = "Decision Tree Regressor"
                reason = "Small dataset - avoid overfitting with simpler models"
            elif n_samples > 10000:
                suggested_model = "Linear Regression"
                reason = "Large dataset - simple models are efficient and effective"
            else:
                suggested_model = "Random Forest Regressor"
                reason = "Medium dataset - ensemble methods provide good balance"
                
        else:  # Classification
            models_info = {
                "Logistic Regression": {"description": "Simple, interpretable, probabilistic", "best_for": "Linear boundaries"},
                "Decision Tree Classifier": {"description": "Handles non-linear data, interpretable", "best_for": "Complex patterns"},
                "Random Forest Classifier": {"description": "Ensemble, robust, handles overfitting", "best_for": "Most classification tasks"},
                "Gradient Boosting Classifier": {"description": "Powerful, often best performance", "best_for": "Competitions, complex data"},
                "K-Nearest Neighbors Classifier": {"description": "Instance-based, no assumptions", "best_for": "Small datasets"},
                "Support Vector Classifier": {"description": "Effective in high dimensions", "best_for": "High-dimensional data"},
                "Voting Classifier": {"description": "Ensemble voting across multiple models", "best_for": "Maximum performance"}
            }
            
            # Check if binary or multiclass
            unique_targets = df[target_col].nunique()
            if unique_targets == 2:
                suggested_model = "Logistic Regression"
                reason = "Binary classification - start with interpretable model"
            else:
                suggested_model = "Random Forest Classifier"
                reason = f"Multi-class ({unique_targets} classes) - robust ensemble method"
        
        # Display suggestion
        st.success(f"🎯 **Recommended Model:** {suggested_model}")
        st.info(f"💡 **Why:** {reason}")
        
        if task == "Regression":
            model_options = list(models_info.keys())
        else:
            model_options = list(models_info.keys())
        
        model_name = st.selectbox("Choose Algorithm", model_options, 
                                index=model_options.index(suggested_model), key="ml-algo")
        
        # Show model info
        if model_name in models_info:
            info = models_info[model_name]
            st.markdown(f"**{model_name}**: {info['description']}")
            st.markdown(f"**Best for:** {info['best_for']}")
    
    # Training Configuration
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5, key="ml-test-size")
        with col2:
            random_state = st.number_input("Random State", value=42, min_value=0, key="random-state")
        with col3:
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, key="cv-folds")

    with st.expander("🧪 Hyperparameter Tuning", expanded=False):
        tuning_model_name = st.selectbox(
            "Model to tune", 
            model_options, 
            index=model_options.index(model_name) if model_name in model_options else 0, 
            key="hpo-model"
        )
        search_method = st.radio("Search method", ["Grid Search", "Random Search"], key="hpo-method")
        if search_method == "Random Search":
            n_iter = st.slider("Iterations", 10, 80, 25, key="hpo-iterations")
        else:
            n_iter = None
        if st.button("Run hyperparameter search", key="hpo-run"):
            try:
                tune_X = df[selected_features].dropna()
                tune_y = df.loc[tune_X.index, target_col].dropna()
                common_idx = tune_X.index.intersection(tune_y.index)
                tune_X = tune_X.loc[common_idx]
                tune_y = tune_y.loc[common_idx]
                if len(tune_X) < 10:
                    st.error("Not enough rows to run hyperparameter search. Please prepare more data.")
                else:
                    search_model = create_model_by_name(tuning_model_name)
                    param_grid = get_search_space(tuning_model_name)
                    if not param_grid:
                        st.warning("No hyperparameter search grid available for this model.")
                    else:
                        if st.session_state.get('scale-features', True):
                            scale = StandardScaler()
                            tune_X = pd.DataFrame(scale.fit_transform(tune_X), columns=tune_X.columns, index=tune_X.index)
                        search = RandomizedSearchCV(search_model, param_grid, n_iter=n_iter, cv=cv_folds, scoring='r2' if task == 'Regression' else 'accuracy', random_state=random_state, n_jobs=-1) if search_method == "Random Search" else GridSearchCV(search_model, param_grid, cv=cv_folds, scoring='r2' if task == 'Regression' else 'accuracy', n_jobs=-1)
                        with st.spinner("Running hyperparameter search..."):
                            search.fit(tune_X, tune_y)
                        best_model = search.best_estimator_
                        st.toast("🔍 Hyperparameter search complete!")
                        st.write("**Best parameters:**")
                        st.write(search.best_params_)
                        st.session_state['trained_model'] = best_model
                        st.session_state['best_hyperparameters'] = search.best_params_
                        st.session_state['best_hyperparameters_model'] = tuning_model_name
                        st.session_state['trained_scaler'] = scale if st.session_state.get('scale-features', True) else None
                        st.session_state['trained_features'] = selected_features
                        st.session_state['trained_target'] = target_col
                        st.session_state['trained_task'] = task
                        st.session_state['trained_model_name'] = tuning_model_name
            except Exception as exc:
                st.error(f"Hyperparameter tuning failed: {exc}")
    
    # Train Model Button
    if st.button("🚀 Train Model", key="ml-train", type="primary"):
        if not selected_features:
            st.error("Please select at least one feature to train on.")
            return df
            
        # Prepare data
        X = df[selected_features].dropna()
        y = df.loc[X.index, target_col].dropna()
        
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) < 10:
            st.error("Not enough data remaining after dropping missing values. Please prepare your data first.")
            return df
        
        # Apply scaling if selected
        scaler = None
        if st.session_state.get('scale-features', True):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=random_state)
            model = create_model_by_name(model_name)
            if model is None:
                st.error("Selected model is not available.")
                return df

            # Auto-apply tuned hyperparameters
            if st.session_state.get('best_hyperparameters_model') == model_name:
                try:
                    model.set_params(**st.session_state['best_hyperparameters'])
                    st.info(f"✨ Automatically applied tuned hyperparameters for {model_name}!")
                except Exception:
                    pass

            with st.spinner("Training model..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_train = model.predict(X_train)

            # Persist full training run state
            st.session_state['trained_model'] = model
            st.session_state['trained_scaler'] = scaler
            st.session_state['trained_features'] = selected_features
            st.session_state['trained_target'] = target_col
            st.session_state['trained_task'] = task
            st.session_state['trained_model_name'] = model_name
            st.session_state['modeling_run'] = {
                'model': model, 'X': X, 'y': y,
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'y_pred': y_pred, 'y_pred_train': y_pred_train,
                'selected_features': selected_features,
                'target_col': target_col, 'task': task,
                'model_name': model_name, 'cv_folds': cv_folds
            }
        except Exception as e:
            st.error(f"Error during training: {e}")

    if 'modeling_run' in st.session_state:
        run = st.session_state['modeling_run']
        model = run['model']
        X = run['X']
        y = run['y']
        X_train = run['X_train']
        X_test = run['X_test']
        y_train = run['y_train']
        y_test = run['y_test']
        y_pred = run['y_pred']
        y_pred_train = run['y_pred_train']
        selected_features = run['selected_features']
        target_col = run['target_col']
        task = run['task']
        model_name = run['model_name']
        cv_folds = run['cv_folds']
        
        st.toast("🎯 Training complete!")
        st.markdown("### 📊 Performance Metrics")
        
        if task == "Regression":
            # Training metrics
            r2_train = r2_score(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            
            # Test metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card'><h3>R² Score</h3><p class='highlight'>{r2:.3f}</p></div>", unsafe_allow_html=True)
                st.caption(f"Train: {r2_train:.3f}")
            with col2:
                st.markdown(f"<div class='metric-card'><h3>RMSE</h3><p class='highlight'>{rmse:.3f}</p></div>", unsafe_allow_html=True)
                st.caption(f"Train: {rmse_train:.3f}")
            with col3:
                st.markdown(f"<div class='metric-card'><h3>MAE</h3><p class='highlight'>{mae:.3f}</p></div>", unsafe_allow_html=True)
            
            # Cross-validation results
            st.markdown("#### 🔄 Cross-Validation Results")
            cv_col1, cv_col2, cv_col3 = st.columns(3)
            with cv_col1:
                st.metric("CV Mean R²", f"{cv_scores.mean():.3f}")
            with cv_col2:
                st.metric("CV Std R²", f"{cv_scores.std():.3f}")
            with cv_col3:
                st.metric("CV Folds", cv_folds)
            
            # Residuals plot
            st.markdown("### 📈 Residuals Analysis")
            residuals = y_test - y_pred
            fig = px.scatter(x=y_pred, y=residuals, title="Residuals vs Predicted",
                           labels={'x': 'Predicted Values', 'y': 'Residuals'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        else:  # Classification
            # Training metrics
            acc_train = accuracy_score(y_train, y_pred_train)
            
            # Test metrics
            acc = accuracy_score(y_test, y_pred)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card'><h3>Accuracy</h3><p class='highlight'>{acc:.3f}</p></div>", unsafe_allow_html=True)
                st.caption(f"Train: {acc_train:.3f}")
            with col2:
                st.markdown(f"<div class='metric-card'><h3>Precision</h3><p class='highlight'>{precision:.3f}</p></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><h3>Recall</h3><p class='highlight'>{recall:.3f}</p></div>", unsafe_allow_html=True)
            
            # F1 Score and CV
            f1_col, cv_col = st.columns(2)
            with f1_col:
                st.markdown(f"<div class='metric-card'><h3>F1 Score</h3><p class='highlight'>{f1:.3f}</p></div>", unsafe_allow_html=True)
            with cv_col:
                st.markdown(f"<div class='metric-card'><h3>CV Accuracy</h3><p class='highlight'>{cv_scores.mean():.3f}±{cv_scores.std():.3f}</p></div>", unsafe_allow_html=True)
            
            # Confusion Matrix
            st.markdown("### 🔢 Confusion Matrix")
            # Build a consistent label list that matches the confusion matrix dimensions.
            if hasattr(model, 'classes_'):
                labels = list(model.classes_)
            else:
                # Ensure we include all observed labels in test or predictions
                labels = list(np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)])))

            # Compute confusion matrix using explicit labels to guarantee matching shape
            try:
                cm = confusion_matrix(y_test, y_pred, labels=labels)
            except Exception:
                # Fallback to default behavior if something unexpected happens
                cm = confusion_matrix(y_test, y_pred)
                labels = list(range(cm.shape[0]))

            labels_str = [str(l) for l in labels]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                          title="Confusion Matrix Heatmap", x=labels_str, y=labels_str,
                          labels={"color":"Count", "x":"Predicted", "y":"Actual"})
            st.plotly_chart(fig, use_container_width=True)

            # ROC Curve for binary classification
            if len(labels) == 2 and hasattr(model, "predict_proba"):
                st.markdown("### 📈 ROC Curve")
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=labels[1])
                    roc_auc = auc(fpr, tpr)
                    fig_roc = px.area(
                        x=fpr, y=tpr,
                        title=f'ROC Curve (AUC={roc_auc:.3f})',
                        labels=dict(x='False Positive Rate', y='True Positive Rate'),
                        width=700, height=500
                    )
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate ROC curve: {e}")
        
        # Feature Importance (for applicable models)
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df.head(10), x='Importance', y='Feature', 
                       title="Top 10 Feature Importances", orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        # SHAP Explainability
        st.markdown("### 🧠 Explainable AI (SHAP)")
        with st.expander("Generate SHAP Feature Analysis (May be slow for large models)"):
            if st.button("Calculate SHAP Values", key="calc-shap"):
                with st.spinner("Calculating SHAP values..."):
                    try:
                        sample_size = min(100, len(X_train)) # Use small sample for speed
                        X_sample = X_train.sample(sample_size, random_state=42)
                        
                        try:
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_sample)
                        except:
                            explainer = shap.KernelExplainer(model.predict, shap.kmeans(X_train, 10))
                            shap_values = explainer.shap_values(X_sample)
                            
                        st.toast("🧠 SHAP calculation successful!")
                        
                        if isinstance(shap_values, list):
                            sv = shap_values
                        elif len(np.shape(shap_values)) == 3:
                            sv = [shap_values[:, :, i] for i in range(np.shape(shap_values)[2])]
                        else:
                            sv = shap_values
                            
                        fig_shap, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(sv, X_sample, show=False)
                        st.pyplot(fig_shap)
                    except Exception as e:
                        st.error(f"Could not calculate SHAP values for this model type: {e}")
                        
        # Partial Dependence Plots (PDP)
        st.markdown("### 📉 Partial Dependence Plots (PDP)")
        with st.expander("Generate Partial Dependence Plot"):
            pdp_feature = st.selectbox("Select feature to analyze", selected_features, key="pdp-feature")
            
            pdp_target = None
            if task == "Classification" and hasattr(model, 'classes_') and len(model.classes_) > 2:
                pdp_target = st.selectbox("Select target class to plot", model.classes_, key="pdp-target")
                
            if st.button("Generate PDP", key="generate-pdp"):
                try:
                    fig_pdp, ax = plt.subplots(figsize=(8, 5))
                    kwargs = {}
                    if pdp_target is not None:
                        kwargs["target"] = pdp_target
                    elif task == "Classification" and hasattr(model, 'classes_') and len(model.classes_) > 2:
                        kwargs["target"] = model.classes_[0]
                        
                    PartialDependenceDisplay.from_estimator(model, X_train, [pdp_feature], ax=ax, **kwargs)
                    fig_pdp.tight_layout()
                    st.pyplot(fig_pdp)
                except Exception as e:
                    st.error(f"Could not generate PDP: {e}")
        
        # Learning Curves (for iterative models)
        if hasattr(model, 'predict') and task == "Regression":
            st.markdown("### 📊 Learning Curves")
            from sklearn.model_selection import learning_curve
            
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv_folds, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='neg_mean_squared_error' if task == "Regression" else 'accuracy'
            )
            
            train_scores_mean = -train_scores.mean(axis=1) if task == "Regression" else train_scores.mean(axis=1)
            val_scores_mean = -val_scores.mean(axis=1) if task == "Regression" else val_scores.mean(axis=1)
            
            fig = px.line(title="Learning Curves")
            fig.add_scatter(x=train_sizes, y=train_scores_mean, name="Training Score", mode='lines+markers')
            fig.add_scatter(x=train_sizes, y=val_scores_mean, name="Validation Score", mode='lines+markers')
            fig.update_layout(xaxis_title="Training Set Size", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

        # Model Architecture Visualization
        st.markdown("### 🧩 Model Architecture Visualization")
        with st.expander("View internal model structure"):
            try:
                if model_name in ["Linear Regression", "Logistic Regression"]:
                    st.write("#### Coefficients (Weights)")
                    if hasattr(model, "coef_"):
                        coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                        coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': coefs})
                        coef_df['Absolute Weight'] = coef_df['Coefficient'].abs()
                        coef_df = coef_df.sort_values('Absolute Weight', ascending=False)
                        fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', 
                                          title=f"{model_name} Coefficients", color='Coefficient', color_continuous_scale="RdBu")
                        st.plotly_chart(fig_coef, use_container_width=True)
                    else:
                        st.warning("Model coefficients not available.")

                elif model_name in ["Decision Tree Regressor", "Decision Tree Classifier"]:
                    st.write("#### Decision Tree Plot")
                    from sklearn.tree import plot_tree
                    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                    class_names = [str(c) for c in model.classes_] if hasattr(model, 'classes_') else None
                    plot_tree(model, feature_names=selected_features, class_names=class_names, 
                              filled=True, max_depth=3, ax=ax_tree, fontsize=10)
                    st.pyplot(fig_tree)
                    st.caption("Note: Tree is truncated to max_depth=3 for visibility.")

                elif "Forest" in model_name or "Boosting" in model_name:
                    st.write(f"#### First Tree in {model_name}")
                    from sklearn.tree import plot_tree
                    if hasattr(model, "estimators_"):
                        if "Gradient" in model_name:
                            first_tree = model.estimators_[0, 0] if len(model.estimators_.shape) > 1 else model.estimators_[0]
                        else:
                            first_tree = model.estimators_[0]
                        fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                        class_names = [str(c) for c in model.classes_] if hasattr(model, 'classes_') else None
                        plot_tree(first_tree, feature_names=selected_features, class_names=class_names, 
                                  filled=True, max_depth=3, ax=ax_tree, fontsize=10)
                        st.pyplot(fig_tree)
                        st.caption("Note: This is only 1 out of hundreds of trees in the ensemble.")

                elif model_name in ["K-Nearest Neighbors Classifier", "K-Nearest Neighbors Regressor", "Support Vector Classifier", "Support Vector Regressor"]:
                    st.write("#### 2D Decision Boundary Map (via PCA)")
                    st.info("Because this model relies on distance or mathematical hyperplanes rather than rules, we compress the data into 2 dimensions to visualize its decision boundary.")
                    if st.button("Generate 2D Map (May take a moment)", key="generate-pca-map"):
                        with st.spinner("Compressing dimensions & plotting boundary..."):
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_train)
                            
                            dummy_model = create_model_by_name(model_name)
                            dummy_model.fit(X_pca, y_train)
                            
                            x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].min() + 1
                            y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].min() + 1
                            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                            
                            Z = dummy_model.predict(np.c_[xx.ravel(), yy.ravel()])
                            
                            if task == "Classification":
                                if Z.dtype == object or Z.dtype.name == 'category':
                                    unique_Z = np.unique(Z)
                                    mapping = {val: i for i, val in enumerate(unique_Z)}
                                    Z = np.array([mapping[val] for val in Z])
                            
                            Z = Z.reshape(xx.shape)
                            
                            fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
                            ax_pca.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")
                            
                            if task == "Classification":
                                scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.factorize(y_train)[0], edgecolor='k', cmap="viridis")
                            else:
                                scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolor='k', cmap="viridis")
                            
                            ax_pca.set_title("2D Decision Surface")
                            ax_pca.set_xlabel("Principal Component 1")
                            ax_pca.set_ylabel("Principal Component 2")
                            st.pyplot(fig_pca)

                elif "Voting" in model_name:
                    st.write("#### Ensemble Weights")
                    if hasattr(model, "weights") and model.weights is not None:
                        weights = model.weights
                        labels = [est[0] for est in model.estimators]
                        fig_vote = px.pie(values=weights, names=labels, title="Voting Classifier Weights")
                        st.plotly_chart(fig_vote)
                    else:
                        st.write("This Voting Ensemble uses equal weighting for all sub-models.")

            except Exception as e:
                st.error(f"Could not generate architecture visualization for {model_name}: {e}")
        
        # Model Persistence
        st.markdown("### 💾 Model Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Model", key="save-model"):
                import joblib
                import os
                os.makedirs("models", exist_ok=True)
                model_filename = f"models/{model_name.replace(' ', '_').lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                joblib.dump(model, model_filename)
                st.toast(f"💾 Model saved as: {model_filename}")
        
        with col2:
            uploaded_model = st.file_uploader("📁 Load Saved Model", type=['joblib'], key="load-model")
            if uploaded_model is not None:
                loaded_model = joblib.load(uploaded_model)
                st.session_state['trained_model'] = loaded_model
                st.session_state['trained_model_name'] = getattr(loaded_model, '__class__', type(loaded_model)).__name__
                st.toast("💾 Model loaded successfully!")
        

    return df


def main():
    set_page_style()

    st.markdown(
        "<div class='hero-banner'><div class='hero-heading'>Modern EDA Studio — Data magic with cinematic polish.</div>"
        "<p class='hero-subtitle'>Upload any CSV or Excel file, explore structure with instant insights, transform data elegantly, and build predictive models with a premium studio experience.</p>"
        "<div class='hero-pill-row'>"
        "<span class='hero-pill'>AI-friendly analytics</span>"
        "<span class='hero-pill'>Auto suggestions</span>"
        "<span class='hero-pill'>Prediction & tuning</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    file = st.file_uploader("Choose your dataset", type=["csv", "xlsx", "xls"])
    if file is None:
        st.info("Start by uploading a dataset to unlock the analysis workspace.")
        return

    if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != file.name:
        st.session_state.uploaded_filename = file.name
        loaded_df = load_data(file)
        if loaded_df is None:
            return
        st.session_state.df = loaded_df
        st.session_state.original_df = loaded_df.copy()

    df = st.session_state.df
    original_df = st.session_state.original_df

    sidebar = st.sidebar
    sidebar.header("Workspace Navigation")
    section = sidebar.radio(
        "Select view",
        ["Overview", "EDA", "Transform", "Visualisations", "Clustering", "Modeling", "AutoML", "Prediction", "Export"],
        index=0,
    )

    if section == "Overview":
        summary_cards(df)
        render_insights(df)

        with st.container():
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Dataset snapshot")
            info_cols = st.columns(2)
            with info_cols[0]:
                st.markdown("**Column types**")
                st.dataframe(df.dtypes.astype(str).to_frame("Type"))
            with info_cols[1]:
                st.markdown("**First 10 rows**")
                st.dataframe(df.head(10))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Quick column summary")
            summary = df.agg(["nunique", "count", "size"]).T
            summary["missing"] = df.isna().sum()
            summary.columns = ["unique", "non-null", "total", "missing"]
            st.dataframe(summary)
            st.markdown("</div>", unsafe_allow_html=True)

    elif section == "EDA":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Missing values & correlation")
        plot_missing(df)
        plot_correlation(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Distribution explorer")
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_columns:
            dist_col = st.selectbox("Choose a numeric column", numeric_columns, key="dist-col")
            plot_distribution(df, dist_col)
            st.markdown("---")
            st.subheader("Pairwise relationships")
            selected = st.multiselect(
                "Select numeric columns for pairwise plot",
                numeric_columns,
                default=numeric_columns[:3],
                key="pairwise-cols",
            )
            if selected:
                plot_pairwise(df, selected)
        else:
            st.warning("No numeric columns available for distribution analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Advanced EDA")
        df = detect_outliers(df)
        imbalance_analysis(df)
        statistical_tests_section(df)
        df = duplicate_detection(df)
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Transform":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Data cleaning & transformation")
        df = transform_missing(df)
        st.divider()
        df = transform_encoding(df)
        st.divider()
        df = transform_scaling(df)
        st.divider()
        df = transform_types(df)
        st.divider()
        df = transform_delete_columns(df)
        st.divider()
        df = transform_feature_engineering(df)
        st.divider()
        df = transform_pca(df)
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Visualisations":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Visualisation Explorer")
        visualization_section(df)
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Clustering":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Clustering segmentation")
        df = cluster_segmentation(df)
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Modeling":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Machine Learning Models")
        df = modeling_section(df)
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "AutoML":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("AutoML: One-Click Training")
        df = automl_section(df)
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Prediction":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Prediction & Scoring")
        df = predict_section(df)
        st.markdown("</div>", unsafe_allow_html=True)

    elif section == "Export":
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Review and export")
        st.write("Use this page to compare the original and transformed versions of your dataset before download.")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Original dataset**")
            st.dataframe(original_df.head())
        with cols[1]:
            st.markdown("**Current dataset**")
            st.dataframe(df.head())

        download_button(df)
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.df = df

    if section != "Export":
        with st.expander("Show dataset preview", expanded=False):
            st.dataframe(df.head(15))
        with st.expander("Show column types", expanded=False):
            st.dataframe(df.dtypes.astype(str).to_frame("type"))


if __name__ == "__main__":
    main()
