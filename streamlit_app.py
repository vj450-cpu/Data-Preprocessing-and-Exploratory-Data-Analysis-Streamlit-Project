import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency, f_oneway, pearsonr

import plotly.io as pio
pio.templates.default = "plotly_white"

sns.set_theme(style="whitegrid", palette="pastel")


def set_page_style():
    st.set_page_config(page_title="Modern EDA Studio", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f6f8fd 0%, #f1f6f9 50%, #f8fbfd 100%);
            background-attachment: fixed;
        }

        /* Gradient glowing titles */
        .header-title {
            font-size: 3.2rem;
            font-weight: 800;
            background: -webkit-linear-gradient(45deg, #4f46e5, #ec4899, #f59e0b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        .header-subtitle {
            font-size: 1.15rem;
            font-weight: 400;
            color: #64748b;
            margin-top: 0;
            margin-bottom: 2rem;
        }
        
        /* Glassmorphism Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.8);
            margin-bottom: 16px;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 14px 40px 0 rgba(79, 70, 229, 0.15);
            background: rgba(255, 255, 255, 0.8);
        }
        .metric-card h3 {
            margin: 0;
            font-size: 1rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card p.highlight {
            margin: 10px 0 0 0;
            font-size: 2.2rem;
            font-weight: 800;
            color: #1e293b;
            background: -webkit-linear-gradient(45deg, #1e293b, #4338ca);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Glassmorphism Section Cards */
        .section-card {
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 24px;
            padding: 32px;
            border: 1px solid rgba(255, 255, 255, 0.6);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.04);
            margin-bottom: 28px;
            transition: box-shadow 0.3s ease;
        }
        .section-card:hover {
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        }

        /* Customizing Streamlit components */
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: white;
            font-weight: 600;
            border-radius: 12px;
            border: none;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.23);
            background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%);
            color: white;
            border-color: transparent;
        }

        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
            border-radius: 12px;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.23);
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            color: white;
            border-color: transparent;
        }

        /* Streamlit Sidebar modern styling */
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.5) !important;
            backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.5);
        }
        
        /* Selectbox and Input focus modernizing */
        .stSelectbox div[data-baseweb="select"] > div, 
        .stTextInput div[data-baseweb="input"] > div,
        .stNumberInput div[data-baseweb="input"] > div {
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
            background-color: #f8fafc !important;
            transition: all 0.2s ease !important;
        }
        .stSelectbox div[data-baseweb="select"] > div:hover,
        .stTextInput div[data-baseweb="input"] > div:hover,
        .stNumberInput div[data-baseweb="input"] > div:hover {
            border-color: #a5b4fc !important;
        }
        .stSelectbox div[data-baseweb="select"] > div:focus-within,
        .stTextInput div[data-baseweb="input"] > div:focus-within,
        .stNumberInput div[data-baseweb="input"] > div:focus-within {
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
            border-color: #6366f1 !important;
        }
        
        /* Expander restyling */
        .streamlit-expanderHeader {
            font-family: 'Outfit', sans-serif;
            font-weight: 600 !important;
            border-radius: 12px !important;
        }
        
        /* DataFrame modernizing */
        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
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
                    col = st.selectbox("Numeric column", numeric_columns, key="uni-num-col")
                    st.write(df[col].describe())
                    fig = px.histogram(df, x=col, nbins=40, title=f"Distribution for {col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                if not categorical_columns:
                    st.warning("No categorical columns available.")
                else:
                    col = st.selectbox("Categorical column", categorical_columns, key="uni-cat-col")
                    counts = df[col].value_counts().reset_index()
                    counts.columns = [col, "count"]
                    st.dataframe(counts)
                    fig = px.bar(counts, x=col, y="count", title=f"Frequency of {col}")
                    st.plotly_chart(fig, use_container_width=True)

        elif test_type == "Numeric vs Numeric":
            if len(numeric_columns) < 2:
                st.warning("At least two numeric columns required.")
            else:
                x_col = st.selectbox("X numeric column", numeric_columns, key="bi-x-col")
                y_col = st.selectbox("Y numeric column", [c for c in numeric_columns if c != x_col], key="bi-y-col")
                data = df[[x_col, y_col]].dropna()
                corr, pvalue = pearsonr(data[x_col], data[y_col])
                st.write(f"Pearson correlation between {x_col} and {y_col}: {corr:.3f} (p={pvalue:.3f})")
                fig = px.scatter(data, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif test_type == "Categorical vs Numeric":
            if not categorical_columns or not numeric_columns:
                st.warning("Need at least one categorical and one numeric column.")
            else:
                cat_col = st.selectbox("Categorical column", categorical_columns, key="cat-num-cat")
                num_col = st.selectbox("Numeric column", numeric_columns, key="cat-num-num")
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
                cat1 = st.selectbox("First categorical column", categorical_columns, key="cat-cat-1")
                cat2 = st.selectbox("Second categorical column", [c for c in categorical_columns if c != cat1], key="cat-cat-2")
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
    col = st.selectbox("Missing value column", df.columns, key="missing-col")
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
        col = st.selectbox("Categorical column", cat_cols, key="enc-col")
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
        col = st.selectbox("Numeric column", num_cols, key="scale-col")
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
        col = st.selectbox("Categorical column", categorical_columns, key="viz-count-col")
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", color=col, title=f"Countplot of {col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Pie":
        if not categorical_columns:
            st.warning("No categorical columns available for pie charts.")
            return
        col = st.selectbox("Categorical column", categorical_columns, key="viz-pie-col")
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.pie(counts, names=col, values="count", title=f"Pie chart of {col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type in ["Scatter", "Line", "Bar"]:
        if len(numeric_columns) < 2:
            st.warning("At least two numeric columns are needed for this chart.")
            return
        x_col = st.selectbox("X axis", numeric_columns, key="viz-x")
        y_col = st.selectbox("Y axis", [c for c in numeric_columns if c != x_col], key="viz-y")
        if chart_type == "Scatter":
            color_col = None
            if categorical_columns:
                color_col = st.selectbox("Color by (optional)", ["None"] + categorical_columns, key="viz-color")
                color_arg = None if color_col == "None" else color_col
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
        col = st.selectbox("Numeric column", numeric_columns, key="viz-hist-col")
        fig = px.histogram(df, x=col, nbins=40, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Box":
        if not numeric_columns:
            st.warning("No numeric columns available for box plots.")
            return
        col = st.selectbox("Numeric column", numeric_columns, key="viz-box-col")
        fig = px.box(df, y=col, title=f"Box plot of {col}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if chart_type == "Cluster segmentation":
        cluster_segmentation(df)
        return


def cluster_segmentation(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    with st.expander("Cluster segmentation", expanded=False):
        if len(numeric_columns) < 2:
            st.warning("Select at least two numeric columns for clustering.")
            return df

        selected = st.multiselect("Numeric columns", numeric_columns, default=numeric_columns[:2], key="cluster-cols")
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
        col = st.selectbox("Column to convert", cols, key="type-col")
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
        delete_cols = st.multiselect("Select columns to remove", cols, key="delete-cols")
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
                if st.button("Create bins", key="bin-apply"):
                    try:
                        if method == "Quantile":
                            df[new_name] = pd.qcut(df[col], q=bins, duplicates="drop")
                        else:
                            df[new_name] = pd.cut(df[col], bins=bins)
                        st.success(f"Created binned column '{new_name}'.")
                        st.dataframe(df[[new_name]].head())
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


def main():
    set_page_style()

    st.markdown("<div class='header-title'>Modern EDA Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='header-subtitle'>Upload any CSV or Excel file, explore structure, clean data, and export a polished dataset with smart visuals.</p>",
        unsafe_allow_html=True,
    )

    file = st.file_uploader("Choose your dataset", type=["csv", "xlsx", "xls"])
    if file is None:
        st.info("Start by uploading a dataset to unlock the analysis workspace.")
        return

    df = load_data(file)
    if df is None:
        return

    sidebar = st.sidebar
    sidebar.header("Workspace Navigation")
    section = sidebar.radio(
        "Select view",
        ["Overview", "EDA", "Transform", "Visualisations", "Clustering", "Export"],
        index=0,
    )

    original_df = df.copy()

    if section == "Overview":
        summary_cards(df)
        render_insights(df)

        with st.container():
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Dataset snapshot")
            info_cols = st.columns(2)
            with info_cols[0]:
                st.markdown("**Column types**")
                st.dataframe(df.dtypes.to_frame("Type"))
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

    if section != "Export":
        with st.expander("Show dataset preview", expanded=False):
            st.dataframe(df.head(15))
        with st.expander("Show column types", expanded=False):
            st.dataframe(df.dtypes.to_frame("type"))


if __name__ == "__main__":
    main()
