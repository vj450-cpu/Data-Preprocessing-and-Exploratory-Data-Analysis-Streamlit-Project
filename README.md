# 🚀 Modern EDA Studio

**Modern EDA Studio** is a comprehensive, production-ready Streamlit application for advanced data preprocessing, exploratory data analysis, machine learning modeling, and interactive visualization. Built with modern UI/UX principles and enterprise-grade features.

Upload any dataset (CSV, Excel) and instantly access a full suite of data science tools including automated EDA, feature engineering, ML model training with hyperparameter tuning, model explainability with SHAP, and real-time prediction simulators.

---

## ✨ Key Highlights

- **🧠 AutoML**: One-click model comparison across multiple algorithms
- **🔍 Advanced EDA**: Outlier detection, imbalance analysis, statistical testing
- **🛠️ Feature Engineering**: Arithmetic operations, polynomial features, binning, datetime extraction
- **🤖 ML Pipeline**: 14 models with hyperparameter tuning, cross-validation, and diagnostics
- **📊 Explainability**: SHAP values, partial dependence plots, feature importance
- **🎯 Prediction Engine**: Interactive what-if simulator and batch scoring
- **💅 Modern UI**: Glassmorphism design with smooth animations and responsive layout

---

## 📋 Complete Feature Overview

### 📊 Overview Dashboard
- **Dataset Summary Cards**: Total rows, columns, numeric/categorical counts, missing values
- **Column Type Inspection**: All columns with data types and missing value counts
- **Quick Insights**: Automatic detection of columns with highest missing values
- **Dataset Snapshot**: First 10 rows with column type information
- **Column Statistics**: Unique values, non-null counts, missing percentages per column

### 🔍 Exploratory Data Analysis (EDA)

**Missing Value Analysis:**
- Interactive bar chart of missing values by column
- Color-coded visualization with percentages

**Correlation Analysis:**
- Correlation heatmaps for numeric columns
- Interactive Red-Blue diverging color scheme

**Distribution & Relationships:**
- Histograms with marginal box plots (40 bins)
- Pairwise scatter matrices (up to 200 samples for performance)
- Individual column distribution explorer

**Advanced EDA:**
- **Outlier Detection** (IQR method): Interactive inspection, bounds calculation, box plot visualization, one-click removal
- **Data Imbalance Analysis**: Class distribution for categorical columns with percentage breakdown and bar charts
- **Duplicate Detection**: Count and preview duplicate rows with one-click removal
- **Statistical Testing**: Univariate summaries, Pearson correlation, ANOVA, Chi-square tests, contingency tables

### 🛠️ Data Transformation & Cleaning

**Missing Value Handling:**
- Drop rows with missing values
- Fill with mean, median, mode, or custom constants

**Categorical Encoding:**
- One-Hot encoding (with first category drop option)
- Ordinal encoding using sklearn OrdinalEncoder
- Label encoding using sklearn LabelEncoder

**Feature Scaling:**
- StandardScaler (z-score normalization)
- MinMaxScaler (0-1 range scaling)

**Type Conversion:**
- Convert columns to int, float, string, datetime, category
- Automatic error handling with "coerce" option

**Column Management:**
- Multi-select column deletion with dtype labels

### 🧠 Feature Engineering Toolkit

**Arithmetic Feature Creation:**
- Add, subtract, multiply, divide operations
- Custom naming for derived features
- Division by zero protection

**Polynomial Features:**
- Configurable degrees (2-3)
- Automatic feature expansion
- Bias term exclusion option

**Binning & Discretization:**
- Equal-width binning
- Quantile-based binning
- Custom or predefined bin labels (Low/Medium/High)
- Bin summary statistics

**Datetime Feature Extraction:**
- Extract year, month, day, hour, minute, weekday
- Automatic datetime parsing
- Multiple date parts in single operation

### 🔬 Dimensionality Reduction (PCA)

- **Component Selection**: Fixed number or target explained variance percentage
- **PCA Visualizations**: Scree plot, cumulative variance table, 2D scatter plot with optional color-coding
- **Summary Metrics**: Explained variance ratio, cumulative variance, total explained percentage

### 📈 Visualization Suite

**Chart Types:**
- Scatter plots (with optional categorical color-coding)
- Line charts for trend analysis
- Bar charts for comparisons
- Histograms (40 bins) with statistics overlay
- Box plots for distribution and outlier detection
- Pie charts for categorical proportions
- Countplots for categorical frequencies
- Correlation heatmaps (Red-Blue diverging scale)
- Pairwise scatter matrices for multi-dimensional relationships

### 🤖 Machine Learning Modeling

**Regression Models (7 algorithms):**
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regressor (SVR)
- Voting Regressor (ensemble of RF, GB, LR)

**Classification Models (7 algorithms):**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors Classifier
- Support Vector Classifier (SVC)
- Voting Classifier (ensemble of RF, GB, LR)

**Training Features:**
- Configurable train/test split (10-50%)
- Random state control for reproducibility
- Automatic feature scaling (StandardScaler)
- Cross-validation (3-10 folds)

### 🧪 Hyperparameter Tuning

- **Grid Search**: Exhaustive parameter optimization
- **Random Search**: Randomized sampling with configurable iterations
- **Model-Specific Grids**: Optimized parameter spaces for each algorithm
- **Cross-Validation Scoring**: R² for regression, accuracy for classification
- **Parallel Processing**: Multi-core optimization (n_jobs=-1)
- **Best Parameter Display**: Automatic application of optimal parameters

**Tunable Parameters by Model:**
- Random Forest: n_estimators, max_depth, min_samples_split/leaf
- Gradient Boosting: n_estimators, learning_rate, max_depth
- Decision Trees: max_depth, min_samples_split/leaf
- K-Neighbors: n_neighbors, weights, distance metric
- SVM: C parameter, gamma, kernel type
- Logistic Regression: C parameter, penalty, solver

### 📊 Model Evaluation & Diagnostics

**Regression Metrics:**
- R² Score (train and test sets)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Cross-validation R² with mean and standard deviation
- Residuals analysis plot

**Classification Metrics:**
- Accuracy (train and test sets)
- Precision, Recall, F1 Score (weighted averages)
- Confusion matrix with heatmap visualization
- ROC Curve with AUC for binary classification
- Cross-validation accuracy scores

**Advanced Diagnostics:**
- **Feature Importance**: Top 10 features with bar chart visualization
- **Learning Curves**: Training vs validation score progression
- **SHAP Values**: Model explainability with summary plots and waterfall charts
- **Partial Dependence Plots (PDP)**: Individual feature effect analysis
- **Decision Tree Visualization**: Full tree structure (truncated to depth 3)
- **Linear Model Coefficients**: Feature coefficient importance charts
- **Distance-Based Models**: 2D decision boundary maps via PCA
- **Ensemble Models**: Component model weights visualization

### 🏎️ AutoML (One-Click Training)

- Automatic model comparison across 4 algorithms
- Cross-validation scoring with progress tracking
- Automatic best model selection (R² for regression, accuracy for classification)
- Visual comparison bar chart
- Automatic model persistence in session state

### 🚀 Prediction & Scoring

**Interactive What-If Simulator:**
- Real-time prediction updates with slider adjustments
- Feature-specific min/max/mean slider values
- Live probability display for classification
- Prediction class/value display

**Batch Scoring:**
- CSV file upload for bulk predictions
- Automatic feature validation against trained model
- Data type conversion and preprocessing
- Scored dataset download as CSV

**FastAPI Code Generation:**
- Auto-generated Python FastAPI code snippet
- Ready-to-deploy REST API template
- Input schema generation from model features

**Model Management:**
- Save trained models as .joblib files
- Load pre-trained models from disk
- Automatic model naming with timestamps

### 🤖 Clustering & Segmentation

- **KMeans Clustering**: Configurable clusters (2-10)
- **Automatic Scaling**: Data preprocessing for clustering
- **Silhouette Score**: Cluster quality evaluation
- **2D Visualization**: Scatter plot with color-coded clusters
- **Cluster Distribution**: Summary statistics per cluster

### 💅 Modern UI/UX Features

**Design Elements:**
- Glassmorphism effects with backdrop blur
- Radial gradient backgrounds (blue and green accents)
- Hero banner with animated gradient text
- Smooth card hover animations and scaling
- Custom color scheme (indigo, emerald, sky blue palette)

**Interactive Components:**
- Toast notifications for user feedback (✅, ✨, 🎯, 🧠, 💾)
- Expandable sections for organized features
- Multi-column responsive layouts
- Color-coded metric cards with highlights
- Custom styled buttons with gradients

**Typography & Layout:**
- "Outfit" font family (Google Fonts)
- Responsive grid layouts
- Section cards with consistent styling
- Info banners for guidance and warnings
- Clear visual hierarchy

### 📁 Data Management

- **File Support**: CSV and Excel (.xlsx, .xls) uploads
- **Data Preservation**: Original vs transformed dataset comparison
- **Export Functionality**: Download processed datasets as CSV
- **Session State Management**: Persistent data across page navigation

### 🔧 Smart Features & Intelligence

- **Auto Column Labeling**: Shows dtype and missing count alongside column names
- **Smart Model Recommendations**: Based on dataset size and characteristics
- **Suggested Target Variables**: Automatic detection for modeling
- **Data Quality Warnings**: Alerts for potential issues
- **Intelligent Feature Selection**: Correlation-based feature ranking with top 5 display
- **Error Handling**: Graceful error messages and recovery options

---

## 🖥️ Tech Stack

**Core Framework:**
- **Streamlit** 1.56.0 - Web application framework

**Data Processing:**
- **Pandas** 3.0.2 - Data manipulation and analysis
- **NumPy** 1.26.4 - Numerical computing

**Machine Learning:**
- **Scikit-learn** 1.8.0 - ML algorithms, preprocessing, and evaluation
- **SHAP** 0.45.0 - Model explainability and interpretability

**Visualization:**
- **Plotly** 6.7.0 - Interactive charts and dashboards
- **Seaborn** 0.13.2 - Statistical data visualization
- **Matplotlib** 3.10.8 - Static plotting and customization

**Statistics & Testing:**
- **SciPy** 1.17.1 - Statistical functions and hypothesis testing

**UI Enhancements:**
- **Streamlit-Aggrid** - Advanced data table components
- **Streamlit-Lottie** - Animated background elements

---

## 📂 Project Structure

```
├── streamlit_app.py       # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── scratch/
    ├── ui_upgrades.py    # UI enhancement utilities
    └── update_script.py  # Update and maintenance scripts
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/modern-eda-studio.git
cd modern-eda-studio
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import streamlit, pandas, sklearn, shap; print('All dependencies installed successfully!')"
```

---

## ▶️ Usage

### Quick Start

1. **Launch the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the App:**
   - Open your browser to `http://localhost:8501`

3. **Upload Data:**
   - Use the file uploader to load CSV or Excel files
   - Supported formats: `.csv`, `.xlsx`, `.xls`

### Workflow Example

1. **Data Upload** → Upload your dataset
2. **Overview** → Review dataset summary and column types
3. **EDA** → Explore distributions, correlations, outliers
4. **Preprocessing** → Clean data, handle missing values, encode categories
5. **Feature Engineering** → Create new features, reduce dimensions
6. **Visualization** → Create interactive charts and plots
7. **Modeling** → Train ML models with hyperparameter tuning
8. **Evaluation** → Analyze model performance and explainability
9. **Prediction** → Use interactive simulator or batch scoring

### Advanced Features

- **AutoML**: Click "AutoML" for automatic model comparison
- **Hyperparameter Tuning**: Use Grid Search or Random Search for optimization
- **SHAP Analysis**: Understand model decisions with explainability plots
- **What-If Simulator**: Adjust feature values to see prediction changes in real-time

---

## 📸 Screenshots

*Add screenshots of the application here*

---

## 🎯 Use Cases

- **Data Analysts**: Rapid dataset exploration and preprocessing
- **Data Scientists**: End-to-end ML pipeline from data to deployment
- **Students**: Learning EDA, feature engineering, and ML concepts
- **Business Analysts**: Interactive data visualization and insights
- **ML Engineers**: Model prototyping, tuning, and explainability
- **Researchers**: Statistical analysis and hypothesis testing

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `python -m pytest`
6. Commit changes: `git commit -am 'Add your feature'`
7. Push to branch: `git push origin feature/your-feature-name`
8. Open a Pull Request

### Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update tests for new features
- Update this README for new features
- Ensure backward compatibility

### Areas for Contribution

- Additional ML algorithms
- New visualization types
- Performance optimizations
- UI/UX improvements
- Documentation enhancements
- Bug fixes and testing

---


## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Scikit-learn](https://scikit-learn.org/)
- Visualization by [Plotly](https://plotly.com/)
- Explainability by [SHAP](https://shap.readthedocs.io/)

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/modern-eda-studio/issues) page
2. Create a new issue with detailed description
3. Include your Python version, OS, and error messages

---

*Last updated: April 2026*


## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

