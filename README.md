# 🚀 Modern EDA Studio

**Modern EDA Studio** is a powerful, all-in-one **Streamlit application** designed for seamless data exploration, preprocessing, and visualization.

Upload any dataset and instantly perform:

* Exploratory Data Analysis (EDA)
* Data cleaning & transformation
* Feature engineering
* Statistical testing
* Interactive visualizations
* Clustering & segmentation

—all from a clean, modern UI.

---

## ✨ Features

### 📊 Overview Dashboard

* Dataset summary (rows, columns, missing values)
* Column type inspection
* Quick insights generation

### 🔍 Exploratory Data Analysis (EDA)

* Missing value visualization
* Correlation heatmaps
* Distribution plots & pairwise analysis
* Outlier detection (IQR method)
* Duplicate detection
* Data imbalance analysis

### 🧪 Statistical Testing

* Pearson correlation (numeric vs numeric)
* ANOVA (categorical vs numeric)
* Chi-square test (categorical vs categorical)
* Univariate summaries

### 🛠 Data Transformation

* Missing value handling (mean, median, mode, constant)
* Encoding:

  * One-hot
  * Ordinal
  * Label encoding
* Feature scaling:

  * StandardScaler
  * MinMaxScaler
* Data type conversion
* Column deletion

### 🧠 Feature Engineering

* Arithmetic feature creation
* Polynomial features
* Binning / discretization
* Datetime feature extraction

### 🔬 Dimensionality Reduction

* PCA (Principal Component Analysis)
* Variance-based component selection
* PCA visualization

### 📈 Visualization Suite

* Scatter, Line, Bar charts
* Histograms & Box plots
* Pie charts & Countplots
* Correlation heatmaps
* Pairwise scatter matrix

### 🤖 Clustering

* KMeans clustering
* Cluster visualization
* Silhouette score evaluation

### 🎯 Machine Learning Modeling

* **Regression Models:**
  * Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * Gradient Boosting Regressor
  * K-Neighbors Regressor
  * Support Vector Regressor (SVR)
* **Classification Models:**
  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier
  * Gradient Boosting Classifier
  * K-Neighbors Classifier
  * Support Vector Classifier (SVC)
* Model training & evaluation
* Performance metrics (R², Accuracy, MSE, Confusion Matrix)
* Train-test split configuration

### 📦 Export

* Compare original vs transformed dataset
* Download processed dataset

---

## 🖥️ Tech Stack

* **Frontend/UI:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly, Seaborn, Matplotlib
* **Machine Learning:** Scikit-learn
* **Statistics:** SciPy

---

## 📂 Project Structure

```
├── app.py                # Main Streamlit application
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/modern-eda-studio.git
cd modern-eda-studio
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Supported File Formats

* CSV (.csv)
* Excel (.xlsx, .xls)

---

## 🎯 Use Cases

* Data analysts exploring datasets quickly
* Data scientists performing preprocessing
* Students learning EDA & feature engineering
* Rapid prototyping before model building

---


## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

