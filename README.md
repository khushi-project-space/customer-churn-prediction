# Customer Churn Prediction

An end-to-end Machine Learning project that predicts customers who may stop using a subscription service. Built using Python, Pandas, Scikit-learn, Matplotlib, Seaborn, and Streamlit.

## Project Highlights

- Exploratory Data Analysis (EDA) and churn visualization.
- Data preprocessing, missing-value handling, one-hot encoding, and feature scaling.
- Machine learning models: Logistic Regression, Random Forest, and optional XGBoost.
- Model evaluation using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- Model comparison and ROC curve visualization.
- Interactive Streamlit dashboard for data exploration and customer churn prediction.
- Included demo dataset for reproducible experimentation.

## Dataset

The project includes a synthetic telecom-style dataset at `data/telco_churn.csv`. It is intended for demonstration and learning purposes and is not the original IBM Telco Customer Churn dataset.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- XGBoost (optional)
- Streamlit

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── train.py
├── eda.py
├── requirements.txt
├── data/
│   └── telco_churn.csv
├── models/
├── reports/
│   └── figures/
└── src/
    └── churn_utils.py
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/khushi-project-space/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Exploratory Data Analysis

```bash
python eda.py
```

### 5. Train and compare models

```bash
python train.py
```

### 6. Launch the Streamlit dashboard

```bash
streamlit run app.py
```

## Model Evaluation

The models are evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix. Recall is especially useful for understanding how many actual churn customers are identified.

## Limitations

- The included dataset is synthetic and intended for educational use.
- Model performance on synthetic data does not establish real-world performance.
- Real-world use requires suitable, representative, and appropriately licensed data.

## Future Improvements

- Deploy the Streamlit application online.
- Evaluate models on a real, appropriately licensed dataset.
- Add additional customer retention insights and visualizations.
