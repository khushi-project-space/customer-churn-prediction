# Customer Churn Prediction

An end-to-end machine-learning project that identifies customers who may stop using a subscription service. Built with Python, pandas, scikit-learn, optional XGBoost, Matplotlib/Seaborn, and Streamlit.

## Project highlights
- Exploratory data analysis (EDA) and churn-pattern visualizations
- Data preprocessing with missing-value handling, one-hot encoding, and scaling
- Logistic Regression, Random Forest, and XGBoost (when installed)
- Evaluation with **Accuracy, Recall, ROC-AUC**, plus precision, F1-score, and confusion matrix
- Model comparison table and ROC curves
- Streamlit dashboard for exploring the data and testing a customer profile
- Reproducible demo dataset included in `data/telco_churn.csv`

> **Dataset note:** The bundled CSV is a synthetic, realistic-shaped telecom-style dataset generated for this portfolio project. It is not the original IBM Telco Customer Churn dataset and should not be presented as real customer data. Replace it with a properly licensed dataset for real-world analysis.

## Folder structure
```text
customer-churn-prediction/
├── app.py                  # Interactive Streamlit app
├── train.py                # Train and compare classification models
├── eda.py                  # Create EDA charts
├── requirements.txt
├── data/telco_churn.csv    # Synthetic demo dataset
├── models/                 # Saved model pipeline + comparison results (created by train.py)
├── reports/figures/        # Generated charts
└── src/churn_utils.py      # Shared preprocessing/training utilities
```

## 1. Run locally
Python 3.10–3.12 recommended.

```bash
# Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux activation: source .venv/bin/activate
pip install -r requirements.txt
python eda.py
python train.py
streamlit run app.py
```

The first training run writes model pipelines and metrics under `models/`, and charts under `reports/figures/`. Logistic Regression and Random Forest run by default. To include XGBoost as well, install the listed dependency and set `CHURN_INCLUDE_XGBOOST=1` before running `python train.py` (Windows CMD: `set CHURN_INCLUDE_XGBOOST=1`).

## 2. Use another dataset
Replace `data/telco_churn.csv` with a CSV containing a target column named `Churn` (`Yes`/`No` or `1`/`0`). The app and training script automatically exclude `customerID` as an identifier. Keep the target and feature names consistent, or update `TARGET` in `src/churn_utils.py`.

## 3. GitHub upload
1. Extract the ZIP.
2. Open the extracted project folder.
3. Create a new GitHub repository named `customer-churn-prediction`.
4. Upload the project files, or run:
   ```bash
   git init
   git add .
   git commit -m "Add customer churn prediction project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/customer-churn-prediction.git
   git push -u origin main
   ```
5. Add screenshots of the running Streamlit app to the README after running it locally.

## 4. Important interpretation
Churn recall measures the fraction of actual churners identified by the model. In retention work, recall can be important because missed churners may represent lost revenue. Accuracy alone can be misleading when classes are imbalanced. Choose a decision threshold based on business costs, and validate models on fresh, representative data before operational use.

## Limitations
- Synthetic data is for demonstration and learning only.
- Scores are not evidence of real-world business performance.
- No customer information is sent anywhere; the app runs locally unless deployed by the user.
