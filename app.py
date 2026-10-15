"""Streamlit interface for exploring data and predicting churn."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from joblib import load

from src.churn_utils import DATA_PATH, MODEL_DIR, load_data, train_and_compare

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Prediction")
st.caption("Explore churn patterns, compare classification models, and score a sample customer profile.")

@st.cache_data
def read_dataset(path):
    return pd.read_csv(path)

@st.cache_resource
def get_training_results():
    return train_and_compare(save=True)

data, X, y = load_data()
if data.empty:
    st.error("No valid customer rows were loaded. Check the CSV file and its Churn column.")
    st.stop()

data = data.copy()
data["ChurnLabel"] = y.map({0: "No", 1: "Yes"})
total_customers = len(data)
churn_rate = float(y.mean()) if len(y) else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{total_customers:,}")
c2.metric("Churn rate", f"{churn_rate:.1%}")
c3.metric("Churned", f"{int(y.sum()):,}")
c4.metric("Retained", f"{int((1-y).sum()):,}")

tab1, tab2, tab3 = st.tabs(["📊 EDA Dashboard", "🤖 Model Evaluation", "🔎 Predict a Customer"])

with tab1:
    st.subheader("Explore the dataset")
    with st.expander("Show data preview"):
        st.dataframe(data.head(25), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=data, x="ChurnLabel", hue="ChurnLabel", legend=False, ax=ax)
        ax.set(title="Churn Distribution", xlabel="Churn", ylabel="Customers")
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        category_options = [
            c for c in [
                "Contract",
                "InternetService",
                "PaymentMethod",
                "OnlineSecurity",
                "TechSupport",
                "PaperlessBilling",
            ]
            if c in data.columns
        ]

        if category_options:
            category = st.selectbox("Compare churn rate by", category_options)
            chart_data = pd.DataFrame({
                "category": data[category].fillna("Unknown").astype(str).to_numpy(),
                "churn_binary": y.to_numpy(),
            })
            rate_data = (
                chart_data.groupby("category")["churn_binary"]
                .mean()
                .sort_values(ascending=False)
            )

            if not rate_data.empty:
                fig, ax = plt.subplots(figsize=(7, 4))
                rate_data.plot(kind="bar", ax=ax)
                ax.set(
                    title=f"Churn Rate by {category}",
                    ylabel="Churn rate",
                    xlabel=category,
                    ylim=(0, 1),
                )
                ax.tick_params(axis="x", rotation=25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No valid category data available for this chart.")
        else:
            st.warning("No supported category columns found in the dataset.")

    if "tenure" in data.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(
            data=data.assign(ChurnLabel=y.map({0: "No", 1: "Yes"})),
            x="tenure",
            hue="ChurnLabel",
            bins=24,
            multiple="stack",
            ax=ax,
        )
        ax.set(title="Tenure and Churn", xlabel="Tenure (months)")
        st.pyplot(fig)
        plt.close(fig)

with tab2:
    st.subheader("Classification model comparison")
    st.write("Training uses a stratified 80/20 holdout split. Categorical fields are one-hot encoded and numeric fields are imputed/scaled inside each pipeline.")
    if st.button("Train / refresh models", type="primary"):
        st.cache_resource.clear()
        st.rerun()
    try:
        _, _, _, _, _, results, pipelines, best_name = get_training_results()
        rows = []
        for name, m in results.items():
            rows.append({"Model": name, "Accuracy": m["Accuracy"], "Recall": m["Recall"], "ROC-AUC": m["ROC-AUC"], "Precision": m["Precision"], "F1-score": m["F1-score"]})
        metrics_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        st.dataframe(metrics_df.style.format({c:"{:.3f}" for c in metrics_df.columns if c != "Model"}), use_container_width=True)
        st.success(f"Highest holdout ROC-AUC in this run: {best_name}")
        selected = st.selectbox("View confusion matrix for", list(results.keys()))
        cm = results[selected]["Confusion Matrix"]
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted No", "Predicted Yes"], yticklabels=["Actual No", "Actual Yes"], ax=ax)
        ax.set_title(f"{selected}: Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Accuracy = overall correct predictions; Recall = fraction of actual churners identified; ROC-AUC = ranking ability across thresholds. Metrics here are from synthetic demo data.")
    except Exception as exc:
        st.error(f"Could not train models: {exc}")

with tab3:
    st.subheader("Score an example customer")
    st.info("Choose feature values and the trained model estimates churn probability. This is a learning demo, not a real retention decision.")
    try:
        _, _, _, _, _, results, pipelines, best_name = get_training_results()
        schema_path = MODEL_DIR / "feature_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        user_values = {}
        with st.form("customer_form"):
            left, right = st.columns(2)
            for i, feature in enumerate(schema["features"]):
                container = left if i % 2 == 0 else right
                with container:
                    if feature in schema.get("categorical_values", {}):
                        options = schema["categorical_values"][feature]
                        default = options[0] if options else ""
                        user_values[feature] = st.selectbox(feature, options, index=options.index(default) if default in options else 0)
                    elif feature in schema.get("numeric_defaults", {}):
                        default = schema["numeric_defaults"][feature]
                        if feature.lower() in {"seniorcitizen"}:
                            user_values[feature] = st.selectbox(feature, [0, 1], index=int(default >= 0.5))
                        else:
                            user_values[feature] = st.number_input(feature, value=float(default))
                    else:
                        user_values[feature] = st.text_input(feature, value="")
            submitted = st.form_submit_button("Predict churn risk", type="primary")
        if submitted:
            sample = pd.DataFrame([user_values], columns=schema["features"])
            model = pipelines[best_name]
            probability = float(model.predict_proba(sample)[0, 1])
            label = "Likely to churn" if probability >= 0.5 else "Less likely to churn"
            st.metric("Estimated churn probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
            if probability >= 0.5:
                st.warning(label)
            else:
                st.success(label)
            st.caption("The 0.50 cutoff is a demo default. A real business should tune the threshold using retention costs and validation data.")
    except Exception as exc:
        st.error(f"Prediction form is unavailable until training succeeds: {exc}")

st.divider()
st.caption("Portfolio project • Python • pandas • scikit-learn • XGBoost (optional) • Streamlit")
