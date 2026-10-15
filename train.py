"""Train churn classifiers and print a model-comparison report."""
import json
import pandas as pd
from src.churn_utils import train_and_compare, MODEL_DIR


def main():
    data, X_train, X_test, y_train, y_test, results, pipelines, best_name = train_and_compare()
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Recall": metrics["Recall"],
            "ROC-AUC": metrics["ROC-AUC"],
            "Precision": metrics["Precision"],
            "F1-score": metrics["F1-score"],
        })
    comparison = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    print("\nCUSTOMER CHURN MODEL COMPARISON")
    print(comparison.to_string(index=False, formatters={
        c: (lambda x: f"{x:.3f}") for c in comparison.columns if c != "Model"
    }))
    print(f"\nSelected model by holdout ROC-AUC: {best_name}")
    print(f"Saved pipeline and metrics to: {MODEL_DIR}")
    print("\nNote: scores are for the included synthetic demo data, not real customer performance.")


if __name__ == "__main__":
    main()
