"""Generate exploratory data analysis charts for the churn dataset."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.churn_utils import load_data, FIGURE_DIR

sns.set_theme(style="whitegrid")


def save_plot(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    data, X, y = load_data()
    data = data.copy()
    data["ChurnLabel"] = y.map({0: "No", 1: "Yes"})
    data["ChurnBinary"] = y.astype(int)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=data, x="ChurnLabel", hue="ChurnLabel", legend=False, ax=ax)
    ax.set(title="Customer Churn Distribution", xlabel="Churn", ylabel="Customers")
    save_plot(fig, "churn_distribution.png")

    if "Contract" in data:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=data, x="Contract", y="ChurnBinary", ax=ax, errorbar=None)
        ax.set(title="Churn Rate by Contract Type", ylabel="Churn rate", xlabel="Contract")
        save_plot(fig, "churn_by_contract.png")

    if "tenure" in data:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=data, x="tenure", hue="ChurnLabel", bins=24, multiple="stack", ax=ax)
        ax.set(title="Tenure Distribution by Churn", xlabel="Tenure (months)")
        save_plot(fig, "tenure_by_churn.png")

    if "MonthlyCharges" in data:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=data, x="ChurnLabel", y="MonthlyCharges", hue="ChurnLabel", legend=False, ax=ax)
        ax.set(title="Monthly Charges by Churn", xlabel="Churn", ylabel="Monthly charges")
        save_plot(fig, "monthly_charges_by_churn.png")

    print(f"EDA complete. Charts saved in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
