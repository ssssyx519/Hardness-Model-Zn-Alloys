"""
Feature Selection Module
========================
This module implements a two-step feature selection process
for zinc alloy hardness prediction:

Step 1: Pearson correlation heatmap + RF R²-based removal of highly correlated features.
        Certain features (e.g., 'Al') can be forced to keep.

Step 2: Mutual Information (MI) ranking and stepwise feature
        addition based on RMSE to determine optimal subset.

Designed for reproducible scientific research and open-source publishing.

Author: Yaxuan Shen
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
#  Step 1 — Pearson Correlation Heatmap & High Correlation Removal
# =====================================================================

def select_high_corr_features_rf(X: pd.DataFrame, y: pd.Series,
                                 threshold: float = 0.90,
                                 random_state: int = 10,
                                 keep_features: list = ['Al','Zn']):
    """
    Identify highly correlated feature pairs, keep one of each pair by RF R² comparison.
    Certain features can be forced to keep (e.g., 'Al').

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target values.
    threshold : float
        Correlation threshold to consider high correlation.
    random_state : int
        Random seed for reproducibility.
    keep_features : list
        Features to always keep and never remove.

    Returns
    -------
    corr_matrix : pd.DataFrame
        Pearson correlation matrix.
    remaining_features : list
        Features after removal.
    """
    if keep_features is None:
        keep_features = []

    corr_matrix = X.corr(method='pearson')
    remaining_features = list(X.columns)

    # Identify high-correlation pairs within threshold and upper_limit
    high_corr_pairs = [(corr_matrix.columns[i], corr_matrix.columns[j])
                       for i in range(len(corr_matrix.columns))
                       for j in range(i)
                       if threshold < abs(corr_matrix.iloc[i, j]) ]

    # Compare RF R² for each pair
    for f1, f2 in high_corr_pairs:
        # Skip feature removal if it must be kept
        if f1 in keep_features and f2 in keep_features:
            continue
        elif f1 in keep_features:
            if f2 in remaining_features:
                remaining_features.remove(f2)
            continue
        elif f2 in keep_features:
            if f1 in remaining_features:
                remaining_features.remove(f1)
            continue

        # RF R² comparison
        temp_features_1 = [f for f in remaining_features if f != f1]
        temp_features_2 = [f for f in remaining_features if f != f2]

        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        r2_1 = cross_val_score(rf, X[temp_features_1], y, cv=5, scoring='r2').mean()
        r2_2 = cross_val_score(rf, X[temp_features_2], y, cv=5, scoring='r2').mean()

        if r2_1 >= r2_2 and f2 in remaining_features:
            remaining_features.remove(f2)
        elif f1 in remaining_features:
            remaining_features.remove(f1)

    print(f"Removed features based on RF R²: {set(X.columns) - set(remaining_features)}")

    # Heatmap with numeric annotation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", square=True, linewidths=0.5)
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    return corr_matrix, remaining_features



# =====================================================================
#  Step 2 — Mutual Information + Stepwise RMSE Evaluation
# =====================================================================

def compute_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Compute mutual information (MI) between features and target.

    Returns
    -------
    pd.Series
        MI scores indexed by feature names (sorted descending).
    """
    mi = mutual_info_regression(X, y)
    mi_series = pd.Series(mi, index=X.columns)
    return mi_series.sort_values(ascending=False)


def plot_mutual_information(mi_series: pd.Series, figsize=(10, 6)):
    """Plot MI ranking as a bar chart."""
    plt.figure(figsize=figsize)
    sns.barplot(x=mi_series.index, y=mi_series.values)

    for i, score in enumerate(mi_series.values):
        plt.text(i, score + 0.005, f"{score:.3f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=90)
    plt.title("Mutual Information Scores")
    plt.xlabel("Feature")
    plt.ylabel("MI Score")
    plt.tight_layout()
    plt.show()


def rmse_metric(y_true, y_pred):
    """Compute RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def stepwise_feature_selection(X: pd.DataFrame,
                               y: pd.Series,
                               mi_series: pd.Series,
                               cv: int = 10,
                               random_state: int = 10) -> pd.DataFrame:
    """
    Stepwise add features following MI ranking, computing RMSE via cross-validation.

    Returns
    -------
    pd.DataFrame
        RMSE results for each added feature.
    """
    rmse_scorer = make_scorer(rmse_metric, greater_is_better=False)

    selected_features = []
    rmse_values = []
    feature_order = mi_series.index.tolist()

    for feat in feature_order:
        selected_features.append(feat)

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            min_samples_split=4
        )

        X_sub = X[selected_features]
        mean_rmse = -cross_val_score(model, X_sub, y, cv=cv, scoring=rmse_scorer).mean()
        rmse_values.append(mean_rmse)

        print(f"[Stepwise] Added: {feat:>6s} → RMSE = {mean_rmse:.4f}")

    result_df = pd.DataFrame({
        "num_features": range(1, len(selected_features) + 1),
        "added_feature": feature_order,
        "RMSE": rmse_values
    })
    return result_df


def plot_rmse_curve(result_df: pd.DataFrame, figsize=(8, 5)):
    """Plot RMSE vs number of selected features."""
    plt.figure(figsize=figsize)
    plt.plot(result_df["num_features"], result_df["RMSE"], marker="o")
    plt.xlabel("Number of Features")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Selected Features")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =====================================================================
#  Full Pipeline Wrapper
# =====================================================================

def run_feature_selection(data_path: str, target_col: str = None,
                          corr_threshold: float = 0.90,
                          keep_features: list = ['Al','Zn']):
    data = pd.read_excel(data_path)
    if target_col is None:
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
    else:
        X, y = data.drop(columns=[target_col]), data[target_col]

    print("\n=== Step 1: Pearson Correlation + RF R² Removal ===")
    corr_matrix, remaining_features = select_high_corr_features_rf(
        X, y, threshold=corr_threshold, keep_features=keep_features
    )

    print("\n=== Step 2: Mutual Information + Stepwise RMSE ===")
    X_filtered = X[remaining_features]
    mi_series = compute_mutual_information(X_filtered, y)
    plot_mutual_information(mi_series)
    stepwise_df = stepwise_feature_selection(X_filtered, y, mi_series)
    plot_rmse_curve(stepwise_df)

    return corr_matrix, remaining_features, mi_series, stepwise_df

# --------------------------- Run example ---------------------------
if __name__ == "__main__":
    data = pd.read_excel('all_data.xlsx')
    run_feature_selection(data)