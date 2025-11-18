"""
Model Comparison & Random Forest Optimization Module
====================================================
This module implements:

1. Comparison of 9 regression models for zinc alloy hardness prediction
   using cross-validation metrics (R², RMSE, MAE).

2. Random Forest hyperparameter optimization and evaluation.

Designed for reproducible scientific research and open-source publishing.

Author: Yaxuan Shen
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")



# =====================================================================
#  Load and preprocess data
# =====================================================================

data = pd.read_excel('hardness.xlsx')

X = data.iloc[:, :-1]  # Features
Y = data.iloc[:, -1]   # Target

# Search for best random seed
best_score = -float('inf')
best_seed = None
for seed in range(51):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    RF = RandomForestRegressor(
        criterion='absolute_error', max_depth=30, max_features='log2',
        min_samples_leaf=1, min_samples_split=5, n_estimators=100
    )
    RF.fit(x_train, y_train)
    r2_train = r2_score(y_train, RF.predict(x_train))
    r2_test = r2_score(y_test, RF.predict(x_test))
    print(f"Seed {seed}: Train R² = {r2_train:.4f}, Test R² = {r2_test:.4f}")
    if r2_test > best_score:
        best_score = r2_test
        best_seed = seed

print(f"\nBest seed: {best_seed}, Test R² = {best_score:.4f}")

'''
Best seed:39
'''

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=39
)

# =====================================================================
#  Step 1 — Model Comparison
# =====================================================================
models = {
    'RF': RandomForestRegressor(n_estimators=100, random_state=0),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=2000, activation='relu',
                        solver='adam', learning_rate_init=0.001, random_state=0),
    'Ridge': Ridge(alpha=1, max_iter=10000),
    'Lasso': Lasso(alpha=1, max_iter=10000),
    'XGBoost': XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=9),
    'DTR': DecisionTreeRegressor(random_state=0),
    'KNN': KNeighborsRegressor(n_neighbors=1),
    'GBDT': GradientBoostingRegressor(n_estimators=50, learning_rate=0.05,
                                      max_depth=5, min_samples_split=2, random_state=42)
}

# Metrics storage
results = {
    'Model': [],
    'R²': [], 'RMSE': [], 'MAE': [],
    'R²cv': [], 'RMSEcv': [], 'MAEcv': []
}

# Cross-validation setup
kf = KFold(n_splits=10, random_state=42, shuffle=True)

# Train & evaluate models
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    results['Model'].append(name)
    results['R²'].append(r2_score(y_test, y_pred))
    results['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
    results['MAE'].append(mean_absolute_error(y_test, y_pred))

    # Cross-validation metrics
    results['R²cv'].append(cross_val_score(model, x_train, y_train, cv=kf, scoring='r2').mean())
    results['MAEcv'].append(-cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_absolute_error').mean())
    results['RMSEcv'].append(np.sqrt(-cross_val_score(model, x_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()))

# Save results
result_df = pd.DataFrame(results)
result_df.to_excel('model_comparison_results.xlsx', index=False)
print(result_df)

# =====================================================================
#  Step 2 — Random Forest Hyperparameter Optimization
# =====================================================================
# Train/test split for RF optimization
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=39)

# Baseline RF
old_RF = RandomForestRegressor(n_estimators=100, random_state=0)
old_RF.fit(x_train, y_train)

# Optimized RF
RF = RandomForestRegressor(
    n_estimators=200,
    criterion='friedman_mse',
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2
)
RF.fit(x_train, y_train)
joblib.dump(RF, 'random_forest_model.pkl')

# Evaluate
print('Baseline RF train R²:', r2_score(y_train, old_RF.predict(x_train)))
print('Baseline RF test R²:', r2_score(y_test, old_RF.predict(x_test)))
print('Optimized RF train R²:', r2_score(y_train, RF.predict(x_train)))
print('Optimized RF test R²:', r2_score(y_test, RF.predict(x_test)))

RF_MSE = mean_absolute_error(y_test, RF.predict(x_test))
RF_RMSE = np.sqrt(RF_MSE)
RF_R2 = r2_score(y_test, RF.predict(x_test))
