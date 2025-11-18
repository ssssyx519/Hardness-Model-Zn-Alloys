import pandas as pd
import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.signal import savgol_filter  # For smoothing and derivative calculation
import matplotlib
import os
import joblib

print(f"SHAP version: {shap.__version__}")
print(f"XGBoost version: {xgboost.__version__}")

matplotlib.use('TkAgg')  # Set matplotlib backend


'''
Helper function: find the knee point of trend change
'''
def find_knee_point(x_data, y_data, window_length=5, polyorder=2):
    """
    Find the most significant trend change point (knee point) in a curve using curvature.
    Smooths y-data and computes the second derivative; the point with the largest abs(second derivative) is the knee.
    
    :param x_data: x coordinates (Pandas Series or NumPy array)
    :param y_data: y coordinates (NumPy array)
    :param window_length: Savitzky-Golay filter window length (must be odd)
    :param polyorder: polynomial order for Savitzky-Golay (must be < window_length)
    :return: x-coordinate of knee point
    """
    if len(x_data) < window_length:
        print(f"Warning: data points ({len(x_data)}) < window length ({window_length}). Returning median.")
        return np.median(x_data)

    if window_length % 2 == 0:
        window_length += 1

    if polyorder >= window_length:
        polyorder = window_length - 1
        if polyorder < 1:
            polyorder = 1

    y_second_deriv = savgol_filter(y_data, window_length, polyorder, deriv=2)
    knee_index = np.argmax(np.abs(y_second_deriv))
    sorted_x = np.array(x_data)[np.argsort(x_data)]
    return sorted_x[knee_index]


'''
Load data
'''
print("--> Loading local Excel data...")

df = pd.read_excel('hardness.xlsx')

print("Data preview:")
print(df.head())

target_column = 'HV'
feature_columns = [col for col in df.columns if col != target_column]
X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)


'''
Train XGBoost model and perform hyperparameter search
'''
model = joblib.load('random_forest_model.pkl')


'''
Compute SHAP values
'''
print("--> Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)


print("--> Plotting SHAP combination figure...")

# ==================== Aesthetic Parameters ====================
aesthetic_params = {
    'suptitle_size': 22,
    'ax_label_size': 16,
    'tick_label_size': 16,
    'legend_size': 14,
    'cbar_label_size': 12,
    'summary_cbar_width': 0.015,
    'summary_cbar_height_shrink': 1.0,
    'summary_cbar_pad': 0.01,
    'dep_cbar_width': 0.005,
    'dep_cbar_height_shrink': 1.0,
    'dep_cbar_pad': 0.002,
    'dep_cbar_tick_length': 1,
    'grid_wspace': 0.45,
    'grid_hspace': 0.4
}
# ============================================================

plt.rcParams['font.family'] = 'Times New Roman'  # Set global font

fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(3, 4, figure=fig,
                       wspace=aesthetic_params['grid_wspace'],
                       hspace=aesthetic_params['grid_hspace'])

# --- Summary plot (left) ---
ax_main = fig.add_subplot(gs[:, :2])
mean_abs_shaps = np.abs(shap_values.values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': mean_abs_shaps
}).sort_values('importance', ascending=True)

ax_main.set_yticks(range(len(feature_importance_df)))

ax_top = ax_main.twiny()
ax_top.barh(range(len(feature_importance_df)),
            feature_importance_df['importance'],
            color="lightgray", alpha=0.6, height=0.7)
ax_top.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
ax_top.grid(False)

cmap = plt.get_cmap("viridis")
for i, feature_name in enumerate(feature_importance_df['feature']):
    original_idx = X.columns.get_loc(feature_name)
    shap_vals_for_feature = shap_values.values[:, original_idx]
    feature_vals_for_color = X.iloc[:, original_idx]
    y_jitter = np.random.normal(0, 0.08, shap_vals_for_feature.shape[0])
    ax_main.scatter(shap_vals_for_feature, i + y_jitter, c=feature_vals_for_color,
                    cmap=cmap, s=15, alpha=0.8, zorder=2)

ax_main.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
ax_main.grid(True, axis='x', linestyle='--', alpha=0.6)

# --- Summary colorbar ---
fig.canvas.draw()
ax_main_pos = ax_main.get_position()
cax_left = ax_main_pos.x1 + aesthetic_params['summary_cbar_pad']
cax_bottom = ax_main_pos.y0 + (ax_main_pos.height * (1 - aesthetic_params['summary_cbar_height_shrink']) / 2)
cax_width = aesthetic_params['summary_cbar_width']
cax_height = ax_main_pos.height * aesthetic_params['summary_cbar_height_shrink']
cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
norm = plt.Normalize(vmin=X_test.values.min(), vmax=X_test.values.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax)
cbar.outline.set_visible(False)
cbar.set_ticks([])
cbar.ax.text(0.6, 1.04, 'High', ha='center', va='top', transform=cbar.ax.transAxes,
             fontsize=aesthetic_params['tick_label_size'])
cbar.ax.text(0.6, -0.04, 'Low', ha='center', va='bottom', transform=cbar.ax.transAxes,
             fontsize=aesthetic_params['tick_label_size'])

# --- Dependence plots (right) ---
top_6_features = feature_importance_df['feature'].tail(6).iloc[::-1].tolist()
axes_scatter = [fig.add_subplot(gs[i, j + 2]) for i in range(3) for j in range(2)]

for i, feature in enumerate(top_6_features):
    ax = axes_scatter[i]
    feature_idx = X.columns.get_loc(feature)
    x_data = X[feature]
    y_data = shap_values.values[:, feature_idx]
    color_data = y
    scatter = ax.scatter(x_data, y_data, c=color_data, cmap=cmap, s=25, alpha=0.8)

    # --- Dependence plot colorbar ---
    fig.canvas.draw()
    ax_pos = ax.get_position()
    cax_dep_left = ax_pos.x1 + aesthetic_params['dep_cbar_pad']
    cax_dep_bottom = ax_pos.y0 + (ax_pos.height * (1 - aesthetic_params['dep_cbar_height_shrink']) / 2)
    cax_dep_width = aesthetic_params['dep_cbar_width']
    cax_dep_height = ax_pos.height * aesthetic_params['dep_cbar_height_shrink']
    cax_dep = fig.add_axes([cax_dep_left, cax_dep_bottom, cax_dep_width, cax_dep_height])
    cbar = fig.colorbar(scatter, cax=cax_dep)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(axis='y', length=3)

    # --- Median and knee threshold lines ---
    median_val = X[feature].median()
    threshold_val = find_knee_point(x_data, y_data)
    ax.axvline(median_val, color='black', linestyle='--', linewidth=1)
    ax.axvline(threshold_val, color='red', linestyle=':', linewidth=1.2)
    ax.tick_params(axis='both', which='major', labelsize=aesthetic_params['tick_label_size'])

# --- Save figure ---
plt.savefig('shap_analysis_plot.tif', dpi=900, bbox_inches='tight', transparent=True)


plt.show()
