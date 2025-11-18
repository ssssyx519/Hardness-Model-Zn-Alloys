"""
GOOSE Algorithm for Zinc Alloy Hardness Optimization
===================================================
This module implements:

1. Loading a pre-trained Random Forest model.
2. Defining alloy composition bounds and the objective function.
3. Running the GOOSE algorithm for global optimization.
4. Saving all evaluated individuals and plotting convergence curve.

Author: Yaxuan Shen
License: MIT
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================================================
#  1. Load Pre-trained Random Forest Model
# =====================================================================
model = joblib.load('random_forest_model.pkl')
evaluation_data = []  # Stores all evaluated individuals

# =====================================================================
#  2. Define Alloy Features and Bounds
# =====================================================================
feature_names = ['Al', 'Cu', 'Mg', 'Ti', 'Mn', 'Si', 'Zn']
bounds = np.array([
    [5, 11],    # Al
    [1, 6],     # Cu
    [0, 0.1],   # Mg
    [0, 0.4],   # Ti
    [0.1, 0.6], # Mn
    [0, 0.8],   # Si
])

def objective_function(components):
    """
    Objective function: compute Zn and predict hardness using RF.
    Maximizes hardness.
    """
    zn_value = 100 - np.sum(components)
    input_data = pd.DataFrame([list(components) + [zn_value]], columns=feature_names)
    predicted_hardness = model.predict(input_data)[0]
    evaluation_data.append(list(components) + [zn_value, predicted_hardness])
    return -predicted_hardness  # maximize hardness

# =====================================================================
#  3. GOOSE Algorithm Implementation
# =====================================================================
def GOOSE(SearchAgents_no, Max_IT):
    dim = bounds.shape[0]
    lb, ub = bounds[:, 0], bounds[:, 1]
    Best_pos = np.zeros(dim)
    Best_score = float('-inf')
    Convergence_curve = np.zeros(Max_IT)

    # Initialize population
    X = np.random.uniform(lb, ub, (SearchAgents_no, dim))

    for loop in range(Max_IT):
        # Evaluate fitness
        for i in range(SearchAgents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            fitness = objective_function(X[i, :])
            if fitness > Best_score:
                Best_score = fitness
                Best_pos = X[i, :].copy()

        # Update positions
        for i in range(SearchAgents_no):
            pro = np.random.rand()
            rnd = np.random.rand()
            coe = min(np.random.rand(), 0.17)
            S_W = np.random.randint(5, 26)
            T_o_A_O, T_o_A_S = np.random.rand(dim), np.random.rand(dim)
            T_T, T_A = np.sum(T_o_A_S) / dim, np.sum(T_o_A_S) / (2 * dim)

            if rnd >= 0.5:
                if pro > 0.2:
                    F_F_S = T_o_A_O * (np.sqrt(S_W) / 9.81) if S_W >= 12 else T_o_A_O * (S_W / 9.81)
                    D_S_T, D_G = 343.2 * T_o_A_S, 0.5 * 343.2 * T_o_A_S
                    X[i, :] = F_F_S + D_G * T_A**2 if S_W >= 12 else F_F_S * D_G * T_A**2 * coe
                else:
                    alpha = 2 - (loop / (Max_IT / 2))
                    X[i, :] = np.random.randn(dim) * (Best_score * alpha) + Best_pos

        Convergence_curve[loop] = -Best_score

    return Best_pos, -Best_score, Convergence_curve

# =====================================================================
#  4. Run GOOSE Optimization
# =====================================================================
if __name__ == "__main__":
    SearchAgents_no = 200
    Max_IT = 500

    best_components, best_hardness, convergence_curve = GOOSE(SearchAgents_no, Max_IT)

    # Compute Zn and output results
    zn_value = 100 - np.sum(best_components)
    best_result = np.append(best_components, zn_value)
    print("Best Components (Al, Cu, Mg, Ti, Mn, Si, Zn):", best_result)
    print("Best Hardness:", best_hardness)

    # =================================================================
    #  5. Save All Evaluated Individuals
    # =================================================================
    df = pd.DataFrame(evaluation_data, columns=feature_names + ['Hardness'])
    df.to_excel('all_goose_generations', index=False)

    # =================================================================
    #  6. Plot Convergence Curve
    # =================================================================
    plt.figure()
    plt.plot(convergence_curve, color='r')
    plt.title("GOOSE Algorithm Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Best Hardness")
    plt.grid(True)
    plt.show()
