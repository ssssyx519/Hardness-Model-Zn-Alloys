"""
UMAP Dimensionality Reduction for Alloy Composition Data
========================================================
This script performs:

1. Load alloy composition data from Excel.
2. Check data completeness.
3. Apply UMAP dimensionality reduction directly on raw data (no normalization).
4. Save results to Excel, including UMAP coordinates and logs.

Author: Yaxuan Shen
License: MIT
"""

import pandas as pd
import numpy as np
import umap
import warnings
import os
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# =====================================================================
#  1. Load Data
# =====================================================================

df = pd.read_excel('grid_space.xlsx')

print(f" Raw data loaded successfully")
print(f"  Number of samples: {len(df):,}")
print(f"  Number of features: {df.shape[1]}\n")


# =====================================================================
#  2. Convert to numeric type
# =====================================================================
features = df.values.astype(np.float32)   # ensure float type
# Normalization (Min-Max)
scaler = MinMaxScaler(feature_range=(0, 1))
try:
    scaled_features = scaler.fit_transform(features)
    print(" Normalization completed")

    # Build normalized DataFrame (keep column names and index)
    normalized_df = pd.DataFrame(
        scaled_features,
        columns=df.columns,
        index=df.index
    )
except Exception as e:
    print(f" Normalization failed: {str(e)}")
    exit()
# =====================================================================
#  3. UMAP Dimensionality Reduction
# =====================================================================
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=10,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    low_memory=True if len(df) > 1e5 else False
)

try:
    embedding = reducer.fit_transform(scaled_features)
    print(f" UMAP reduction completed (samples: {embedding.shape[0]:,})")
except Exception as e:
    print(f" UMAP reduction failed: {str(e)}")
    exit()

# =====================================================================
#  4. Data Integrity Check
# =====================================================================
assert len(embedding) == len(df), "Error: sample count mismatch after UMAP!"
print("\n Data integrity check passed")

# =====================================================================
#  5. Construct Result DataFrame
# =====================================================================
result_df = pd.DataFrame(
    embedding,
    columns=['UMAP1', 'UMAP2'],
    index=df.index
)

# =====================================================================
#  6. Save to Excel
# =====================================================================


try:
    with pd.ExcelWriter('UMAP.xlsx', engine='openpyxl') as writer:
        # Main results
        result_df.to_excel(
            writer,
            sheet_name='UMAP_Results',
            index=False
        )
        # Process log
        pd.DataFrame({
            'Processed_Time': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Original_Features': [df.shape[1]],
            'Reduced_Features': [2]
        }).to_excel(writer, sheet_name='Process_Log', index=False)

    print(f"   File size: {os.path.getsize('UMAP.xlsx') / 1024 ** 2:.2f} MB")

except PermissionError:
    print("\n Please close the Excel file and try again.")
except Exception as e:
    print(f"\n Write failed: {str(e)}")

# =====================================================================
#  7. Preview Results
# =====================================================================
print("\n Preview of final UMAP results:")
print(result_df.head())
