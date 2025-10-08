"""# train_kmeans.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# =========================
# Load dataset
# =========================
CSV_PATH = "UsedCarDataset.csv"
df_2f = pd.read_csv(CSV_PATH)
CURRENT_YEAR = pd.Timestamp.now().year

# =========================
# Compute numeric features
# =========================
df_2f['manufacturing_year'] = pd.to_numeric(df_2f['manufacturing_year'], errors='coerce')
df_2f['age'] = CURRENT_YEAR - df_2f['manufacturing_year']
df_2f['kms_driven'] = pd.to_numeric(df_2f['kms_driven'], errors='coerce')

# Fill only numeric columns with median
numeric_cols = ['age', 'kms_driven']
df_2f[numeric_cols] = df_2f[numeric_cols].fillna(df_2f[numeric_cols].median())

# =========================
# 2-feature scaler + weighting
# =========================
scaler_2f = StandardScaler().fit(df_2f[numeric_cols])
weighted_scaled = scaler_2f.transform(df_2f[numeric_cols]) * np.array([1.5, 1.5])

# =========================
# Fit KMeans
# =========================
kmeans_2f = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(weighted_scaled)

# =========================
# Map cluster to High/Medium/Low risk by age
# =========================
cluster_mean_age = df_2f.join(pd.Series(kmeans_2f.labels_, name='cluster')) \
                         .groupby('cluster')['age'].mean()
cluster_order = cluster_mean_age.sort_values().index.tolist()
risk_map_2f = {cluster_order[0]: 'High', cluster_order[1]: 'Medium', cluster_order[2]: 'Low'}

# =========================
# Save everything
# =========================
os.makedirs("results", exist_ok=True)
joblib.dump(scaler_2f, "results/scaler_2f.joblib")
joblib.dump(kmeans_2f, "results/kmeans_2f.joblib")
joblib.dump(risk_map_2f, "results/risk_map_2f.joblib")

print("Saved scaler_2f, kmeans_2f, and risk_map_2f in 'results/' folder.")
"""


# ================== train_kmeans.py ==================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Load dataset
df_2f = pd.read_csv("UsedCarDataset.csv")

# Current year for age calculation
CURRENT_YEAR = pd.Timestamp.now().year

# Preprocess for clustering
df_2f['manufacturing_year'] = pd.to_numeric(df_2f['manufacturing_year'], errors='coerce')
df_2f['age'] = CURRENT_YEAR - df_2f['manufacturing_year']
df_2f['kms_driven'] = pd.to_numeric(df_2f['kms_driven'], errors='coerce')

# Fill numeric missing values only (ignore non-numeric columns)
for col in ['age', 'kms_driven']:
    df_2f[col].fillna(df_2f[col].median(), inplace=True)

# 2-feature scaler
scaler_2f = StandardScaler().fit(df_2f[['age', 'kms_driven']])

# Weighted scaling (give more importance to age and kms)
weights = np.array([1.5, 1.5])  # tweak if needed
weighted_scaled = scaler_2f.transform(df_2f[['age', 'kms_driven']]) * weights

# Fit KMeans
kmeans_2f = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans_2f.fit(weighted_scaled)

# Determine cluster risk based on weighted mean of age+kms_driven
cluster_stats = df_2f[['age', 'kms_driven']].copy()
cluster_stats['cluster'] = kmeans_2f.labels_
cluster_mean_score = cluster_stats.groupby('cluster').apply(lambda x: np.mean(x['age']*1.5 + x['kms_driven']*1.5))
cluster_order = cluster_mean_score.sort_values(ascending=False).index.tolist()  # Descending → High risk first

# Map cluster to High/Medium/Low risk
risk_map_2f = {
    cluster_order[0]: 'High',
    cluster_order[1]: 'Medium',
    cluster_order[2]: 'Low'
}

# Save scaler, kmeans model, and risk map
joblib.dump(scaler_2f, "results/scaler_2f.joblib")
joblib.dump(kmeans_2f, "results/kmeans_2f.joblib")
joblib.dump(risk_map_2f, "results/risk_map_2f.joblib")

print("✅ 2-feature KMeans, scaler, and risk map saved successfully!")
