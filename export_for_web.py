import joblib
import pandas as pd
import json
import os

print("Loading models...")
scaler = joblib.load('workbook/scaler.joblib')
kmeans = joblib.load('workbook/kmeans_model.joblib')
df = pd.read_parquet('workbook/dashboard_data.parquet')

print(f"Loaded {len(df)} customers, {df['Cluster'].nunique()} clusters")

scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}

centroids = kmeans.cluster_centers_.tolist()

n_sample = min(1000, len(df))
sample = df.sample(n=n_sample, random_state=42).reset_index(drop=True)

customers = [
    {
        'recency': float(row['recency']),
        'frequency': float(row['frequency']),
        'monetary': round(float(row['monetary']), 2),
        'cluster': int(row['Cluster'])
    }
    for _, row in sample.iterrows()
]

data = {
    'scaler': scaler_params,
    'centroids': centroids,
    'customers': customers
}

os.makedirs('docs', exist_ok=True)
with open('docs/data.json', 'w') as f:
    json.dump(data, f)

print(f"\nExported to docs/data.json")
print(f"  Customers : {len(customers)}")
print(f"  Scaler mean  : {[round(x, 4) for x in scaler_params['mean']]}")
print(f"  Scaler scale : {[round(x, 4) for x in scaler_params['scale']]}")
print(f"  Centroids    : {len(centroids)} clusters x {len(centroids[0])} features")
print(f"\nNext: commit docs/ folder and enable GitHub Pages from the docs/ folder in repo settings.")
