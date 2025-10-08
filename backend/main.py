

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

# ==================================================================
# Load the RAW CSV
CSV_PATH = "UsedCarDataset.csv"
# ==================================================================

app = FastAPI(title="Used Car Price + Market Risk API")

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load & Clean RAW Dataset ----------
if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"RAW CSV not found at {CSV_PATH}. Please ensure 'UsedCarDataset.csv' is in the project root.")

df = pd.read_csv(CSV_PATH, index_col=0)

# 1. Rename awkward column names from RAW CSV
rename_map = {
    "price(in lakhs)": "price_in_lakhs",
    "mileage(kmpl)": "mileagekmpl",
    "engine(cc)": "enginecc",
    "max_power(bhp)": "max_powerbhp",
    "torque(Nm)": "torqueNm"
}
df.rename(columns=rename_map, inplace=True)

# 2. Coerce price
df['price_in_lakhs'] = pd.to_numeric(df['price_in_lakhs'], errors='coerce')

# 3. Create 'brand' feature and reduce cardinality
"""df['brand'] = df['car_name'].astype(str).apply(lambda s: s.split()[0] if isinstance(s, str) else 'Unknown')
top_brands = df['brand'].value_counts().nlargest(12).index.tolist()
df['brand'] = df['brand'].apply(lambda b: b if b in top_brands else 'Other')"""
# Take the second word as brand (assuming format "<year> <brand> ...")
df['brand'] = df['car_name'].astype(str).apply(lambda s: s.split()[1] if len(s.split()) > 1 else 'Unknown')
top_brands = df['brand'].value_counts().nlargest(12).index.tolist()
df['brand'] = df['brand'].apply(lambda b: b if b in top_brands else 'Other')


# 4. Compute Age feature from manufacturing_year
CURRENT_YEAR = pd.Timestamp.now().year
df['manufacturing_year'] = pd.to_numeric(df['manufacturing_year'], errors='coerce')
df['age'] = CURRENT_YEAR - df['manufacturing_year']

# 5. Define Feature Lists
target_price = 'price_in_lakhs'
features = ['brand', 'fuel_type', 'transmission', 'ownsership', 'insurance_validity',
            'seats', 'kms_driven', 'mileagekmpl', 'age'] 

# 6. Final Imputation 
for c in features + [target_price]:
    if c not in df.columns:
        df[c] = np.nan

    if df[c].dtype in [np.dtype('float64'), np.dtype('int64')]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna('Unknown').astype(str)


# ---------- Preprocessor & Model Training (Price) ----------
cat_features = ['brand', 'fuel_type', 'transmission', 'ownsership', 'insurance_validity']
num_features = ['seats', 'kms_driven', 'mileagekmpl', 'age'] 

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features)
    ],
    remainder='drop'
)

# Price Prediction Pipeline (Random Forest Regressor)
price_pipeline = Pipeline([
    ('pre', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) 
])

X = df[features]
y = df[target_price]
mask = y.notna()
X = X[mask]
y = y[mask]

# Apply LOG TRANSFORMATION TO TARGET VARIABLE
y_log = np.log1p(y)

X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.18, random_state=42)
price_pipeline.fit(X_train, y_log_train)

train_score = price_pipeline.score(X_train, y_log_train)
test_score = price_pipeline.score(X_test, y_log_test)


import joblib

scaler_2f = joblib.load("results/scaler_2f.joblib")
kmeans_2f = joblib.load("results/kmeans_2f.joblib")
risk_map_2f = joblib.load("results/risk_map_2f.joblib")



# ---------- Market Risk Clustering (K-Means) ----------
from sklearn.preprocessing import StandardScaler

# Select key features
cluster_cols = ['age', 'kms_driven', 'price_in_lakhs']
cluster_df = df[cluster_cols].copy()
cluster_df.fillna(cluster_df.median(), inplace=True)

# Apply scaling for balance
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_df)

# Apply feature weighting (age and kms matter more than price)
weights = np.array([1.5, 1.5, 0.7])   # adjust these ratios if needed
cluster_scaled_weighted = cluster_scaled * weights

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(cluster_scaled_weighted)
df['cluster_label'] = kmeans.labels_

# Map cluster ID to Risk Label based on mean price
cluster_mean_price = df.groupby('cluster_label')['price_in_lakhs'].mean().sort_values()
cluster_order = list(cluster_mean_price.index)

risk_map = {}
if len(cluster_order) >= 3:
    risk_map[cluster_order[0]] = 'High'   # lowest mean price → highest risk
    risk_map[cluster_order[1]] = 'Medium'
    risk_map[cluster_order[2]] = 'Low'    # highest mean price → lowest risk
else:
    for i, cid in enumerate(cluster_order):
        risk_map[cid] = ['High', 'Medium', 'Low'][i]

df['market_risk'] = df['cluster_label'].map(risk_map)


# Save models
os.makedirs("results", exist_ok=True)
joblib.dump(price_pipeline, "results/price_pipeline.joblib")
joblib.dump(kmeans, "results/kmeans_model.joblib")
joblib.dump(scaler, "results/kmeans_scaler.joblib")


# ---------- Pydantic Models & API Endpoints ----------

class CarFeatures(BaseModel):
    car_name: str = None
    fuel_type: str = None
    transmission: str = None
    ownership: str = None
    insurance_validity: str = None
    seats: int = None
    kms_driven: int = None
    mileagekmpl: float = None
    enginecc: float = None 
    max_powerbhp: float = None 
    torqueNm: float = None 
    manufacturing_year: int = None


@app.get("/")
def root():
    return {"message": "Used Car Price + Market Risk API is running."}


@app.get("/summary")
def summary():
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "price_test_score": round(float(test_score), 3),
        "cluster_risks": risk_map
    }


@app.post("/predict_price")
def predict_price(car: CarFeatures):
    payload = car.dict()
    brand = (payload.get('car_name') or '').split()[0] if payload.get('car_name') else 'Other'
    if brand not in top_brands:
        brand = 'Other'
        
    row = {
        'brand': brand,
        'fuel_type': payload.get('fuel_type') or 'Unknown',
        'transmission': payload.get('transmission') or 'Unknown',
        'ownership': payload.get('ownership') or 'Unknown',
        'insurance_validity': payload.get('insurance_validity') or 'Unknown',
        'seats': payload.get('seats') or df['seats'].median(),
        'kms_driven': payload.get('kms_driven') or df['kms_driven'].median(),
        'mileagekmpl': payload.get('mileagekmpl') or df['mileagekmpl'].median(),
        'age': None
    }
    
    my = payload.get('manufacturing_year')
    row['age'] = CURRENT_YEAR - int(my) if my else df['age'].median()
    
    Xpred = pd.DataFrame([row])[features] 
    pred_log = price_pipeline.predict(Xpred)[0]
    pred = np.expm1(pred_log) 
    
    return {"predicted_price_in_lakhs": round(float(pred), 2)}



"""


@app.post("/predict_risk")
def predict_risk(car: CarFeatures):
    payload = car.dict()

    # Compute age
    age = CURRENT_YEAR - int(payload.get('manufacturing_year')) if payload.get('manufacturing_year') else df['age'].median()
    kms_driven = float(payload.get('kms_driven') or df['kms_driven'].median())
    ownsership = payload.get('ownsership') or 'Unknown'

    # --- Predict price using pipeline ---
    brand = (payload.get('car_name') or '').split()[0] if payload.get('car_name') else 'Other'
    if brand not in top_brands:
        brand = 'Other'

    row = {
        'brand': brand,
        'fuel_type': payload.get('fuel_type') or 'Unknown',
        'transmission': payload.get('transmission') or 'Unknown',
        'ownsership': ownsership,
        'insurance_validity': payload.get('insurance_validity') or 'Unknown',
        'seats': payload.get('seats') or df['seats'].median(),
        'kms_driven': kms_driven,
        'mileagekmpl': payload.get('mileagekmpl') or df['mileagekmpl'].median(),
        'age': age
    }

    Xpred_price = pd.DataFrame([row])[features]
    pred_log = price_pipeline.predict(Xpred_price)[0]
    predicted_price = np.expm1(pred_log)

    # --- Adjust price based on ownership ---
    if ownsership.lower() == '2nd owner':
        predicted_price *= 0.85  # reduce by 15%
    elif ownsership.lower() == '3rd owner':
        predicted_price *= 0.7   # reduce by 30%

    # --- Determine risk ---
    if ownsership.lower() == '3rd owner':
        predicted_risk = 'High'
        cluster_id = None
    else:
        # KMeans-based risk
        kmeans_input = np.array([[age, kms_driven]])
        scaled_input = scaler_2f.transform(kmeans_input) * np.array([1.5, 1.5])
        cluster_id = kmeans_2f.predict(scaled_input)[0]
        predicted_risk = risk_map_2f.get(cluster_id, "Unknown")

    return {
        "predicted_market_risk": predicted_risk,
        "predicted_price_in_lakhs": round(float(predicted_price), 2),
        "cluster_id": None if ownsership.lower() == '3rd owner' else int(cluster_id),
        "age": int(age),
        "kms_driven": kms_driven,
        "ownsership": ownsership
    }
"""


@app.post("/predict_risk")
def predict_risk(car: CarFeatures):
    payload = car.dict()

    # Compute age
    my = payload.get('manufacturing_year')
    age = CURRENT_YEAR - int(my) if my else 0

    kms_driven = float(payload.get('kms_driven') or 0)
    ownership = payload.get('ownership', '').lower().strip()
    mileage = float(payload.get('mileagekmpl') or 0)

    # Base KMeans prediction
    kmeans_input = np.array([[age, kms_driven]])
    scaled_input = scaler_2f.transform(kmeans_input) * np.array([1.5, 1.5])
    cluster_id = kmeans_2f.predict(scaled_input)[0]
    predicted_risk = risk_map_2f.get(cluster_id, "Unknown")

    # --- Ownership logic ---
    ownership_factor = 1.0
    if "second" in ownership:
        ownership_factor = 0.93    # Slight drop
    elif "third" in ownership:
        ownership_factor = 0.87    # Noticeable drop
        predicted_risk = "High"    # Automatically high risk
    elif "fourth" in ownership:
        ownership_factor = 0.80
        predicted_risk = "High"

    # --- Price estimation (simple illustrative model) ---
    base_price = max(2, 12 - (age * 0.4) - (kms_driven / 30000))
    price_adjusted = base_price * ownership_factor * (1 + (mileage - 15) / 100)

    return {
        "predicted_market_risk": predicted_risk,
        "predicted_price_in_lakhs": round(price_adjusted, 2),
        "age": int(age),
        "kms_driven": kms_driven,
        "ownership": ownership.title()
    }


@app.get("/car_names")
def get_car_names():
    return {"car_names": sorted(df['car_name'].unique().tolist())}


@app.get("/clusters")
def clusters():
    centers = kmeans.cluster_centers_.tolist()
    sample = df[['car_name', 'age', 'kms_driven', 'price_in_lakhs', 'cluster_label', 'market_risk']].head(40).to_dict(orient='records')
    return {"centers": centers, "sample": sample, "risk_map": risk_map}


@app.get("/eda")
def eda():
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    clean_numeric = [c for c in numeric if c not in ['enginecc', 'max_powerbhp', 'torqueNm']]
    corr_dict = df[clean_numeric].corr().round(3).to_dict()
    brand_counts = df["brand"].value_counts().head(10).to_dict()
    return {"correlations": corr_dict, "brand_counts": brand_counts}


