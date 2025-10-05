# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

app = FastAPI(title="Used Car Price + Market Risk API")

CSV_PATH = os.path.join("data", "UsedCarDataset.csv")  # adjust path if needed

# ---------- Load dataset ----------
if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"CSV not found at {CSV_PATH}. Put your CSV at this path.")

df = pd.read_csv(CSV_PATH)

# ---------- Clean / normalize columns ----------
# rename awkward column names
rename_map = {
    "price(in lakhs)": "price_in_lakhs",
    "mileage(kmpl)": "mileagekmpl",
    "engine(cc)": "enginecc",
    "max_power(bhp)": "max_powerbhp",
    "torque(Nm)": "torqueNm"
}
df.rename(columns=rename_map, inplace=True)

# ensure price numeric
df['price_in_lakhs'] = pd.to_numeric(df['price_in_lakhs'], errors='coerce')

# create a 'brand' extracted from car_name (first token) to reduce cardinality
df['brand'] = df['car_name'].astype(str).apply(lambda s: s.split()[0] if isinstance(s, str) else 'Unknown')

# convert manufacturing_year to numeric (coerce errors)
df['manufacturing_year'] = pd.to_numeric(df['manufacturing_year'], errors='coerce')

# compute age feature (use manufacturing_year if available; otherwise try to parse registration_year)
CURRENT_YEAR = pd.Timestamp.now().year
def compute_age(row):
    if pd.notna(row.get('manufacturing_year')):
        return CURRENT_YEAR - int(row['manufacturing_year'])
    # fallback: try to extract year from registration_year if it's numeric-ish
    ry = str(row.get('registration_year', ''))
    import re
    m = re.search(r'(\d{4})', ry)
    if m:
        try:
            return CURRENT_YEAR - int(m.group(1))
        except:
            return np.nan
    # maybe format like '17-Jul' -> treat as 2017 if '17' seems year
    m2 = re.match(r'(\d{2})', ry)
    if m2:
        y = int(m2.group(1))
        # heuristic: if > 30 then 19xx else 20xx
        if y > 30:
            y = 1900 + y
        else:
            y = 2000 + y
        return CURRENT_YEAR - y
    return np.nan

df['age'] = df.apply(compute_age, axis=1)

# numeric fields: coerce to numeric
numeric_cols = ['seats', 'kms_driven', 'mileagekmpl', 'enginecc', 'max_powerbhp', 'torqueNm', 'age', 'price_in_lakhs']
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# fill missing numeric with median
for c in numeric_cols:
    if c in df.columns:
        df[c].fillna(df[c].median(), inplace=True)

# categorical columns to use
categorical = ['brand', 'insurance_validity', 'fuel_type', 'ownsership', 'transmission']
for c in categorical:
    if c not in df.columns:
        df[c] = 'Unknown'
    df[c] = df[c].fillna('Unknown').astype(str)

# reduce brand cardinality: keep top 12 brands, rest -> "Other"
top_brands = df['brand'].value_counts().nlargest(12).index.tolist()
df['brand'] = df['brand'].apply(lambda b: b if b in top_brands else 'Other')

# ---------- Features for regression/classification ----------
# For price regression we use a mixture of numeric + categorical (brand)
features = ['brand', 'fuel_type', 'transmission', 'ownsership', 'insurance_validity',
            'seats', 'kms_driven', 'mileagekmpl', 'enginecc', 'max_powerbhp', 'torqueNm', 'age']

target_price = 'price_in_lakhs'

# ensure the feature columns exist
for f in features:
    if f not in df.columns:
        df[f] = 0

# ---------- Preprocessor & models ----------
cat_features = ['brand', 'fuel_type', 'transmission', 'ownsership', 'insurance_validity']
num_features = ['seats', 'kms_driven', 'mileagekmpl', 'enginecc', 'max_powerbhp', 'torqueNm', 'age']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features)
    ],
    remainder='drop'
)

price_pipeline = Pipeline([
    ('pre', preprocessor),
    ('reg', LinearRegression())
])

# ---------- Prepare data and train regression ----------
X = df[features]
y = df[target_price]

# drop rows where target missing
mask = y.notna()
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)
price_pipeline.fit(X_train, y_train)

# optional: evaluate
train_score = price_pipeline.score(X_train, y_train)
test_score = price_pipeline.score(X_test, y_test)

# ---------- Market risk clustering ----------
# We'll cluster on (age, kms_driven, price) and then map clusters to risk label:
cluster_df = df[['age', 'kms_driven', 'price_in_lakhs']].copy()
cluster_df.fillna(cluster_df.median(), inplace=True)
# KMeans with 3 clusters (Low, Medium, High risk)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(cluster_df)
cluster_labels = kmeans.labels_

# assign cluster label into DataFrame
df['cluster_label'] = cluster_labels

# compute cluster mean price to sort risk ordering:
cluster_mean_price = df.groupby('cluster_label')['price_in_lakhs'].mean().sort_values()

# map cluster id -> risk: lowest mean price -> High risk, middle -> Medium, highest -> Low
cluster_order = list(cluster_mean_price.index)
risk_map = {}
if len(cluster_order) >= 3:
    # cluster_order[0] lowest price -> High risk
    risk_map[cluster_order[0]] = 'High'
    risk_map[cluster_order[1]] = 'Medium'
    risk_map[cluster_order[2]] = 'Low'
else:
    # fallback
    for i, cid in enumerate(cluster_order):
        risk_map[cid] = ['High', 'Medium', 'Low'][i]

df['market_risk'] = df['cluster_label'].map(risk_map)

# ---------- Train a classifier to predict market_risk from same features as regression ----------
# Convert market_risk to numeric labels
df['market_risk_label'] = df['market_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})

# For classifier, reuse preprocessor but maybe simpler OneHot + StandardScaler via a pipeline
clf_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', DecisionTreeClassifier(max_depth=6, random_state=42))
])

clf_mask = df['market_risk_label'].notna()
X_clf = df.loc[clf_mask, features]
y_clf = df.loc[clf_mask, 'market_risk_label']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.18, random_state=42)
clf_pipeline.fit(Xc_train, yc_train)
clf_acc = accuracy_score(yc_test, clf_pipeline.predict(Xc_test))

# Save models to disk (optional)
joblib.dump(price_pipeline, "price_pipeline.joblib")
joblib.dump(clf_pipeline, "risk_pipeline.joblib")
joblib.dump(kmeans, "kmeans_model.joblib")

# ---------- Pydantic models for predict endpoints ----------
class CarFeatures(BaseModel):
    car_name: str = None
    registration_year: str = None
    insurance_validity: str = None
    fuel_type: str = None
    seats: int = None
    kms_driven: int = None
    ownsership: str = None
    transmission: str = None
    manufacturing_year: int = None
    mileagekmpl: float = None
    enginecc: float = None
    max_powerbhp: float = None
    torqueNm: float = None

# ---------- API endpoints ----------
@app.get("/")
def root():
    return {"message": "Used Car Price + Market Risk API is running."}

@app.get("/summary")
def summary():
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "columns_list": df.columns.tolist(),
        "price_train_score": float(train_score),
        "price_test_score": float(test_score),
        "risk_classifier_accuracy": float(clf_acc)
    }

@app.get("/eda")
def eda():
    # return simple correlations for numeric columns
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric].corr().round(3).to_dict()
    # return top brand counts
    brand_counts = df['brand'].value_counts().head(12).to_dict()
    return {"correlations": corr, "brand_counts": brand_counts}

@app.get("/clusters")
def clusters():
    centers = kmeans.cluster_centers_.tolist()
    # sample few rows with cluster and risk
    sample = df[['car_name', 'age', 'kms_driven', 'price_in_lakhs', 'cluster_label', 'market_risk']].head(40).to_dict(orient='records')
    return {"centers": centers, "sample": sample, "risk_map": risk_map}

@app.post("/predict_price")
def predict_price(car: CarFeatures):
    # build single-row DataFrame in same format as training features
    payload = car.dict()
    # brand extraction
    brand = (payload.get('car_name') or '').split()[0] if payload.get('car_name') else 'Unknown'
    if brand not in top_brands:
        brand = 'Other'
    row = {
        'brand': brand,
        'fuel_type': payload.get('fuel_type') or 'Unknown',
        'transmission': payload.get('transmission') or 'Unknown',
        'ownsership': payload.get('ownsership') or 'Unknown',
        'insurance_validity': payload.get('insurance_validity') or 'Unknown',
        'seats': payload.get('seats') or df['seats'].median(),
        'kms_driven': payload.get('kms_driven') or df['kms_driven'].median(),
        'mileagekmpl': payload.get('mileagekmpl') or df['mileagekmpl'].median(),
        'enginecc': payload.get('enginecc') or df['enginecc'].median(),
        'max_powerbhp': payload.get('max_powerbhp') or df['max_powerbhp'].median(),
        'torqueNm': payload.get('torqueNm') or df['torqueNm'].median(),
        'age': None
    }
    # compute age from manufacturing_year if provided
    my = payload.get('manufacturing_year')
    if my:
        row['age'] = CURRENT_YEAR - int(my)
    else:
        row['age'] = df['age'].median()
    Xpred = pd.DataFrame([row])[features]
    pred = price_pipeline.predict(Xpred)[0]
    return {"predicted_price_in_lakhs": round(float(pred), 2)}

@app.post("/predict_risk")
def predict_risk(car: CarFeatures):
    payload = car.dict()
    brand = (payload.get('car_name') or '').split()[0] if payload.get('car_name') else 'Unknown'
    if brand not in top_brands:
        brand = 'Other'
    row = {
        'brand': brand,
        'fuel_type': payload.get('fuel_type') or 'Unknown',
        'transmission': payload.get('transmission') or 'Unknown',
        'ownsership': payload.get('ownsership') or 'Unknown',
        'insurance_validity': payload.get('insurance_validity') or 'Unknown',
        'seats': payload.get('seats') or int(df['seats'].median()),
        'kms_driven': payload.get('kms_driven') or int(df['kms_driven'].median()),
        'mileagekmpl': payload.get('mileagekmpl') or float(df['mileagekmpl'].median()),
        'enginecc': payload.get('enginecc') or float(df['enginecc'].median()),
        'max_powerbhp': payload.get('max_powerbhp') or float(df['max_powerbhp'].median()),
        'torqueNm': payload.get('torqueNm') or float(df['torqueNm'].median()),
        'age': None
    }
    my = payload.get('manufacturing_year')
    if my:
        row['age'] = CURRENT_YEAR - int(my)
    else:
        row['age'] = df['age'].median()

    Xpred = pd.DataFrame([row])[features]
    pred_label = clf_pipeline.predict(Xpred)[0]
    inv_map = {0:'Low',1:'Medium',2:'High'}
    return {"predicted_market_risk": inv_map.get(int(pred_label), "Unknown")}

# optional endpoint to return a few rows sample
@app.get("/sample")
def sample(n: int = 20):
    return df.head(n).to_dict(orient='records')

from pydantic import BaseModel

class CarFeatures(BaseModel):
    registration_year: int
    kms_driven: float
    seats: int
    enginecc: float
    max_powerbhp: float
    mileagekmpl: float

@app.post("/predict")
def predict_price(car: CarFeatures):
    try:
        # Put inputs into same order as features list
        input_data = pd.DataFrame([[
            car.registration_year,
            car.kms_driven,
            car.seats,
            car.enginecc,
            car.max_powerbhp,
            car.mileagekmpl
        ]], columns=features)

        prediction = model.predict(input_data)[0]
        return {"predicted_price_in_lakhs": round(float(prediction), 2)}
    except Exception as e:
        return {"error": str(e)}
