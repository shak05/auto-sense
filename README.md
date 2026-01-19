# AutoSense

AutoSense is a full-stack machine learning application for used car price analysis and market value risk assessment. The project combines data analysis, clustering, and a web-based interface to provide insights into used car pricing trends.

## Overview
The system analyzes used car data based on features such as age, mileage, brand, ownership, and kilometers driven. It predicts expected prices and segments cars into different market value risk categories using machine learning techniques.

The project is structured with a clear separation between frontend and backend.

## Project Structure
AUTOSENSE/
├── backend/
│   ├── main.py
│   ├── train_kmeans.py
│   ├── requirements.txt
│   ├── UsedCarDataset.csv
│   ├── price_pipeline.joblib
│   ├── risk_pipeline.joblib
│   └── kmeans_model.joblib
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
└── README.md

## Features
- Used car price prediction using machine learning
- Market value risk analysis using clustering techniques
- Exploratory Data Analysis (EDA) on key vehicle attributes
- Separate frontend and backend architecture
- Interactive web interface for displaying insights

## Tech Stack
- **Backend:** Python, Machine Learning, FastAPI/Flask
- **ML Libraries:** Pandas, NumPy, Scikit-learn
- **Frontend:** HTML, CSS, JavaScript, Vite
- **Modeling:** Regression, K-Means Clustering

## How to Run

### Backend
1. Navigate to the backend folder
2. Install dependencies:
   pip install -r requirements.txt
3. Run the backend server:
   python main.py

### Frontend
1. Navigate to the frontend folder
2. Install dependencies:
   npm install
3. Start the frontend:
   npm run dev

## Purpose
AutoSense was developed as a learning-focused project to explore machine learning, data analysis, and full-stack application development in the automotive domain.

## Tags
machine-learning, data-analysis, used-car-price, full-stack, autosense
