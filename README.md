# insurance-fraud-detection

PROJECT STRUCTURE

├── prepros.py # Data preprocessing & encoding 
├── modelex.py # Model training, evaluation & saving 
├── app.py # FastAPI app: prediction form + dashboard + DB 
├── insurance_claims.csv # Raw dataset 
├── processed_test.csv # Cleaned dataset 
├── model.pkl # Trained ML model (XGBoost) 
├── label_encoder_ins.pkl # Encoded categorical features

pip install -r requirements.txt

REQUIREMENTS
Main libraries used:

pandas, numpy, scikit-learn, xgboost
seaborn, matplotlib (EDA/visualization)
plotly (interactive dashboard)
fastapi, uvicorn (web app)
sqlalchemy, psycopg2 (PostgreSQL ORM & driver)
joblib, pickle (model persistence)

Install :
pip install fastapi uvicorn pandas numpy scikit-learn xgboost seaborn matplotlib plotly sqlalchemy psycopg2 joblib

DATA PREPROCESSING (prepros.py)
  Loads insurance_claims.csv
  Removes null values & irrelevant column (_c39)
  Encodes categorical columns using LabelEncoder
  Saves cleaned dataset as processed_test.csv

MODEL TRAINING (model.py)
  Splits dataset into train/test
  Tests multiple models (RandomForest, KNN, SVC, XGBoost)
  Uses XGBoost as final model
  Evaluates accuracy, precision, recall, F1-score
  Saves trained model as model.pkl
  Exports test set for later predictions (test.csv)

WEB APPLICATION (app.py)

FastAPI endpoints:

/ → Homepage with navigation
/predict (GET) → Fraud prediction form
/predict (POST) → Accepts form input, encodes categorical fields, predicts fraud using model, stores result in PostgreSQL


DATABASE (PostgreSQL)

The app stores user submissions + predictions in a PostgreSQL DB.

Database URL in app.py:

DATABASE_URL = "postgresql://postgres:pw1234@localhost:5432/insurance"

Make sure PostgreSQL is running and database insurance exists


RUNNING THE App
1. Run preprocessing
python prepros.py

2. Train and save model
python modelex.py

3. Start FastAPI app
uvicorn app:app --reload


Now open:

http://127.0.0.1:8000/ → Homepage
http://127.0.0.1:8000/predict → Fraud prediction form
http://127.0.0.1:8000/dashboard → Interactive dashboard
