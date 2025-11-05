import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
import os


def load_dataset(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[
        'age', 'policy_number', 'policy_bind_date', 'policy_state', 'policy_csl',
        'umbrella_limit', 'insured_zip', 'insured_relationship', 'incident_date',
        'incident_state', 'incident_city', 'incident_location', 'vehicle_claim',
        'auto_make', 'fraud_reported'
    ])
    y = df['fraud_reported']
    print("Data loaded successfully for modeling.")
    return X, y

#checking fro cross-val scores of diff models
def compare_models(X, y):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=60),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(),
        "XGB": XGBClassifier()
    }
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=10)
        print(f"{name} CV scores: {scores} | Mean: {scores.mean():.4f}")


def train_xgb_model(X_train, y_train):
    model = XGBClassifier(n_estimators=250, learning_rate=0.25, max_depth=4)
    cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print("Cross-validation scores:", cv_scores)
    print("Mean CV accuracy:", cv_scores.mean())

    model.fit(X_train, y_train)
    return model

#scores for various evaluation metries
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    print("Train accuracy:", accuracy_score(y_train, y_train_pred))

    y_test_pred = model.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall:", recall_score(y_test, y_test_pred))
    print("F1 Score:", f1_score(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model saved successfully at: {os.path.abspath(path)}")


def modeling_pipeline(processed_path: str, model_save_path: str):
    X, y = load_dataset(processed_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    print("Train/Test split:", X_train.shape, X_test.shape)

    compare_models(X, y)
    model = train_xgb_model(X_train, y_train)
    evaluate_model(model, X_train, y_train, X_test, y_test)
    save_model(model, model_save_path)

    X_test.to_csv("test.csv", index=False)
    print("Test data saved as test.csv")

    return model_save_path


if __name__ == "__main__":
    modeling_pipeline(
        processed_path=r"D:\python\insurance fraud\processed.csv",
        model_save_path=r"D:\python\insurance fraud\model.pkl"
    )
