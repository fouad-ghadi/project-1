import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb


# Chargement du dataset
df = pd.read_csv(r"data\nouvelle dataset equilibrée.csv")

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs("models", exist_ok=True)


# ─── Random Forest ───────────────────────────────────────

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "scr/heart_model.pkl")
    print("Random Forest sauvegardé : scr/heart_model.pkl")


# ─── XGBoost ─────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "scr/heart_model_xgb.pkl")
    print("XGBoost sauvegardé : scr/heart_model_xgb.pkl")


# ─── LightGBM ────────────────────────────────────────────

def train_lightgbm(X_train, y_train):
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        random_state=42,
        class_weight="balanced",
        verbose=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "scr/heart_model_lgbm.pkl")
    print("LightGBM sauvegardé : scr/heart_model_lgbm.pkl")


# ─── Logistic Regression ─────────────────────────────────

def train_logistic_regression(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, "scr/heart_model_lr.pkl")
    print("Logistic Regression sauvegardée : scr/heart_model_lr.pkl")


# ─── Lancement ───────────────────────────────────────────

train_random_forest(X_train, y_train)
train_xgboost(X_train, y_train)
train_lightgbm(X_train, y_train)
train_logistic_regression(X_train, y_train)


print("\nTous les modèles sont entraînés et sauvegardés.")
