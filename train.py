"""Treinamento do modelo (Obesity.csv) - POSTECH Tech Challenge Fase 04
- Pipeline completa: arredondamento + one-hot + RandomForest
- Gera model.pkl e metrics.json
"""
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "Obesity.csv"
OUT_MODEL_PATH = "model.pkl"
OUT_METRICS_PATH = "metrics.json"

class RoundColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xc = X.copy()
        for c in self.cols:
            if c in Xc.columns:
                Xc[c] = np.round(Xc[c]).astype(float)
        return Xc

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Obesity"])
    y = df["Obesity"]

    numeric_cols = ["Age","Height","Weight","FCVC","NCP","CH2O","FAF","TUE"]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = Pipeline(steps=[
        ("rounder", RoundColumns(["FCVC","NCP","CH2O","FAF","TUE"])),
        ("ct", ColumnTransformer([
            ("num","passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]))
    ])

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)

    joblib.dump(pipe, OUT_MODEL_PATH)
    with open(OUT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "test_accuracy": float(acc),
            "classification_report": classification_report(y_test, pred),
            "classes": list(pipe.classes_)
        }, f, ensure_ascii=False, indent=2)

    print("OK - Accuracy:", acc)

if __name__ == "__main__":
    main()

"""
(OPCIONAL) Tuning com GridSearchCV (pode demorar alguns minutos localmente):
from sklearn.model_selection import GridSearchCV
param_grid = {
  "model__n_estimators":[200, 350, 500],
  "model__max_depth":[None, 14, 20],
  "model__min_samples_split":[2, 6]
}
gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, scoring="accuracy")
gs.fit(X_train, y_train)
best = gs.best_estimator_
"""
