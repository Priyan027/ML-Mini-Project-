"""Titanic survival model: feature engineering, training, prediction."""
import json, os
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA, MODEL, METRICS = (os.path.join(BASE, f) for f in ("train.csv", "model.pkl", "metrics.json"))

TITLES = {"Mr": "Mr", "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss", "Mrs": "Mrs",
          "Mme": "Mrs", "Master": "Master"}
NUM = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CAT = ["Sex", "Embarked", "Title"]


def add_features(df):
    df = df.copy()
    t = df["Name"].fillna("").str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["Title"] = t.map(TITLES).fillna(t.where(t.isna(), "Rare")).fillna("Mr")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df[NUM + CAT]


def build(clf):
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])


def train():
    raw = pd.read_csv(DATA)
    X, y = add_features(raw), raw["Survived"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
    results, fitted = [], {}
    for name, clf in candidates.items():
        pipe = build(clf)
        cvs = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="accuracy")
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        fitted[name] = pipe
        results.append({"name": name, "cv": round(cvs.mean(), 4), "accuracy": round(accuracy_score(yte, p), 4),
                        "precision": round(precision_score(yte, p), 4), "recall": round(recall_score(yte, p), 4),
                        "f1": round(f1_score(yte, p), 4), "cm": confusion_matrix(yte, p).tolist()})
    best = max(results, key=lambda r: r["cv"])["name"]
    joblib.dump(fitted[best], MODEL)
    grp = lambda c: {str(k): round(float(v), 3) for k, v in raw.groupby(c)["Survived"].mean().items()}
    info = {"best": best, "models": results, "rows": len(raw),
            "survival_rate": round(float(y.mean()), 3),
            "by_sex": grp("Sex"), "by_class": grp("Pclass")}
    json.dump(info, open(METRICS, "w"), indent=2)
    return info


def load():
    if not os.path.exists(MODEL):
        if not os.path.exists(DATA):
            return None
        train()
    return joblib.load(MODEL)


def predict(pipe, d):
    age = d.get("age")
    row = pd.DataFrame([{
        "Name": d.get("name", ""), "Sex": d["sex"], "Pclass": int(d["pclass"]),
        "Age": np.nan if age in (None, "") else float(age),
        "SibSp": int(d.get("sibsp", 0)), "Parch": int(d.get("parch", 0)),
        "Fare": float(d.get("fare", 0)), "Embarked": d.get("embarked", "S")}])
    return float(pipe.predict_proba(add_features(row))[0][1])


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
