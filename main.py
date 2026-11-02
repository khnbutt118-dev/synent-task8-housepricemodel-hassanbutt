"""Synent Data Science Internship - Task 8: Machine Learning Model (House Price Prediction)
Author: Hassan Butt
Run: python main.py
"""

import sys, os, re, json, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


DATASETS = {'house': {'files': ['Housing.csv', 'housing.csv'],
           'kaggle': 'yasserh/housing-prices-dataset',
           'name': 'Housing Prices dataset',
           'page': 'https://www.kaggle.com/datasets/yasserh/housing-prices-dataset'}}



def _outdir(repo_dir):
    p = Path(repo_dir) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, repo_dir, name):
    fig.savefig(_outdir(repo_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def save_json(obj, repo_dir, name):
    (_outdir(repo_dir) / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_csv(path, spec):
    kw = {"encoding": spec.get("encoding", "utf-8")}
    if spec.get("no_header_file") and Path(path).name == spec["no_header_file"]:
        kw["header"] = None
        kw["names"] = spec["names"]
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def _builtin(key):
    if key == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.drop(columns=["target"])
        df["species"] = [d.target_names[i] for i in d.target]
        return df
    raise KeyError(key)


def get_data(key, repo_dir):
    """Load a dataset: local data/ folder -> built-in -> kagglehub -> public URL."""
    spec = DATASETS[key]
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        p = data_dir / name
        if p.exists():
            print(f"  [data] using local file {p}")
            return _read_csv(p, spec)
    if spec.get("builtin"):
        print("  [data] using built-in copy")
        return _builtin(key)
    if spec.get("kaggle"):
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(spec["kaggle"]))
            cands = list(folder.rglob("*.csv"))
            pick = None
            for want in spec["files"]:
                for c in cands:
                    if c.name.lower() == want.lower():
                        pick = c
                        break
                if pick:
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                dest = data_dir / pick.name
                shutil.copy(pick, dest)
                print(f"  [data] downloaded from Kaggle: {dest}")
                return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] kagglehub download not available ({type(e).__name__}: {e})")
    if spec.get("url"):
        try:
            import urllib.request
            dest = data_dir / (spec.get("no_header_file") or spec["files"][0])
            urllib.request.urlretrieve(spec["url"], dest)
            print(f"  [data] downloaded from public URL: {dest}")
            return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] public URL download failed ({type(e).__name__}: {e})")
    raise FileNotFoundError(
        f"Dataset '{spec['name']}' not found.\n"
        f"  -> Download it from {spec['page']}\n"
        f"  -> Save the CSV as one of {spec['files']} inside: {data_dir.resolve()}\n"
        f"  -> Then run the script again.")


def _to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _snake(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()


def task8_run(repo_dir):
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    df = get_data("house", repo_dir)
    df.columns = [_snake(c) for c in df.columns]
    shape = [int(df.shape[0]), int(df.shape[1])]
    df = df.drop_duplicates().dropna()
    target = "price" if "price" in df.columns else df.columns[-1]
    # Preprocessing: yes/no -> 1/0, one-hot for the other text columns
    for c in df.select_dtypes(exclude="number").columns:
        vals = set(df[c].astype(str).str.lower().unique())
        if vals <= {"yes", "no"}:
            df[c] = (df[c].astype(str).str.lower() == "yes").astype(int)
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    X, y = df.drop(columns=[target]), df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature selection (fit on training data only)
    k = min(10, X.shape[1])
    sel = SelectKBest(f_regression, k=k).fit(Xtr, ytr)
    chosen = list(X.columns[sel.get_support()])
    Xtr_s, Xte_s = Xtr[chosen], Xte[chosen]
    pd.Series(chosen, name="selected_features").to_csv(_outdir(repo_dir) / "selected_features.csv", index=False)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    rows, preds = [], {}
    for name, mdl in models.items():
        cv_r2 = cross_val_score(mdl, Xtr_s, ytr, cv=5, scoring="r2").mean()
        mdl.fit(Xtr_s, ytr)
        p = mdl.predict(Xte_s)
        preds[name] = p
        rows.append({"model": name,
                     "RMSE": round(float(np.sqrt(mean_squared_error(yte, p))), 2),
                     "MAE": round(float(mean_absolute_error(yte, p)), 2),
                     "R2": round(float(r2_score(yte, p)), 3),
                     "CV_R2_train": round(float(cv_r2), 3),
                     "Within_20pct_accuracy_%": round(float(np.mean(np.abs(p - yte) / yte <= 0.2) * 100), 1)})
    res = pd.DataFrame(rows).sort_values("RMSE")
    res.to_csv(_outdir(repo_dir) / "model_comparison.csv", index=False)
    best = res.iloc[0]["model"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, (name, p) in zip(ax, preds.items()):
        a.scatter(yte, p, alpha=0.6, s=25)
        lim = [min(yte.min(), p.min()), max(yte.max(), p.max())]
        a.plot(lim, lim, "r--")
        a.set_title(f"{name}: actual vs predicted")
        a.set_xlabel("Actual")
        a.set_ylabel("Predicted")
    fig.tight_layout()
    save_fig(fig, repo_dir, "actual_vs_predicted.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    res.set_index("model")["RMSE"].plot(kind="bar", ax=ax, color="#2980b9")
    ax.set_title("Model comparison: RMSE (lower is better)")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, repo_dir, "model_comparison_rmse.png")
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=chosen).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot(kind="barh", ax=ax, color="#16a085")
    ax.set_title("Random Forest feature importance")
    save_fig(fig, repo_dir, "feature_importance.png")
    resid = yte - preds[best]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(resid, bins=30, kde=True, ax=ax, color="#8e44ad")
    ax.set_title(f"Residuals of best model ({best})")
    save_fig(fig, repo_dir, "residuals_best_model.png")

    top3 = ", ".join(imp.sort_values(ascending=False).index[:3])
    b = res.iloc[0]
    insights = [
        f"{best} performed best on the unseen test set (RMSE {b['RMSE']:,.0f}, R2 {b['R2']}).",
        f"The most influential features were: {top3}.",
        f"About {b['Within_20pct_accuracy_%']}% of predictions fall within 20% of the real price (regression 'accuracy').",
        "Linear Regression gives an interpretable baseline, while tree-based models capture non-linear patterns.",
    ]
    return {"headline": {"Best model": best, "RMSE": f"{b['RMSE']:,.0f}", "R2": b["R2"],
                         "Within 20% accuracy": f"{b['Within_20pct_accuracy_%']}%", "Features used": len(chosen)},
            "insights": insights, "dataset_shape": shape,
            "files": ["model_comparison.csv", "actual_vs_predicted.png", "model_comparison_rmse.png",
                      "feature_importance.png", "residuals_best_model.png", "selected_features.csv"]}



if __name__ == "__main__":
    res = task8_run(Path(__file__).parent)
    print(json.dumps(res, indent=2, default=str))
