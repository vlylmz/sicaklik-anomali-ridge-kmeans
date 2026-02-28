# veri_Madeni.py
import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import (
    r2_score, explained_variance_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ---------------- Helpers ----------------
def parse_coord(s: str) -> float:
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([NSEW])$", s, re.IGNORECASE)
    if not m:
        return np.nan
    val = float(m.group(1))
    hemi = m.group(2).upper()
    return -val if hemi in ("S", "W") else val


def load_city_data(path="data/GlobalLandTemperaturesByMajorCity.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.dropna(subset=["dt", "AverageTemperature", "City", "Country"])
    df["year"] = df["dt"].dt.year.astype(int)
    df["month"] = df["dt"].dt.month.astype(int)

    if "Latitude" in df.columns and "Longitude" in df.columns:
        df["lat"] = df["Latitude"].map(parse_coord)
        df["lon"] = df["Longitude"].map(parse_coord)
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan

    keep = ["dt", "year", "month", "City", "Country", "lat", "lon", "AverageTemperature"]
    df = df[keep].copy()
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def aggregate_city_month(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["City", "Country", "year", "month"], as_index=False)
          .agg(avg_temp=("AverageTemperature", "mean"),
               lat=("lat", "median"),
               lon=("lon", "median"))
    )
    agg["date_key"] = agg["year"] * 100 + agg["month"]
    agg = agg.sort_values(["City", "Country", "date_key"]).reset_index(drop=True)
    return agg


def add_city_month_climatology(agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clim = (
        agg.groupby(["City", "Country", "month"], as_index=False)["avg_temp"]
           .mean()
           .rename(columns={"avg_temp": "clim_temp"})
    )
    out = agg.merge(clim, on=["City", "Country", "month"], how="left")
    out["anomaly"] = out["avg_temp"] - out["clim_temp"]
    return out, clim


def add_cyclic_and_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["years_since_1970"] = df["year"] - 1970
    return df


def add_lag_features(df_anom: pd.DataFrame, train_cutoff_date: int | None = None) -> pd.DataFrame:
    """
    Leakage-safe lag:
    - train_cutoff_date verilirse, cutoff SONRASI anomaly NaN yapılıp lag sadece train’den türetilir.
    """
    df = df_anom.sort_values(["City", "Country", "date_key"]).copy()

    if train_cutoff_date is not None:
        tmp = df.copy()
        tmp.loc[tmp["date_key"] > train_cutoff_date, "anomaly"] = np.nan
        g = tmp.groupby(["City", "Country"])["anomaly"]
    else:
        g = df.groupby(["City", "Country"])["anomaly"]

    df["anom_lag_1"] = g.shift(1)
    df["anom_lag_12"] = g.shift(12)
    df["anom_roll_12"] = g.shift(1).rolling(12, min_periods=1).mean()
    return df


# ---------------- Time folds ----------------
def expanding_time_folds(unique_dates: np.ndarray, n_splits: int = 5):
    unique_dates = np.array(sorted(unique_dates))
    blocks = np.array_split(unique_dates, n_splits + 1)
    folds = []
    for i in range(1, n_splits + 1):
        train_dates = np.concatenate(blocks[:i])
        test_dates = blocks[i]
        if len(train_dates) and len(test_dates):
            folds.append((set(train_dates.tolist()), set(test_dates.tolist())))
    return folds


# ---------------- Regression pipeline ----------------
def feature_columns():
    return [
        "year", "lat", "lon",
        "month_sin", "month_cos",
        "years_since_1970",
        "anom_lag_1", "anom_lag_12", "anom_roll_12",
        "City", "Country"
    ]


def build_anomaly_pipeline(alpha: float = 1.0):
    num_cols = [
        "year", "lat", "lon",
        "month_sin", "month_cos",
        "years_since_1970",
        "anom_lag_1", "anom_lag_12", "anom_roll_12"
    ]
    cat_cols = ["City", "Country"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = Ridge(
        alpha=float(alpha),
        random_state=42,
        solver="sparse_cg",
        max_iter=2000,
        tol=1e-4
    )
    return Pipeline([("pre", pre), ("model", model)])


def train_and_evaluate_anomaly_model(df_base: pd.DataFrame, n_splits=5, alpha: float = 1.0):
    unique_dates = df_base["date_key"].unique()
    folds = expanding_time_folds(unique_dates, n_splits=n_splits)

    mae_list, rmse_list = [], []
    pipe = build_anomaly_pipeline(alpha=alpha)

    for train_dates, test_dates in folds:
        cutoff = max(train_dates)
        df_fold = add_lag_features(df_base, train_cutoff_date=cutoff)

        train_mask = df_fold["date_key"].isin(train_dates)
        test_mask = df_fold["date_key"].isin(test_dates)

        X = df_fold[feature_columns()].copy()
        y = df_fold["anomaly"].values

        X_train, y_train = X.loc[train_mask], y[train_mask.values]
        X_test, y_test = X.loc[test_mask], y[test_mask.values]

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        mae_list.append(mean_absolute_error(y_test, pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, pred)))

    df_full = add_lag_features(df_base, train_cutoff_date=None)
    pipe.fit(df_full[feature_columns()], df_full["anomaly"].values)

    metrics = {
        "rows_used": int(len(df_base)),
        "folds": int(len(folds)),
        "anomaly_mae_mean": float(np.mean(mae_list)),
        "anomaly_mae_std": float(np.std(mae_list)),
        "anomaly_rmse_mean": float(np.mean(rmse_list)),
        "anomaly_rmse_std": float(np.std(rmse_list)),
        "alpha": float(alpha),
    }
    return pipe, metrics, df_full


# ---------------- Backtest + Predictions ----------------
def baseline_mae_anomaly(df_anom: pd.DataFrame) -> float:
    return float(np.mean(np.abs(df_anom["anomaly"].values)))


def holdout_backtest(df_base: pd.DataFrame, quantile: float = 0.8,
                     focus_cities: list[tuple[str, str]] | None = None,
                     alpha: float = 1.0):
    if focus_cities is None:
        focus_cities = [("Ankara", "Turkey"), ("Istanbul", "Turkey")]

    unique_dates = np.array(sorted(df_base["date_key"].unique()))
    cut_idx = int(len(unique_dates) * quantile) - 1
    cut_idx = max(0, min(cut_idx, len(unique_dates) - 2))
    cutoff = unique_dates[cut_idx]

    df_bt = add_lag_features(df_base, train_cutoff_date=int(cutoff))
    train = df_bt[df_bt["date_key"] <= cutoff].copy()
    test = df_bt[df_bt["date_key"] > cutoff].copy()

    pipe = build_anomaly_pipeline(alpha=alpha)
    pipe.fit(train[feature_columns()], train["anomaly"].values)
    pred_anom = pipe.predict(test[feature_columns()])

    base_mae = float(np.mean(np.abs(test["anomaly"].values)))
    model_mae = float(mean_absolute_error(test["anomaly"].values, pred_anom))
    model_rmse = float(np.sqrt(mean_squared_error(test["anomaly"].values, pred_anom)))

    rows = [{
        "scope": "GLOBAL_TEST",
        "train_cutoff_date_key": int(cutoff),
        "test_rows": int(len(test)),
        "baseline_mae": base_mae,
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "improvement_mae": base_mae - model_mae
    }]

    test = test.reset_index(drop=True)
    pred_s = pd.Series(pred_anom)

    for city, country in focus_cities:
        mask = (test["City"] == city) & (test["Country"] == country)
        if mask.sum() == 0:
            rows.append({
                "scope": f"{city},{country}",
                "train_cutoff_date_key": int(cutoff),
                "test_rows": 0,
                "baseline_mae": np.nan,
                "model_mae": np.nan,
                "model_rmse": np.nan,
                "improvement_mae": np.nan
            })
            continue

        y_c = test.loc[mask, "anomaly"].values
        p_c = pred_s.loc[mask.index[mask]].values
        rows.append({
            "scope": f"{city},{country}",
            "train_cutoff_date_key": int(cutoff),
            "test_rows": int(mask.sum()),
            "baseline_mae": float(np.mean(np.abs(y_c))),
            "model_mae": float(mean_absolute_error(y_c, p_c)),
            "model_rmse": float(np.sqrt(mean_squared_error(y_c, p_c))),
            "improvement_mae": float(np.mean(np.abs(y_c)) - mean_absolute_error(y_c, p_c))
        })

    return pd.DataFrame(rows), int(cutoff), df_bt


def make_test_predictions(df_bt: pd.DataFrame, cutoff: int, alpha: float = 1.0):
    train = df_bt[df_bt["date_key"] <= cutoff].copy()
    test = df_bt[df_bt["date_key"] > cutoff].copy()

    pipe = build_anomaly_pipeline(alpha=alpha)
    pipe.fit(train[feature_columns()], train["anomaly"].values)

    pred_anom = pipe.predict(test[feature_columns()])
    out = test.copy()
    out["pred_anom"] = pred_anom
    out["pred_temp"] = out["clim_temp"].values + out["pred_anom"].values
    out["baseline_temp"] = out["clim_temp"].values
    out["true_temp"] = out["avg_temp"].values
    out["residual"] = out["pred_temp"].values - out["true_temp"].values
    return out


def quick_alpha_tune(df_base: pd.DataFrame,
                     alphas=(0.1, 0.5, 1.0, 2.0, 5.0),
                     quantile=0.8,
                     sample_frac=0.25):
    unique_dates = np.array(sorted(df_base["date_key"].unique()))
    cut_idx = int(len(unique_dates) * quantile) - 1
    cut_idx = max(0, min(cut_idx, len(unique_dates) - 2))
    cutoff = int(unique_dates[cut_idx])

    df_bt = add_lag_features(df_base, train_cutoff_date=cutoff)
    train = df_bt[df_bt["date_key"] <= cutoff].copy()
    valid = df_bt[df_bt["date_key"] > cutoff].copy()

    if sample_frac < 1.0:
        train = train.sample(frac=sample_frac, random_state=42)

    Xtr = train[feature_columns()]
    ytr = train["anomaly"].values
    Xva = valid[feature_columns()]
    yva = valid["anomaly"].values

    best = None
    for a in alphas:
        pipe = build_anomaly_pipeline(alpha=float(a))
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        mae = float(mean_absolute_error(yva, pred))
        print(f"alpha={a}  valid_MAE={mae:.4f}")
        if best is None or mae < best[1]:
            best = (a, mae)

    print("BEST alpha:", best[0], "MAE:", best[1])
    return float(best[0])


def create_model_plots(test_df: pd.DataFrame, cutoff: int,
                       city_focus=("Ankara", "Turkey"),
                       sample_n: int = 30000,
                       out_dir: str = "plots"):
    os.makedirs(out_dir, exist_ok=True)

    # 1) MAE by year
    mae_year = (
        test_df.assign(
            abs_err_model=lambda d: (d["pred_temp"] - d["true_temp"]).abs(),
            abs_err_base=lambda d: (d["baseline_temp"] - d["true_temp"]).abs()
        )
        .groupby("year", as_index=False)[["abs_err_model", "abs_err_base"]]
        .mean()
        .rename(columns={"abs_err_model": "mae_model", "abs_err_base": "mae_baseline"})
        .sort_values("year")
    )

    plt.figure()
    plt.plot(mae_year["year"], mae_year["mae_baseline"], label="Baseline (climatology)")
    plt.plot(mae_year["year"], mae_year["mae_model"], label="Model (Ridge + lags)")
    plt.title(f"Holdout Test MAE by Year (cutoff={cutoff})")
    plt.xlabel("Year"); plt.ylabel("MAE (°C)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_mae_by_year.png"), dpi=200)
    plt.close()

    # 2) Scatter
    s = test_df.sample(n=min(sample_n, len(test_df)), random_state=42)
    plt.figure()
    plt.scatter(s["true_temp"], s["pred_temp"], s=5, alpha=0.3)
    mn = float(min(s["true_temp"].min(), s["pred_temp"].min()))
    mx = float(max(s["true_temp"].max(), s["pred_temp"].max()))
    plt.plot([mn, mx], [mn, mx])
    plt.title("Actual vs Predicted Temperature (Holdout sample)")
    plt.xlabel("Actual (°C)"); plt.ylabel("Predicted (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_actual_vs_pred_scatter.png"), dpi=200)
    plt.close()

    # 3) Residual hist
    r = test_df["residual"].dropna()
    r = r.sample(n=min(sample_n, len(r)), random_state=42)
    plt.figure()
    plt.hist(r, bins=60)
    plt.title("Residual Distribution (Pred - Actual) (Holdout sample)")
    plt.xlabel("Residual (°C)"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_residual_hist.png"), dpi=200)
    plt.close()

    # 4) City time series
    city, country = city_focus
    c = test_df[(test_df["City"] == city) & (test_df["Country"] == country)].copy()
    c = c.sort_values("date_key")
    if len(c) > 0:
        x = pd.to_datetime(c["year"].astype(str) + "-" + c["month"].astype(str) + "-01")
        plt.figure()
        plt.plot(x, c["true_temp"], label="Actual")
        plt.plot(x, c["baseline_temp"], label="Baseline")
        plt.plot(x, c["pred_temp"], label="Model")
        plt.title(f"{city}, {country} - Holdout Period (Actual vs Pred)")
        plt.xlabel("Date"); plt.ylabel("Avg Temp (°C)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"04_{city}_{country}_timeseries.png"), dpi=200)
        plt.close()


# ---------------- Correlation Matrix ----------------
def create_correlation_matrix_plot(df: pd.DataFrame,
                                   cols: list[str],
                                   sample_n: int = 200000,
                                   out_dir: str = "plots",
                                   filename: str = "00_correlation_matrix.png"):
    os.makedirs(out_dir, exist_ok=True)

    use_cols = [c for c in cols if c in df.columns]
    d = df[use_cols].copy()
    d = d.select_dtypes(include=[np.number])

    if d.shape[1] < 2:
        print("Correlation matrix skipped: not enough numeric columns.")
        return

    if len(d) > sample_n:
        d = d.sample(n=sample_n, random_state=42)

    d = d.dropna()
    if len(d) < 10:
        print("Correlation matrix skipped: too few rows after dropna.")
        return

    corr = d.corr()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr.values, aspect="auto")
    plt.title("Correlation Matrix (sampled)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=200)
    plt.close()


# ---------------- Confusion Matrix helpers ----------------
def _temp_bin_series(s: pd.Series, q=(1/3, 2/3)):
    s = s.dropna().astype(float)
    t1, t2 = np.quantile(s.values, [q[0], q[1]])
    return float(t1), float(t2)

def _temp_bin_value(v: float, t1: float, t2: float) -> str:
    if pd.isna(v):
        return np.nan
    if v < t1:
        return "cold"
    elif v < t2:
        return "mild"
    else:
        return "hot"

def save_confusion_plots(y_true, y_pred, labels, title, out_path, normalize=False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cm = confusion_matrix(
        y_true, y_pred,
        labels=labels,
        normalize=("true" if normalize else None)
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots()
    disp.plot(
        ax=ax,
        colorbar=False,
        values_format=(".2f" if normalize else "d")
    )
    ax.set_title(title + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def create_prediction_confusion_matrices(test_df: pd.DataFrame, out_dir: str = "plots"):
    """
    Regresyon (true_temp vs pred_temp) çıktısını 3 sınıfa (cold/mild/hot) çevirip confusion matrix üretir.
    """
    os.makedirs(out_dir, exist_ok=True)

    d = test_df[["true_temp", "pred_temp"]].dropna().copy()
    if len(d) == 0:
        print("Prediction CM skipped: no rows.")
        return

    t1, t2 = _temp_bin_series(d["true_temp"], q=(1/3, 2/3))
    d["true_bin"] = d["true_temp"].apply(lambda v: _temp_bin_value(v, t1, t2))
    d["pred_bin"] = d["pred_temp"].apply(lambda v: _temp_bin_value(v, t1, t2))

    labels = ["cold", "mild", "hot"]
    title = f"Temp Category Confusion (true terciles)  t1={t1:.2f}, t2={t2:.2f}"

    save_confusion_plots(
        d["true_bin"], d["pred_bin"], labels, title,
        out_path=os.path.join(out_dir, "07_temp_bin_confusion.png"),
        normalize=False
    )
    save_confusion_plots(
        d["true_bin"], d["pred_bin"], labels, title,
        out_path=os.path.join(out_dir, "08_temp_bin_confusion_normalized.png"),
        normalize=True
    )


# ---------------- Clustering + Plots ----------------
def cluster_cities_for_period(raw_df: pd.DataFrame, year: int, month: int, k: int = 3):
    if k < 2:
        raise ValueError("k en az 2 olmalıdır.")

    d = raw_df[(raw_df["year"] == year) & (raw_df["month"] == month)].copy()
    if d.empty:
        raise ValueError(f"Bu yıl/ay ({year}/{month}) için veri yok.")

    agg = (
        d.groupby(["City", "Country"], as_index=False)
         .agg(lat=("lat", "median"), lon=("lon", "median"), avg_temp=("AverageTemperature", "mean"))
         .dropna(subset=["avg_temp"])
    )

    feats = agg[["lat", "lon", "avg_temp"]].dropna().reset_index(drop=True)
    agg = agg.loc[feats.index].reset_index(drop=True)

    scaler = StandardScaler()
    Z = scaler.fit_transform(feats.values)

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(Z)
    agg["cluster"] = labels

    cluster_means = agg.groupby("cluster")["avg_temp"].mean().sort_values()
    if k == 3:
        name_map = {cluster_means.index[0]: "cold", cluster_means.index[1]: "mild", cluster_means.index[2]: "hot"}
    else:
        name_map = {cluster_means.index[i]: f"temp_rank_{i+1}_of_{k}" for i in range(k)}
    agg["cluster_name"] = agg["cluster"].map(name_map)

    info = {"year": year, "month": month, "k": k, "n_cities": int(len(agg))}
    return agg, info


def create_clustering_plots(clustered_df: pd.DataFrame, year: int, month: int, out_dir: str = "plots"):
    os.makedirs(out_dir, exist_ok=True)

    order = ["cold", "mild", "hot"] if set(["cold","mild","hot"]).issubset(set(clustered_df["cluster_name"].unique())) else None
    counts = clustered_df["cluster_name"].value_counts()
    if order:
        counts = counts.reindex(order)

    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"City Counts by Cluster ({year}-{month:02d})")
    plt.xlabel("Cluster"); plt.ylabel("Number of Cities")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"11_cluster_counts_{year}_{month:02d}.png"), dpi=200)
    plt.close()

    labels = order if order else sorted(clustered_df["cluster_name"].unique())
    data = [clustered_df.loc[clustered_df["cluster_name"] == lab, "avg_temp"].dropna().values for lab in labels]
    plt.figure()
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title(f"Avg Temperature Distribution by Cluster ({year}-{month:02d})")
    plt.xlabel("Cluster"); plt.ylabel("Average Temperature (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"12_cluster_temp_boxplot_{year}_{month:02d}.png"), dpi=200)
    plt.close()

    plt.figure()
    for lab in labels:
        d = clustered_df[clustered_df["cluster_name"] == lab]
        plt.scatter(d["lon"], d["lat"], s=6, alpha=0.4, label=str(lab))
    plt.title(f"City Locations by Cluster ({year}-{month:02d})")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"13_cluster_latlon_scatter_{year}_{month:02d}.png"), dpi=200)
    plt.close()

    prof = (clustered_df.groupby("cluster_name", as_index=False)
            .agg(mean_temp=("avg_temp", "mean"), mean_lat=("lat", "mean"), mean_lon=("lon", "mean"), n=("City", "count")))
    prof.to_csv(os.path.join(out_dir, f"cluster_profile_{year}_{month:02d}.csv"), index=False)

    plt.figure()
    plt.bar(prof["cluster_name"].astype(str), prof["mean_temp"].values)
    plt.title(f"Cluster Mean Temperature ({year}-{month:02d})")
    plt.xlabel("Cluster"); plt.ylabel("Mean Avg Temp (°C)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"14_cluster_mean_temp_{year}_{month:02d}.png"), dpi=200)
    plt.close()

    # --- NEW: Clustering confusion matrix vs temp terciles (pseudo-label) ---
    dcm = clustered_df[["avg_temp", "cluster_name"]].dropna().copy()
    if len(dcm) > 0:
        t1, t2 = _temp_bin_series(dcm["avg_temp"], q=(1/3, 2/3))
        dcm["temp_bin"] = dcm["avg_temp"].apply(lambda v: _temp_bin_value(v, t1, t2))

        cm_labels = ["cold", "mild", "hot"]
        title = f"Clustering vs Temp-Terciles ({year}-{month:02d}) t1={t1:.2f}, t2={t2:.2f}"

        save_confusion_plots(
            dcm["temp_bin"], dcm["cluster_name"], cm_labels, title,
            out_path=os.path.join(out_dir, f"16_cluster_vs_tempbin_confusion_{year}_{month:02d}.png"),
            normalize=False
        )
        save_confusion_plots(
            dcm["temp_bin"], dcm["cluster_name"], cm_labels, title,
            out_path=os.path.join(out_dir, f"17_cluster_vs_tempbin_confusion_norm_{year}_{month:02d}.png"),
            normalize=True
        )


def clustering_elbow_plot(raw_df: pd.DataFrame, year: int, month: int, k_min=2, k_max=10, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    d = raw_df[(raw_df["year"] == year) & (raw_df["month"] == month)].copy()
    agg = (
        d.groupby(["City", "Country"], as_index=False)
         .agg(lat=("lat", "median"), lon=("lon", "median"), avg_temp=("AverageTemperature", "mean"))
         .dropna(subset=["avg_temp"])
    )
    feats = agg[["lat", "lon", "avg_temp"]].dropna().reset_index(drop=True)

    scaler = StandardScaler()
    Z = scaler.fit_transform(feats.values)

    ks, inertias = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        km.fit(Z)
        ks.append(k)
        inertias.append(km.inertia_)

    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.title(f"Elbow Method (Inertia) {year}-{month:02d}")
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"15_elbow_{year}_{month:02d}.png"), dpi=200)
    plt.close()


# ---------------- Auto-lags prediction (pipeline uyumlu) ----------------
def predict_with_auto_lags(pipe, df_full_with_lags: pd.DataFrame, clim: pd.DataFrame,
                           city: str, country: str, year: int, month: int,
                           lat=None, lon=None) -> float:
    row = clim[(clim["City"] == city) & (clim["Country"] == country) & (clim["month"] == month)]
    if row.empty:
        raise ValueError("Climatology bulunamadı.")
    clim_temp = float(row["clim_temp"].iloc[0])

    date_key = int(year) * 100 + int(month)

    hist = df_full_with_lags[(df_full_with_lags["City"] == city) & (df_full_with_lags["Country"] == country)].copy()
    hist = hist.sort_values("date_key")

    prev = hist[hist["date_key"] < date_key].tail(1)
    prev12_key = (int(year) - 1) * 100 + int(month)
    prev12 = hist[hist["date_key"] == prev12_key].tail(1)
    roll = hist[hist["date_key"] < date_key].tail(12)

    anom_lag_1 = float(prev["anomaly"].iloc[0]) if len(prev) else np.nan
    anom_lag_12 = float(prev12["anomaly"].iloc[0]) if len(prev12) else np.nan
    anom_roll_12 = float(roll["anomaly"].mean()) if len(roll) else np.nan

    x = pd.DataFrame([{
        "year": int(year),
        "lat": np.nan if lat is None else float(lat),
        "lon": np.nan if lon is None else float(lon),
        "month_sin": np.sin(2 * np.pi * int(month) / 12),
        "month_cos": np.cos(2 * np.pi * int(month) / 12),
        "years_since_1970": int(year) - 1970,
        "anom_lag_1": anom_lag_1,
        "anom_lag_12": anom_lag_12,
        "anom_roll_12": anom_roll_12,
        "City": city,
        "Country": country,
    }])

    pred_anom = float(pipe.predict(x)[0])
    return clim_temp + pred_anom


def print_extra_metrics(df_base: pd.DataFrame,
                        quantile: float = 0.8,
                        alpha: float = 1.0,
                        event_threshold_true: float = 1.0,
                        focus_cities: list[tuple[str, str]] | None = None):
    """
    Holdout test üzerinde:
      - Regression: R2, ExplainedVariance, MAE, RMSE (anomali + mutlak sıcaklık)
      - Event classification: True event |anomaly|>=event_threshold_true
        Pred threshold otomatik kalibre edilir (pred event rate ~= true event rate)
      - Confusion matrix plotlar plots/ altına kaydedilir
    """
    if focus_cities is None:
        focus_cities = [("Ankara", "Turkey"), ("Istanbul", "Turkey")]

    unique_dates = np.array(sorted(df_base["date_key"].unique()))
    cut_idx = int(len(unique_dates) * quantile) - 1
    cut_idx = max(0, min(cut_idx, len(unique_dates) - 2))
    cutoff = int(unique_dates[cut_idx])

    df_bt = add_lag_features(df_base, train_cutoff_date=cutoff)
    train = df_bt[df_bt["date_key"] <= cutoff].copy()
    test = df_bt[df_bt["date_key"] > cutoff].copy()

    pipe = build_anomaly_pipeline(alpha=alpha)
    pipe.fit(train[feature_columns()], train["anomaly"].values)

    y_true_anom = test["anomaly"].values
    y_pred_anom = pipe.predict(test[feature_columns()])

    y_true_temp = test["avg_temp"].values
    y_pred_temp = test["clim_temp"].values + y_pred_anom

    reg_anom = {
        "R2": float(r2_score(y_true_anom, y_pred_anom)),
        "ExplVar": float(explained_variance_score(y_true_anom, y_pred_anom)),
        "MAE": float(mean_absolute_error(y_true_anom, y_pred_anom)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_anom, y_pred_anom))),
    }
    reg_temp = {
        "R2": float(r2_score(y_true_temp, y_pred_temp)),
        "ExplVar": float(explained_variance_score(y_true_temp, y_pred_temp)),
        "MAE": float(mean_absolute_error(y_true_temp, y_pred_temp)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_temp, y_pred_temp))),
    }

    y_true_evt = (np.abs(y_true_anom) >= float(event_threshold_true)).astype(int)
    true_rate = float(y_true_evt.mean())
    true_rate = min(max(true_rate, 1e-6), 1 - 1e-6)
    pred_thr = float(np.quantile(np.abs(y_pred_anom), 1 - true_rate))
    y_pred_evt = (np.abs(y_pred_anom) >= pred_thr).astype(int)

    cls_evt = {
        "Accuracy": float(accuracy_score(y_true_evt, y_pred_evt)),
        "Precision": float(precision_score(y_true_evt, y_pred_evt, zero_division=0)),
        "Recall": float(recall_score(y_true_evt, y_pred_evt, zero_division=0)),
        "F1": float(f1_score(y_true_evt, y_pred_evt, zero_division=0)),
        "TrueEventRate": float(y_true_evt.mean()),
        "PredEventRate": float(y_pred_evt.mean()),
        "PredThreshold(|anom|)": pred_thr,
    }

    os.makedirs("plots", exist_ok=True)

    cm = confusion_matrix(y_true_evt, y_pred_evt, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No event", "Event"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(
        f"Confusion Matrix - Event Detection (|anom|>={event_threshold_true}°C)\n"
        f"cutoff={cutoff}, alpha={alpha}, pred_thr={pred_thr:.3f}"
    )
    plt.tight_layout()
    plt.savefig("plots/05_confusion_matrix_event.png", dpi=200)
    plt.close()

    cmn = confusion_matrix(y_true_evt, y_pred_evt, labels=[0, 1], normalize="true")
    dispn = ConfusionMatrixDisplay(cmn, display_labels=["No event", "Event"])
    fig, ax = plt.subplots()
    dispn.plot(ax=ax, colorbar=False, values_format=".2f")
    ax.set_title(
        "Confusion Matrix (Normalized) - Event Detection\n"
        f"cutoff={cutoff}, alpha={alpha}"
    )
    plt.tight_layout()
    plt.savefig("plots/06_confusion_matrix_event_normalized.png", dpi=200)
    plt.close()

    print("\n=== EXTRA METRICS (HOLDOUT TEST) ===")
    print(f"cutoff={cutoff}  alpha={alpha}")
    print(f"true_event: |anomaly| >= {event_threshold_true}°C  (pred threshold auto-calibrated)")

    print("\n[Regression - Anomaly]")
    for k, v in reg_anom.items():
        print(f"  {k:>8}: {v:.6f}")

    print("\n[Regression - Absolute Temperature]")
    for k, v in reg_temp.items():
        print(f"  {k:>8}: {v:.6f}")

    print("\n[Event Classification]")
    for k, v in cls_evt.items():
        print(f"  {k:>16}: {v:.6f}")

    print("\n[City Focus]")
    test = test.reset_index(drop=True)
    pred_s = pd.Series(y_pred_anom)
    for city, country in focus_cities:
        m = (test["City"] == city) & (test["Country"] == country)
        if m.sum() == 0:
            print(f"  {city},{country}: no rows")
            continue
        yt = test.loc[m, "anomaly"].values
        yp = pred_s.loc[m.index[m]].values
        print(f"  {city},{country}  n={int(m.sum())}  MAE={mean_absolute_error(yt, yp):.4f}")


# ---------------- MAIN ----------------
def main():
    raw = load_city_data("data/GlobalLandTemperaturesByMajorCity.csv")
    agg = aggregate_city_month(raw)
    df_base, clim = add_city_month_climatology(agg)

    df_base = add_cyclic_and_trend_features(df_base)

    # İstersen alpha tune aç:
    # alpha = quick_alpha_tune(df_base, alphas=(0.1, 0.5, 1.0, 2.0, 5.0), sample_frac=0.25)
    alpha = 1.0

    model, metrics, df_full = train_and_evaluate_anomaly_model(df_base, n_splits=5, alpha=alpha)
    print("Anomaly regression metrics:", metrics)

    joblib.dump(model, "temperature_anomaly_ridge.joblib")
    joblib.dump(clim, "city_month_climatology.joblib")
    print("Saved: temperature_anomaly_ridge.joblib")
    print("Saved: city_month_climatology.joblib")

    print("Baseline MAE (always 0 anomaly, ALL DATA):", baseline_mae_anomaly(df_base))

    report, cutoff, df_bt = holdout_backtest(
        df_base, quantile=0.8,
        focus_cities=[("Ankara", "Turkey"), ("Istanbul", "Turkey")],
        alpha=alpha
    )
    print("\n=== BACKTEST REPORT (time holdout) ===")
    print(report.to_string(index=False))
    report.to_csv("backtest_report.csv", index=False)
    print("Saved: backtest_report.csv")

    # Correlation matrix
    corr_cols = [
        "year", "month", "lat", "lon",
        "month_sin", "month_cos", "years_since_1970",
        "avg_temp", "clim_temp", "anomaly",
        "anom_lag_1", "anom_lag_12", "anom_roll_12"
    ]
    create_correlation_matrix_plot(
        df_bt, cols=corr_cols, sample_n=200000,
        out_dir="plots", filename="00_correlation_matrix.png"
    )
    print("Saved: plots/00_correlation_matrix.png")

    # Predictions + plots
    test_df = make_test_predictions(df_bt, cutoff=cutoff, alpha=alpha)
    create_model_plots(test_df, cutoff=cutoff, city_focus=("Ankara", "Turkey"), out_dir="plots")
    print("Saved model plots into ./plots/")

    # NEW: prediction confusion matrices (true_temp vs pred_temp -> cold/mild/hot)
    create_prediction_confusion_matrices(test_df, out_dir="plots")
    print("Saved: plots/07_temp_bin_confusion.png")
    print("Saved: plots/08_temp_bin_confusion_normalized.png")

    # Clustering + plots
    year, month = 2010, 7
    clustered, info = cluster_cities_for_period(raw, year=year, month=month, k=3)
    print("Clustering info:", info)
    clustered.to_csv(f"clusters_{year}_{month}.csv", index=False)
    print(f"Saved: clusters_{year}_{month}.csv")

    create_clustering_plots(clustered, year=year, month=month, out_dir="plots")
    clustering_elbow_plot(raw, year=year, month=month, k_min=2, k_max=10, out_dir="plots")
    print("Saved clustering plots + elbow into ./plots/")
    print(f"Saved: plots/16_cluster_vs_tempbin_confusion_{year}_{month:02d}.png")
    print(f"Saved: plots/17_cluster_vs_tempbin_confusion_norm_{year}_{month:02d}.png")

    # Event confusion matrices (anomali -> event / no-event)
    print_extra_metrics(
        df_base=df_base,
        quantile=0.8,
        alpha=alpha,
        event_threshold_true=1.0,
        focus_cities=[("Ankara", "Turkey"), ("Istanbul", "Turkey")]
    )
    print("Saved: plots/05_confusion_matrix_event.png")
    print("Saved: plots/06_confusion_matrix_event_normalized.png")

    # Example forward prediction
    try:
        pred = predict_with_auto_lags(
            pipe=model,
            df_full_with_lags=df_full,
            clim=clim,
            city="Ankara",
            country="Turkey",
            year=2030,
            month=7,
            lat=39.93,
            lon=32.86,
        )
        print("Example prediction w/auto-lags (Ankara, 2030-07):", pred)
    except Exception as e:
        print("Example prediction skipped:", e)


if __name__ == "__main__":
    main()
