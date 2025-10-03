# tempo_predict_no2.py
# Predict next-day mean total NO2 column (troposphere + stratosphere) over an ROI using TEMPO L3 .nc files.

import os
import glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import netCDF4 as nc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------
# User settings - adjust
# -------------------------
DATA_DIR = "./tempo_files"   # folder with TEMPO .nc files (downloaded)
POI_lat = 38.0               # center latitude of region of interest
POI_lon = -96.0              # center longitude of region of interest
dlat = 5.0                   # half-height of ROI (degrees)
dlon = 6.0                   # half-width of ROI (degrees)
LAGS = [1, 2, 3, 7]          # lag-days to use as features
RANDOM_SEED = 42
# -------------------------

def read_TEMPO_NO2_L3(fn):
    """Return lat, lon, strat_NO2 (1,nlat,nlon), fv_strat, trop_NO2 (1,nlat,nlon), fv_trop, NO2_unit, QF, and granule_datetime."""
    ds = nc.Dataset(fn)
    try:
        prod = ds.groups['product']
        var = prod.variables['vertical_column_stratosphere']
        strat_NO2_column = var[:].astype(np.float64)
        fv_strat_NO2 = var.getncattr('_FillValue')
        var = prod.variables['vertical_column_troposphere']
        trop_NO2_column = var[:].astype(np.float64)
        fv_trop_NO2 = var.getncattr('_FillValue')
        NO2_unit = var.getncattr('units') if 'units' in var.ncattrs() else ''
        QF = prod.variables['main_data_quality_flag'][:]
        lat = ds.variables['latitude'][:].astype(np.float64)
        lon = ds.variables['longitude'][:].astype(np.float64)
        # Try to parse datetime from filename if present (common TEMPO filename format)
        basename = os.path.basename(fn)
        # Example filename: TEMPO_NO2_L3_V03_20240901T153815Z_S007.nc
        granule_datetime = None
        try:
            # find YYYYMMDDThhmmssZ
            parts = basename.split('_')
            for p in parts:
                if p.endswith('Z') and 'T' in p:
                    ts = p
                    granule_datetime = datetime.strptime(ts, "%Y%m%dT%H%M%SZ")
                    break
        except Exception:
            granule_datetime = None
    finally:
        ds.close()
    return lat, lon, strat_NO2_column, fv_strat_NO2, trop_NO2_column, fv_trop_NO2, NO2_unit, QF, granule_datetime

def total_column_and_mask(strat, trop, fv_strat, fv_trop, QF):
    """Return total column array and boolean mask of 'best' pixels where QF==0 and both >0 and not fill."""
    # Inputs: arrays shaped (1, nlat, nlon)
    mask_valid = (strat != fv_strat) & (trop != fv_trop)
    mask_quality = (QF == 0)
    mask_positive = (trop > 0.0) & (strat > 0.0)
    best_mask = (mask_valid & mask_quality & mask_positive)
    total = strat + trop
    return total, best_mask

def aggregate_granule_mean(fn, poi_lat, poi_lon, dlat, dlon):
    """Read granule and compute mean total NO2 over ROI for 'best' pixels. Returns (date, mean_value, count_valid)."""
    lat, lon, strat, fv_strat, trop, fv_trop, unit, QF, gdt = read_TEMPO_NO2_L3(fn)
    total, best_mask = total_column_and_mask(strat, trop, fv_strat, fv_trop, QF)
    # flatten singleton time dim:
    total2d = total[0]
    best2d = best_mask[0]
    # mask lat/lon indices for ROI
    mask_lat = (lat > poi_lat - dlat) & (lat < poi_lat + dlat)
    mask_lon = (lon > poi_lon - dlon) & (lon < poi_lon + dlon)
    if mask_lat.sum() == 0 or mask_lon.sum() == 0:
        return None  # no overlap
    sub = total2d[np.ix_(mask_lat, mask_lon)]
    sub_mask = best2d[np.ix_(mask_lat, mask_lon)]
    valid_values = sub[sub_mask]
    if valid_values.size == 0:
        return None
    mean_val = float(np.mean(valid_values))
    count_valid = int(valid_values.size)
    # Determine date string for aggregation: use granule datetime if available, else file modification time
    if gdt is None:
        gdt = datetime.utcfromtimestamp(os.path.getmtime(fn))
    date = gdt.date()
    return {"date": date, "mean_total_NO2": mean_val, "count_valid": count_valid, "unit": unit, "filename": os.path.basename(fn)}

def build_daily_series(data_dir, poi_lat, poi_lon, dlat, dlon):
    """Scan folder for .nc files and produce daily aggregated DataFrame of mean_total_NO2."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    records = []
    for fn in files:
        try:
            r = aggregate_granule_mean(fn, poi_lat, poi_lon, dlat, dlon)
            if r:
                records.append(r)
        except Exception as e:
            print(f"Warning: failed to process {fn}: {e}")
    if not records:
        raise RuntimeError("No valid granule aggregations found. Check DATA_DIR and ROI settings.")
    df = pd.DataFrame(records)
    # sometimes multiple granules per date: average them to get a daily mean
    df_daily = df.groupby("date").agg({
        "mean_total_NO2": "mean",
        "count_valid": "sum"
    }).reset_index().sort_values("date")
    return df_daily

def create_lag_features(df, lags):
    df = df.copy()
    df = df.set_index("date").asfreq('D')  # fill missing dates with NaN
    for lag in lags:
        df[f"lag_{lag}"] = df["mean_total_NO2"].shift(lag)
    # optional: also include rolling mean
    df["rolling_3"] = df["mean_total_NO2"].rolling(3, min_periods=1).mean().shift(1)
    df = df.dropna(subset=[f"lag_{lags[0]}"])  # drop rows without earliest lag
    df = df.reset_index()
    return df

def train_and_evaluate(df_features, target_col="mean_total_NO2"):
    # Prepare X, y to predict next-day value: create y = mean_total_NO2 shifted -1
    df = df_features.copy()
    df["y_next"] = df[target_col].shift(-1)
    df = df.dropna(subset=["y_next"])  # drop last row which has no next day
    feature_cols = [c for c in df.columns if c.startswith("lag_")] + ["rolling_3", "count_valid"]
    X = df[feature_cols].values
    y = df["y_next"].values
    # time-series cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    r2s = []
    fold = 1
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        print(f"Fold {fold} -> RMSE: {rmse:.3e}, R2: {r2:.3f}")
        rmses.append(rmse); r2s.append(r2)
        fold += 1
    print(f"Mean RMSE: {np.mean(rmses):.3e}, Mean R2: {np.mean(r2s):.3f}")
    # Train final model on entire dataset
    final_model = RandomForestRegressor(n_estimators=400, random_state=RANDOM_SEED)
    final_model.fit(X, y)
    return final_model, feature_cols, df

def plot_predictions(df_with_y, model, feature_cols):
    # create predictions for the available dataset (one-step-ahead using features at each row)
    X_all = df_with_y[feature_cols].values
    y_true = df_with_y["y_next"].values
    y_pred = model.predict(X_all)
    dates = df_with_y["date"].values
    plt.figure(figsize=(10,4))
    plt.plot(dates, y_true, label="True next-day mean NO2", marker='o')
    plt.plot(dates, y_pred, label="Predicted next-day mean NO2", marker='x')
    plt.xlabel("Date")
    plt.ylabel("Mean total NO2 (molecules/cm^2)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------
# Main pipeline
# -------------------------
if __name__ == "__main__":
    # 1) Build daily series from granules
    print("Scanning folder and aggregating granules into daily means...")
    df_daily = build_daily_series(DATA_DIR, POI_lat, POI_lon, dlat, dlon)
    print(f"Found {len(df_daily)} daily aggregated records.")
    print(df_daily.head())

    # 2) Build lag features
    df_features = create_lag_features(df_daily, LAGS)
    print("Feature dataframe sample:")
    print(df_features.head())

    # 3) Train + evaluate
    model, feature_cols, df_model_ready = train_and_evaluate(df_features)

    # 4) Plot predictions (on historical data)
    plot_predictions(df_model_ready, model, feature_cols)

    # 5) Example: Predict next day using the latest available row
    latest_row = df_model_ready.iloc[-1]
    X_latest = latest_row[feature_cols].values.reshape(1, -1)
    next_day_prediction = model.predict(X_latest)[0]
    next_day_date = latest_row["date"] + timedelta(days=1)
    print(f"Predicted mean total NO2 for {next_day_date} is {next_day_prediction:.3e} {latest_row.get('unit','molecules/cm^2')}")
    # Save model if desired (requires joblib)
    try:
        import joblib
        joblib.dump(model, "tempo_no2_nextday_model.joblib")
        print("Saved model to tempo_no2_nextday_model.joblib")
    except Exception:
        print("joblib not available; skipping model save.")
