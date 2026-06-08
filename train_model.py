"""
Maize Price Prediction Model
==============================
Uses merged maize + weather dataset to train and compare:
  1. Random Forest
  2. Gradient Boosting (XGBoost-equivalent, swap in XGBoost below)

Target variable : Wholesale price (KSh per kg)
Run             : python train_model.py
Outputs         : model_results.png, feature_importance.png, predictions.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# ── Optional: swap in XGBoost if available on your machine ──────────────────
try:
    from xgboost import XGBRegressor
    USE_XGBOOST = True
    print("✅ XGBoost found — will use XGBRegressor")
except ImportError:
    USE_XGBOOST = False
    print("ℹ️  XGBoost not installed — using GradientBoostingRegressor instead")
    print("   Install with: pip install xgboost")
# ────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("\n📂 Loading dataset...")
df = pd.read_csv("merged_maize_weather.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Remove extreme outliers (prices above 500 KSh — only 5 records)
df = df[df['Wholesale'] <= 500].copy()
df.reset_index(drop=True, inplace=True)

print(f"   Records after cleaning : {len(df)}")
print(f"   Date range             : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"   Counties               : {df['County'].nunique()}")
print(f"   Target (Wholesale) mean: KSh {df['Wholesale'].mean():.2f}")


# ─────────────────────────────────────────
# 2. DEFINE FEATURES AND TARGET
# ─────────────────────────────────────────
FEATURES = [
    # Lag price features (most important)
    'price_lag_7d',
    'price_lag_14d',
    'price_lag_30d',
    'price_change_7d',

    # Weather features
    'temperature_mean_c',
    'rain_sum_mm',
    'rain_7d_rolling',
    'rain_30d_rolling',
    'temp_7d_rolling',

    # Time features
    'month',
    'week',
    'year',
    'day_of_year',

    # Categorical (encoded)
    'season_encoded',
    'county_encoded',
    'classification_encoded',

    # Supply
    'Supply Volume',
]

TARGET = 'Wholesale'

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\n🔧 Features used: {len(FEATURES)}")
print(f"   Target        : {TARGET}")


# ─────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# Time-based split — train on earlier dates, test on recent ones
# This is more realistic than random split for time-series data
# ─────────────────────────────────────────
df_sorted = df.sort_values('Date').reset_index(drop=True)
X_sorted  = df_sorted[FEATURES]
y_sorted  = df_sorted[TARGET]

split_idx  = int(len(df_sorted) * 0.8)   # 80% train, 20% test
X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
dates_test = df_sorted['Date'].iloc[split_idx:]

print(f"\n📊 Train/test split (time-based 80/20):")
print(f"   Train : {len(X_train)} records  ({df_sorted['Date'].iloc[0].date()} → {df_sorted['Date'].iloc[split_idx-1].date()})")
print(f"   Test  : {len(X_test)}  records  ({df_sorted['Date'].iloc[split_idx].date()} → {df_sorted['Date'].iloc[-1].date()})")


# ─────────────────────────────────────────
# 4. DEFINE MODELS
# ─────────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
}

# Add XGBoost if available
if USE_XGBOOST:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )


# ─────────────────────────────────────────
# 5. TRAIN, EVALUATE, CROSS-VALIDATE
# ─────────────────────────────────────────
results    = {}
kf         = KFold(n_splits=5, shuffle=False)   # time-aware — no shuffle

print("\n🚀 Training models...\n")

for name, model in models.items():
    print(f"   Training {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # Cross-validation RMSE on training set
    cv_scores = cross_val_score(model, X_train, y_train,
                                 cv=kf, scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores.mean()

    results[name] = {
        'model'  : model,
        'y_pred' : y_pred,
        'RMSE'   : rmse,
        'MAE'    : mae,
        'R2'     : r2,
        'CV_RMSE': cv_rmse
    }

    print(f"      RMSE : {rmse:.2f} KSh")
    print(f"      MAE  : {mae:.2f} KSh")
    print(f"      R²   : {r2:.4f}")
    print(f"      CV RMSE (5-fold): {cv_rmse:.2f} KSh\n")


# ─────────────────────────────────────────
# 6. PICK BEST MODEL
# ─────────────────────────────────────────
best_name  = min(results, key=lambda k: results[k]['RMSE'])
best       = results[best_name]
print(f"🏆 Best model: {best_name}  (RMSE = {best['RMSE']:.2f} KSh)")


# ─────────────────────────────────────────
# 7. RESULTS TABLE
# ─────────────────────────────────────────
print("\n📋 Model Comparison Table:")
print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'CV RMSE':>10}")
print("-" * 60)
for name, r in results.items():
    marker = " ← best" if name == best_name else ""
    print(f"{name:<22} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} {r['R2']:>8.4f} {r['CV_RMSE']:>10.2f}{marker}")


# ─────────────────────────────────────────
# 8. SAVE PREDICTIONS CSV
# ─────────────────────────────────────────
pred_df = pd.DataFrame({
    'Date'           : dates_test.values,
    'County'         : df_sorted['County'].iloc[split_idx:].values,
    'Actual_Price'   : y_test.values,
})
for name, r in results.items():
    col = name.replace(" ", "_") + "_Predicted"
    pred_df[col] = r['y_pred'].round(2)

pred_df['Best_Predicted'] = best['y_pred'].round(2)
pred_df['Error_KSh']      = (pred_df['Best_Predicted'] - pred_df['Actual_Price']).round(2)
pred_df.to_csv("predictions.csv", index=False)
print("\n💾 predictions.csv saved")


# ─────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────
print("🖼️  Generating plots...")

# ── Plot 1: Model Results (2x2 grid) ─────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Maize Price Prediction — Model Evaluation", fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# (a) Actual vs Predicted — best model
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, best['y_pred'], alpha=0.4, s=15, color=colors[0])
lims = [min(y_test.min(), best['y_pred'].min()),
        max(y_test.max(), best['y_pred'].max())]
ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
ax1.set_xlabel("Actual Wholesale Price (KSh)")
ax1.set_ylabel("Predicted Wholesale Price (KSh)")
ax1.set_title(f"Actual vs Predicted\n({best_name})")
ax1.legend(fontsize=9)
ax1.text(0.05, 0.92, f"R² = {best['R2']:.4f}", transform=ax1.transAxes,
         fontsize=10, color='navy', fontweight='bold')

# (b) Predicted vs Actual over time
ax2 = fig.add_subplot(gs[0, 1])
sample = pred_df.sort_values('Date').head(200)
ax2.plot(sample['Date'], sample['Actual_Price'],    label='Actual',    color='black',    linewidth=1.2)
ax2.plot(sample['Date'], sample['Best_Predicted'],  label='Predicted', color=colors[0],  linewidth=1.2, alpha=0.8)
ax2.set_xlabel("Date")
ax2.set_ylabel("Wholesale Price (KSh)")
ax2.set_title("Actual vs Predicted Over Time\n(first 200 test records)")
ax2.legend(fontsize=9)
ax2.tick_params(axis='x', rotation=30)

# (c) Model comparison bar chart
ax3 = fig.add_subplot(gs[1, 0])
model_names = list(results.keys())
rmse_vals   = [results[n]['RMSE'] for n in model_names]
mae_vals    = [results[n]['MAE']  for n in model_names]
x = np.arange(len(model_names))
width = 0.35
bars1 = ax3.bar(x - width/2, rmse_vals, width, label='RMSE', color=colors[0], alpha=0.85)
bars2 = ax3.bar(x + width/2, mae_vals,  width, label='MAE',  color=colors[1], alpha=0.85)
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, fontsize=9)
ax3.set_ylabel("Error (KSh)")
ax3.set_title("Model Comparison — RMSE vs MAE")
ax3.legend()
for bar in bars1:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

# (d) Residuals distribution
ax4 = fig.add_subplot(gs[1, 1])
residuals = y_test.values - best['y_pred']
ax4.hist(residuals, bins=40, color=colors[0], alpha=0.75, edgecolor='white')
ax4.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero error')
ax4.set_xlabel("Prediction Error (KSh)")
ax4.set_ylabel("Count")
ax4.set_title(f"Residuals Distribution\n({best_name})")
ax4.legend()
ax4.text(0.65, 0.92, f"MAE = {best['MAE']:.2f} KSh", transform=ax4.transAxes,
         fontsize=9, color='navy')

plt.savefig("model_results.png", dpi=150, bbox_inches='tight')
print("   model_results.png saved")

# ── Plot 2: Feature Importance ────────────
fig2, ax = plt.subplots(figsize=(10, 8))
best_model = best['model']

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
    colors_imp = ['#d62728' if v == feat_imp.max() else
                  '#1f77b4' if v >= feat_imp.quantile(0.75) else '#aec7e8'
                  for v in feat_imp.values]
    feat_imp.plot(kind='barh', ax=ax, color=colors_imp)
    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Feature Importance — {best_name}", fontsize=14, fontweight='bold')
    ax.axvline(feat_imp.mean(), color='red', linestyle='--', alpha=0.7, label='Mean importance')
    ax.legend()
    for i, (val, name) in enumerate(zip(feat_imp.values, feat_imp.index)):
        ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
print("   feature_importance.png saved")

# ─────────────────────────────────────────
# 10. SAVE BEST MODEL
# ─────────────────────────────────────────
joblib.dump(best_model, "best_model.pkl")
joblib.dump(FEATURES,   "model_features.pkl")
print("   best_model.pkl saved  (load with joblib.load('best_model.pkl'))")

# ─────────────────────────────────────────
# 11. EXAMPLE: PREDICT A SINGLE NEW RECORD
# ─────────────────────────────────────────
print("\n🔮 Example — predict price for a new market entry:")
example = pd.DataFrame([{
    'price_lag_7d'           : 50.0,
    'price_lag_14d'          : 48.5,
    'price_lag_30d'          : 46.0,
    'price_change_7d'        : 1.5,
    'temperature_mean_c'     : 19.2,
    'rain_sum_mm'            : 3.5,
    'rain_7d_rolling'        : 12.0,
    'rain_30d_rolling'       : 35.0,
    'temp_7d_rolling'        : 18.8,
    'month'                  : 4,
    'week'                   : 16,
    'year'                   : 2026,
    'day_of_year'            : 110,
    'season_encoded'         : 0,    # long_rains
    'county_encoded'         : 5,    # e.g. Nakuru
    'classification_encoded' : 1,    # White Maize
    'Supply Volume'          : 5000,
}])
predicted_price = best_model.predict(example[FEATURES])[0]
print(f"   Input : April 2026, long rains, Nakuru, White Maize")
print(f"   Predicted wholesale price: KSh {predicted_price:.2f} per kg")

print("\n✅ Training complete! Files saved:")
print("   - model_results.png")
print("   - feature_importance.png")
print("   - predictions.csv")
print("   - best_model.pkl")
