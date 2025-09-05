import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. Load Data ---
df = pd.read_csv(r'D:\zyxthesis\zvoting_2012\zvoting_2012.csv')
df['Longitude'] = df['geo_X']
df['Latitude'] = df['geo_Y']
df = df.drop(columns=['geo_X', 'geo_Y'])
df = df[(df['new_pct_dem'] >= 0) & (df['new_pct_dem'] <= 100)]
df.dropna(inplace=True)

# --- 2. Features and Target ---
target = 'new_pct_dem'
base_features = ['total_pop','sex_ratio','pct_black','pct_hisp','pct_bach',
                 'median_income','pct_65_over','pct_age_18_29','gini','pct_manuf',
                 'ln_pop_den','pct_fb','pct_insured']
features_with_coords = base_features + ['Latitude','Longitude']

X = df[features_with_coords]
y = df[target]

# --- 3. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Train XGBoost Model ---
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='rmse'
)
model.fit(X_train, y_train, verbose=False)

# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred),
      "RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)),
      "MAE:", mean_absolute_error(y_test, y_pred))

# --- 6. SHAP Values ---
explainer = shap.Explainer(model)
shap_values = explainer(X)

# --- 7. Create GeoDataFrame ---
gdf_wgs84 = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df['Longitude'], df['Latitude'])],
    crs="EPSG:4326"
)
gdf = gdf_wgs84.to_crs("EPSG:3857").reset_index(drop=True)

shap_df = pd.DataFrame(shap_values.values, columns=[f'{col}_shap' for col in X.columns])
df_plot = pd.concat([df.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)

# --- 8. Global Color Mapping ---
global_min = np.nanmin(shap_df.values)
global_max = np.nanmax(shap_df.values)
total_range = global_max - global_min
zero_pos = abs(global_min) / total_range

# Output zero position in the colormap
print(f" SHAP Value range: {global_min:.8f} ~ {global_max:.8f}")

# Transition near zero
transition_width = 0.05
neg_transition_start = max(0.0, zero_pos - transition_width/2)
pos_transition_end = min(1.0, zero_pos + transition_width/2)

colors = ["#3a7bd5", "#bfe9ff", "#ffecec", "#ff6f61", "#93291e"]
positions = [0.0, neg_transition_start, zero_pos, pos_transition_end, 1.0]

custom_cmap = LinearSegmentedColormap.from_list("smooth_diverging", list(zip(positions, colors)))
norm = Normalize(vmin=global_min, vmax=global_max)

def shap_to_hex(val):
    return to_hex(custom_cmap(norm(val)))

# --- 9. Point Size Mapping (6 Quantile Classes) ---
SIZE_CLASSES = [8, 13, 18, 22, 27, 32]

def size_from_feature_quantile(arr, feature_name):
    arr = np.asarray(arr, dtype=float)

    # Six quantiles
    quantiles = [0.0, 0.023, 0.159, 0.500, 0.841, 0.977, 1.0]
    bins = np.quantile(arr, quantiles)

    sizes = np.zeros_like(arr)
    for i in range(len(arr)):
        for j in range(6):
            if bins[j] <= arr[i] <= bins[j+1]:
                sizes[i] = SIZE_CLASSES[j]
                break

    # Print quantile binning results
    print(f"\n📌 Feature: {feature_name}")
    print(f"  Feature value range: {np.nanmin(arr):.8f} ~ {np.nanmax(arr):.8f}")
    for j in range(6):
        lower, upper = bins[j], bins[j+1]
        mask = (arr >= lower) & (arr <= upper)
        print(f"  Bin {j+1}: "
              f"{lower:.6f} - {upper:.6f} ")

    return sizes

# --- 10. Generate Color & Size Fields ---
for feature in features_with_coords:
    shap_col = f'{feature}_shap'
    vals = df_plot[shap_col].to_numpy()
    feature_vals = df_plot[feature].to_numpy()

    df_plot[f'{feature}_color'] = [shap_to_hex(v) for v in vals]
    df_plot[f'{feature}_size'] = size_from_feature_quantile(feature_vals, feature)

# --- 11. Export to GeoPackage ---
out_path = r"D:\zyxthesis\shap_output"
out_file = "shap_results_zzzz.gpkg"

export_cols = [f"{f}_shap" for f in features_with_coords] + \
              [f"{f}_color" for f in features_with_coords] + \
              [f"{f}_size" for f in features_with_coords]

gdf_export = gdf.join(df_plot[export_cols])

gdf_export.to_file(
    filename=f"{out_path}\\{out_file}",
    driver="GPKG",
    encoding="utf-8"
)

print("Exported GeoPackage, can be loaded directly in ArcGIS Pro:", f"{out_path}\\{out_file}")

# --- 12. Mean Absolute SHAP Bar Plot ---
print("Generating mean absolute SHAP bar plot...")
mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns)
mean_abs_shap = mean_abs_shap.sort_values(ascending=True)

plt.figure(figsize=(12, 8))
mean_abs_shap.plot(kind='barh', color='cornflowerblue')
plt.title('Feature Importance (Based on the Mean Absolute SHAP Value)', fontsize=16)
plt.xlabel('Mean Absolute SHAP Score (Impact on Model Output Size)')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('shap_mean_absolute_bar_chart.png', dpi=300)
plt.close()
print("Feature importance bar chart saved as 'shap_mean_absolute_bar_chart.png'")

# --- 13. SHAP Summary Plot ---
print("Generating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False, plot_size=None)
plt.title('SHAP Summary Plot for Democratic Party Vote Percentage', fontsize=16)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300)
plt.close()
print("SHAP summary plot saved as 'shap_summary_plot.png'")
