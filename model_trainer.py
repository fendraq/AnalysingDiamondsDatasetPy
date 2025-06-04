import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

data_path = PROJECT_ROOT / "data" / "diamonds_clean_cor_without_x_y_z.csv"
df_original_clean = pd.read_csv(data_path, index_col=0)

carat_bin_categories = [
    '(0.197, 0.46]', '(0.46, 0.72]', '(0.72, 0.98]', '(0.98, 1.24]', '(1.24, 1.5]',
    '(1.5, 1.76]', '(1.76, 2.02]', '(2.02, 2.28]', '(2.28, 2.54]', '(2.54, 2.8]'
]

carat_bin_edges = [0.197, 0.46, 0.72, 0.98, 1.24, 1.5, 1.76, 2.02, 2.28, 2.54, 2.8]

df_original_clean['carat_bins'] = pd.cut(
    df_original_clean['carat'],
    bins=carat_bin_edges,
    labels=carat_bin_categories,
    include_lowest=True,
    right=True,
    ordered=True
)


df_original_clean['carat_bins'] = pd.Categorical(df_original_clean['carat_bins'], categories=carat_bin_categories, ordered=True)
df_original_clean['carat_bins_encoded'] = df_original_clean['carat_bins'].cat.codes

features = ['carat', 'depth', 'table', 'cut_encoded', 'color_encoded', 'clarity_encoded']
target = 'price'

X = df_original_clean[features].values
y = df_original_clean[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#passa in scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Träna modellen
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

score = model.score(X_test_scaled, y_test)
print(f"Test R^2 score {score:.4f}")

# spara scaler och modell
model_dir = PROJECT_ROOT / "web_app" / "models"
model_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

joblib.dump(scaler, model_dir / "scaler_without_bins.pkl")
joblib.dump(model, model_dir / "model_without_bins.pkl")

features_inc_bins = ['carat', 'depth', 'table', 'cut_encoded', 'color_encoded', 'clarity_encoded', 'carat_bins_encoded']
target = 'price'


X = df_original_clean[features_inc_bins].values
y = df_original_clean[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#passa in scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Träna modellen
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

score = model.score(X_test_scaled, y_test)
print(f"Test R^2 score {score:.4f}")

# spara scaler och modell
joblib.dump(scaler, model_dir / "scaler_with_bins.pkl")
joblib.dump(model, model_dir / "model_with_bins.pkl")

