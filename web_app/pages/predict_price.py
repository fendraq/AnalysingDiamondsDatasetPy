import streamlit as st
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Value settings
FEATURES = ['carat', 'depth', 'table', 'cut_encoded', 'color_encoded', 'clarity_encoded']
FEATURES_WITH_BIN = FEATURES + ['carat_bin_encoded']
ATTR_LABELS = {
    'carat': 'Carat',
    'cut_encoded': 'Cut (encoded)',
    'color_encoded': 'Color (encoded)',
    'clarity_encoded': 'Clarity (encoded)',
    'depth': 'Depth',
    'table': 'Table',
    'carat_bin_encoded': 'Carat Bin (encoded)'
}

# Models
MODEL_WITHOUT_BIN = 'models/model_without_bins.pkl'
SCALER_WITHOUT_BINS = 'models/scaler_without_bins.pkl'
MODEL_WITH_BIN = 'models/model_with_bins.pkl' 
SCALER_WITH_BINS = 'models/scaler_with_bins.pkl'

# Carat bins
carat_bins = [
    (0.197, 0.46),
    (0.46, 0.72),
    (0.72, 0.98),
    (0.98, 1.24),
    (1.24, 1.5),
    (1.5, 1.76),
    (1.76, 2.02),
    (2.02, 2.28),
    (2.28, 2.54),
    (2.54, 2.8)
]

def carat_to_bin(carat, bins):
    """Map a carat value to its corresponding bin index."""
    for idx, (low, high) in enumerate(bins):
        if low <= carat <= high:  # Changed condition to include lower bound
            return idx
    return None

# Page

st.title("Price estimation - Comparison between with and without carat groups")

st.markdown("## Select diamond attributes")
input_cols = st.columns(len(FEATURES))
input_values = []
for idx, feat in enumerate(FEATURES):
    if feat == 'carat':
        min_,  max_, val = 0.2, 2.72, 1.0
        step = 0.1
    elif 'encoded' in feat:
        min_,  max_, val = 0, 7, 3
        step = 1
    elif feat == 'depth':
        min_, max_, val = 55.6, 66.9, 61.0 
        step = 0.1
    else:
        min_, max_, val = 50.0, 68.0, 59.0 
        step = 0.1
    val = input_cols[idx].slider(ATTR_LABELS[feat], min_value=min_, max_value=max_, value=val, step=step)
    input_values.append(val)
    
carat_value = input_values[0]
carat_bin_idx = carat_to_bin(carat_value, carat_bins)
if carat_bin_idx is None:
    st.error("Carat value is out of bin range")
    st.stop()
    
def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Couldn't load model or scaler from {model_path}, {scaler_path}: {e}")
        st.stop()
        
model1, scaler1 = load_model_and_scaler(MODEL_WITHOUT_BIN, SCALER_WITHOUT_BINS)
model2, scaler2 = load_model_and_scaler(MODEL_WITH_BIN, SCALER_WITH_BINS)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model 1: not trained with carat groups")
    X1 = np.array([input_values])
    X1_scaled = scaler1.transform(X1)
    price1 = model1.predict(X1_scaled)[0]
    st.markdown(f"<h2 style='color:green;'>Predicted Price: ${price1:,.2f}</h2>", unsafe_allow_html=True)
    
    st.markdown("**Feature importance:**")
    if hasattr(model1, "feature_importances_"):
        importances1 = model1.feature_importances_
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        y_pos = np.arange(len(FEATURES))
        ax1.barh(y_pos, importances1, align='center')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([ATTR_LABELS[f] for f in FEATURES])
        ax1.invert_yaxis()
        ax1.set_xlabel("Importance")
        ax1.set_title("Model 1 Feature importances")
        st.pyplot(fig1)
    else:
        st.info("Feature importance not available for Model 1.")
        
with col2:
    st.subheader("Model 2: trained with carat groups")
    input_values2 = input_values + [carat_bin_idx]
    X2 = np.array([input_values2])
    X2_scaled = scaler2.transform(X2)
    price2 = model2.predict(X2_scaled)[0]
    st.markdown(f"<h2 style='color:blue;'>Predicted Price: ${price2:,.2f}</h2>", unsafe_allow_html=True)
    st.markdown(f"<small>Carat bin (auto-detected): <b>{carat_bin_idx}</b> for carat value {carat_value}</small>", unsafe_allow_html=True)
    
    st.markdown("**Feature importance**")
    if hasattr(model2, "feature_importances_"):
        importances2 = model2.feature_importances_
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        y_pos2 = np.arange(len(FEATURES_WITH_BIN))
        ax2.barh(y_pos2, importances2, align='center')
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels([ATTR_LABELS[f] for f in FEATURES_WITH_BIN])
        ax2.invert_yaxis()
        ax2.set_xlabel("Importance")
        ax2.set_title("Model 2 Feature importances")
        st.pyplot(fig2)
    else:
        st.info("Feature importance not available for Model 2.")
        
st.markdown("---")
st.markdown("## Comparison summary")

diff = price2 - price1
color = "green" if diff > 0 else "red" if diff < 0 else "grey"
st.markdown(
    f"<h3>Price difference (Model 2 - Model 1):"
    f"<span style='color:{color};'> ${diff:,.2f}</span></h2>", unsafe_allow_html=True
)

if 'importances1' in locals() and 'importances2' in locals():
    st.markdown("### Feature importance comparison")
    
    common_features = FEATURES
    importances2_common = importances2[:len(FEATURES)]
    diff_imp = np.array(importances2_common) - np.array(importances1)
    
    comp_table = {
        "Feature": [ATTR_LABELS[f] for f in common_features],
        "Model 1 importance": importances1,
        "Model 2 importance": importances2_common,
        "Difference (2-1)": diff_imp
    }
    st.dataframe(comp_table)
    
    figd, axd = plt.subplots(figsize=(5,3))
    y_posd = np.arange(len(common_features))
    axd.barh(y_posd, diff_imp, align='center', color=['#ff6347' if v < 0 else '#90ee90' for v in diff_imp])
    axd.set_yticks(y_posd)
    axd.set_yticklabels([ATTR_LABELS[f] for f in common_features])
    axd.invert_yaxis()
    axd.set_xlabel('Importance Difference')
    axd.set_title('Feature Importance Difference (Model 2 - Model 1)')
    st.pyplot(figd)

else:
    st.info("Feature importances are not available for both models.")