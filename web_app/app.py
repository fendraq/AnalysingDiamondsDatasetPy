import streamlit as st
import pandas as pd
import numpy as np
# import joblib
st.set_page_config(page_title="Diamond Data Presentation", layout="wide")

st.title("Analysing the dataset diamonds")

descriptive_page = st.Page("pages/understanding_data.py", title="Dataset Information")
cleaning_page = st.Page("pages/cleaning_data.py", title="How the Data Was Cleaned")
presenting_page = st.Page("pages/presenting_data.py", title="Data Presentation")
predict_price_page = st.Page("pages/predict_price.py", title="Diamond Price Prediction")

pg = st.navigation({"Menu":[descriptive_page, cleaning_page, presenting_page, predict_price_page]})

pg.run()

