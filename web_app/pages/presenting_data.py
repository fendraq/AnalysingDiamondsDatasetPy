import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

import io

df = pd.read_csv('data/normalized_diamonds_clean_corr_carat_without_x_y_z.csv', index_col=0)

st.title("Data Presentation")

st.markdown(
    """
    ## The insights of the Diamonds dataset
    Below is summary of insights relevant for the dataset originating from the analysis. As mensioned in the *Cleaned*-section i only used the data that had correct calculated carat based on the measurement. After the cleaning I also normalized the data for making it possible to do a correlation and feature analisys. 
    """
)

st.markdown(
    """
    ### Correlation analysis
    The first plot show the correlation between different attributes of the diamond:
    """
)

fig = plt.figure(figsize=(7, 7))

correlation = df.select_dtypes(include=['number']).corr()
mask = np.triu(correlation)
sns.heatmap( correlation, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Korrelationsmatris för hela datasetet')
plt.tight_layout()

col1, col2, col3 = st.columns([1, 2, 1])
with col2: 
    st.pyplot(fig)

st.markdown(
    """
    In this correlation plot it shows that carat is almost exclusively the only factor possitively effecting the price. If this were the case then all diamonds, independent on other attributes such as cut wouldn't matter at all. I found this strange and decided to check a bit deeper on the carat and how it looked when splitting it into ten categories, as shown below:
    """
)

# Eventuellt lägga till en scatterplot för att visualisera tydliga värdeförändringar beroende på carat.

carat_bin = df['carat_bins'].unique()
nrows, ncols = 5, 2
fig, axs = plt.subplots(nrows, ncols, figsize=(12, 18), constrained_layout=True)

for i, category in enumerate(carat_bin):
    subset = df[df['carat_bins'] == category]
    count = len(subset)
    correlation = subset.select_dtypes(include=['number']).corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    ax = axs[i // ncols, i % ncols]
    sns.heatmap(
        correlation, annot=True, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
        ax=ax, annot_kws={"size": 8}, cbar=False, fmt=".2f"
    )
    ax.set_title(f"Correlation Matrix: {category}\nRows: {count}", fontsize=10, pad=12)

    if i % ncols == 0:
        ax.tick_params(axis='y', labelsize=8)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])

    if i // ncols == nrows - 1:
        ax.tick_params(axis='x', labelsize=8, rotation=45)
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
sns.heatmap(
    np.array([[0, 1], [-1, 0]]), cmap='coolwarm', vmin=-1, vmax=1, 
    cbar=True, cbar_ax=cbar_ax, annot=False
)
cbar_ax.set_ylabel('Correlation', fontsize=12)

plt.suptitle("Correlation Correct Calculated Carat", y=0.99, fontsize=16)
plt.tight_layout(rect=[0, 0, 0.9, 1]) 
st.pyplot(fig)

st.markdown(
    """
    When carat is split into ten groups another pattern is shown: 
    - For diamonds in the carat range 0.197–0.72, there is a strong positive correlation between carat and price and a moderately strong positive correlation between clarity and price, while color and cut are less decisive, but still show a weak positive correlation.
    - For diamonds in the carat range 0.72–0.98, the correlation between carat and price starts to decrease somewhat, and the correlations between both clarity and color with price even out.
    - For diamonds in the carat range 0.98–1.24, the correlation between carat and price has decreased significantly to 0.24, while clarity shows a very strong positive correlation with price.
    - For diamonds in the carat range 1.24–1.5, clarity maintains its strong positive correlation, but color becomes somewhat more important. These correlations decrease slightly in the carat range 1.5–1.76, and the correlation between carat and price remains weakly positive.
    - For diamonds in the carat range 1.76–2.02, the correlation between clarity and price decreases further, while color maintains its relatively strong correlation with price.
    """
)

st.markdown(
    """
    ### Feature Analysis
    To make the pattern more visual on how the attributes differ depending on the carat group it is shown in a feature plot below: 
    """
)

features = ['cut_encoded', 'color_encoded', 'clarity_encoded', 'depth', 'table']
bins = df['carat_bins'].unique()
importance_per_bin = {}

for bin in bins:
    subset = df[df['carat_bins'] == bin]
    X = subset[features]
    y = subset['price']
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    importance_per_bin[bin] = model.feature_importances_
    
importance_df = pd.DataFrame(importance_per_bin, index=features)

importance_long = importance_df.T.reset_index().melt(id_vars='index')
importance_long.columns = ['carat_bin', 'feature', 'importance']

fig = plt.figure(figsize=(12, 6))
sns.lineplot(
    data=importance_long,
    x='feature',
    y='importance',
    hue='carat_bin',
    marker='o'
)
plt.ylabel('Feature Importance')
plt.title('Feature Importance for Predicting Price per Carat Bin')
plt.legend(title='Carat Bin')
plt.tight_layout()
st.pyplot(fig)