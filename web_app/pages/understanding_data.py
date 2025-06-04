import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df = pd.read_csv("data/diamonds.csv")

st.title("Understanding the Data")

st.markdown(
    """
    ## The Dataset
    The dataset *Diamonds* is a dataset of 53940 rows with the columns:
    - price, in US dollars - ranging from 326 to 18,823
    - carat, defining the weight of the diamond - ranging from 0.2 to 5.01
    - cut, quality of the cut - defined through Fair, Good, Very Good, Premium and Ideal
    - color, diamond color - ranging from J (worst) to D (best)
    - clarity, how clear the diamond is - ranging from I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1 and IF (best)
    - depth, a mesurement of the relation between the hight (z) one one side and width (z) and length (y) on the other - ranging from 43 to 79
    - table, the flat area on top of the diamond in relation to the wwidest points - ranging from 43 to 95
    - x, the length in mm - ranging from 0 to 10.74
    - y, the width in mm - ranging from 0 to 58.9
    - z, height in mm - ranging from 0 to 31.
    
    *For more information go to [Education](https://www.diamonds.pro/education/)*
    """
)

st.markdown(
    """
    ## Descriptive Information on the Dataset
    The shape of the is 53940 on y and 10 on x. 
    ### Datatypes
    The datatypes for the dataset is as following:
    """
)
info_df = pd.DataFrame({
    'Column': df.columns,
    'Non-Null Count': [df[col].count() for col in df.columns],
    'Dtype': [str(df[col].dtype) for col in df.columns]
})
st.dataframe(info_df)

st.markdown(
    """
    As seen above there is no null values. To be taken into account is the attributes cut, color and clarity, which are strings. For doing an analysis these three need to be converted to categories.
    """
)

st.markdown(
    """
    ### Descriptive Statistics
    """
)

st.dataframe(df.describe())
st.markdown(
    """
    The summary shows zero values in the attributes x, y and z, which isn't plausable and therefore need to be cleaned. In carat the max value of 5.01 differs quite much from the Q3, which also need to be looked closer at. The same goes for the max value of table. y and z does allso have some high max values compared to the Q3.
    """
)

st.markdown(
    """
    ### Duplicate Rows
    The dataset contains 146 duplicated rows, which isn't many but will be cleaned.
    """
)

st.markdown(
    """
    ### Distibution within attributes
    As seen in the distribution of clarity in the figure below the order of the clarity values on x is in the wrong order (I1 is the worst clarity and should be closest to origo), which need to be adressed in cleaning.
    """
)

fig = plt.figure(figsize=(15, 15))
columns = min(4, len(df.columns))
rows = (len(df.columns) + columns -1) //columns
for i, column in enumerate(df.select_dtypes(include=['number', 'object']).columns):
    plt.subplot(rows, columns, i+1)
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    
st.pyplot(fig)

st.markdown(
    """
    In a box plot (as seen below) it's clear that there is quite alot of extreme values, which need to be adressed in the cleaning.
    """
)

sns.set_theme(style="whitegrid")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

sns.boxplot(x=df["price"], ax=axs[0, 0])
axs[0, 0].set_title("Price")

sns.boxplot(x=df["carat"], ax=axs[0, 1])
axs[0, 1].set_title("Carat")

depth_table_df = pd.DataFrame({
    'Table': df['table'],
    'Depth': df['depth']
})
depth_table_melted = pd.melt(depth_table_df, var_name='measurement', value_name='value')
sns.boxplot(
    x='measurement',
    y='value',
    data=depth_table_melted, 
    ax=axs[1, 0]
    )
axs[1, 0].set_title("Depth and Table")

xyz_df = pd.DataFrame({
    'Length': df['x'],
    'Width': df['y'],
    'Hight': df['z']
})
xyz_melted = pd.melt(xyz_df, var_name='measurement', value_name='value')
sns.boxplot(
    x="measurement",
    y="value",
    data=xyz_melted,
    ax=axs[1, 1]
)
axs[1, 1].set_title("X, Y, Z Measurements")

plt.tight_layout()

st.pyplot(fig)

st.markdown(
    """
    ### Reliability of values in carat dependent on x, y and z
    In this section I want to see if the carat is correct depending on a diamonds measurement, eg if the x, y and z is used for calculating the carat, do i get the same value as in the dataset. For this i used the formula:
    """
)
st.latex(r"(\frac{x + y}{2})^2 \times z \times 0.0061")
st.markdown(
    """
    with a error margin of 10 percent ([Calculate Carat](https://niceice.com/diamond-mm-to-carat-weight-calculation/)). The result is shown in the plot below:
    """
)

df['calculated_carat'] = np.nan
df['is_correct_carat'] = False

extreme_errors = []

for index, row in df.iterrows():
    x, y, z = row['x'], row['y'], row['z']
    actual_carat = row['carat']
    
    if abs(x - y) <= 0.02 * (x + y / 2):
        calculated = ((x + y) / 2)**2 * z * 0.0061
    else:
        calculated = x * y * z * 0.0062
        
    df.at[index, 'calculated_carat'] = calculated
    df.at[index, 'is_correct_carat'] = (abs(calculated - actual_carat) / actual_carat <= 0.02)
    if abs(calculated - actual_carat) / actual_carat > 2:
        extreme_errors.append(index)
        
df['carat_calculated_status'] = df['is_correct_carat'].map({True: 'Correct Calculation', False: 'Incorrect Calculation'})
    
fig = plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x='carat',
    y='calculated_carat',
    hue='carat_calculated_status',
    palette={'Correct Calculation': 'green', 'Incorrect Calculation': 'red'},
    alpha=0.7
)    

max_val = max(df['carat'].max(), df['calculated_carat'].max())
plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

plt.title('Actual vs. Calculated Carat Weight')
plt.xlabel('Actual Carat')
plt.ylabel('Calculated Carat')
plt.grid(True, alpha=0.3)
plt.tight_layout()

st.pyplot(fig)

st.markdown(
    """
    As seen in the plot there is alot of carat values that doesn't add up with the calculation. The total extent is:
    - 23534 data points with correct calculated carat
    - 30386 data points with incorrect calculated carat
    There is two ways of viewing this: Either the measurements as a whole isn't reliable and needed to be cleaned. Or the carat values aren't reliable and need to be cleand. In this instance I believe a dataset that have measurements that add upp to the carat and still is quite large is more important than keeping the rows with incorrect calculated carat. Other factors which I can't take into account is the source of the dataset or the timeperiod for which the data was collected.
    """
)