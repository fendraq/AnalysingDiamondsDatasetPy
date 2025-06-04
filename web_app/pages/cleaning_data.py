import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.title("Cleaning the Data")

st.markdown(
    """
    ## Cleaning Steps
    The cleaning steps made to the dataset based on the information collected on the dataset:
    - removed zero values
    - removed extreme outliers based on the interquartile range (IQR)
    - changed object datatype to catagory, which also was encoded
    - removed original category columns of cut, color and clarity and kept the encoded columns
    - removed duplets
    - removed rows with incorrect calculated carat
    - removed x, y and z, because these are now represented in the carat and depth
    """
)

st.markdown(
    """
    ## The Clean Dataset
    Below is a review on the new clean dataset
    """
)

df = pd.read_csv("data/diamonds_clean_cor_without_x_y_z.csv", index_col=0)

with st.container():
    tabs = st.tabs([
        "Preview",
        "Shape",
        "Info",
        "Missing & Zero Values",
        "Summary Stats",
        "Duplicates",
        "Visualizations",
        "Correlation Heatmap",
    ])

    with tabs[0]:
        st.subheader("Preview of the Data")
        st.write("First 5 rows:")
        st.dataframe(df.head())
        st.write("Last 5 rows:")
        st.dataframe(df.tail())

    with tabs[1]:
        st.subheader("Dataset Dimensions")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

    with tabs[2]:
        st.subheader("Data Types and Info")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Non-Null Count': [df[col].count() for col in df.columns],
            'Dtype': [str(df[col].dtype) for col in df.columns]
        })
        st.dataframe(info_df)

    with tabs[3]:
        st.subheader("Missing and Zero Values")
        st.write("Missing values per column:")
        st.dataframe(df.isnull().sum())
        st.write("Zero values per column (numeric only):")
        st.dataframe((df == 0).sum(numeric_only=True))

    with tabs[4]:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include='all'))

    with tabs[5]:
        st.subheader("Duplicate Rows")
        num_dups = df.duplicated().sum()
        st.write(f"Number of duplicate rows: {num_dups}")

    with tabs[6]:
        st.subheader("Quick Visualizations")
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            columns = min(4, len(num_cols))
            rows = (len(num_cols) + columns - 1) // columns
            fig = plt.figure(figsize=(5*columns, 4*rows))
            for i, column in enumerate(num_cols):
                plt.subplot(rows, columns, i+1)
                sns.histplot(df[column].dropna(), kde=True)
                plt.title(f"Distribution of {column}")
                plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No numeric columns to plot.")

    with tabs[7]:
        st.subheader("Correlation Heatmap")
        num_cols = df.select_dtypes(include=['number']).columns
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for correlation heatmap.")