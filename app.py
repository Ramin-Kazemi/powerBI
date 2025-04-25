
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple Agritech Viewer", layout="centered")

st.title("📄 Agritech CSV Viewer")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Basic Statistics")
    st.write(df.describe())

    if 'Date' in df.columns and 'Catture' in df.columns:
        st.subheader("Insect Catture Over Time")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values("Date")

        fig, ax = plt.subplots()
        ax.plot(df['Date'], df['Catture'])
        ax.set_xlabel("Date")
        ax.set_ylabel("Catture")
        ax.set_title("Insect Catture Over Time")
        st.pyplot(fig)
    else:
        st.info("This dataset must contain 'Date' and 'Catture' columns for visualization.")
else:
    st.info("Please upload a CSV file to get started.")
