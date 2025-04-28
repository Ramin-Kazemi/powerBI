import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io

# Directory where CSVs are uploaded (current directory)
BASE_DIR = '.'

@st.cache_data
def load_capture_chart(path: str = BASE_DIR) -> dict:
    """
    Load specific Capture Chart CSV files using their exact filenames.
    Returns a dict of DataFrames keyed by dataset identifier.
    """
    files = {
        "Capture_Chart_Cicalino_1": "Capture_Chart(Cicalino_1).csv",
        "Capture_Chart_Cicalino_2": "Capture_Chart(Cicalino_2).csv",
        "Capture_Chart_Imola_1": "Capture_Chart(Imola_1).csv",
        "Capture_Chart_Imola_2": "Capture_Chart(Imola_2).csv",
        "Capture_Chart_Imola_3": "Capture_Chart(Imola_3).csv",
    }
    dfs = {}
    for key, fname in files.items():
        full_path = os.path.join(path, fname)
        if os.path.exists(full_path):
            dfs[key] = pd.read_csv(full_path)
        else:
            st.warning(f"File not found: {fname}")
    return dfs

@st.cache_data
def load_weather_data(path: str = BASE_DIR) -> dict:
    """
    Load specific Historical Weather Data CSV files using their exact filenames.
    Returns a dict of DataFrames keyed by dataset identifier.
    """
    files = {
        "Historical_Weather_Data_Cicalino_1": "Historical_Weather_Data(Cicalino_1).csv",
        "Historical_Weather_Data_Cicalino_2": "Historical_Weather_Data(Cicalino_2).csv",
        "Historical_Weather_Data_Imola_1": "Historical_Weather_Data(Imola_1).csv",
        "Historical_Weather_Data_Imola_2": "Historical_Weather_Data(Imola_2).csv",
        "Historical_Weather_Data_Imola_3": "Historical_Weather_Data(Imola_3).csv",
    }
    dfs = {}
    for key, fname in files.items():
        full_path = os.path.join(path, fname)
        if os.path.exists(full_path):
            dfs[key] = pd.read_csv(full_path)
        else:
            st.warning(f"File not found: {fname}")
    return dfs

def show_dataframes(dfs: dict):
    """
    Display each DataFrame briefly and provide a download link.
    """
    for name, df in dfs.items():
        st.write(f"### {name}")
        st.dataframe(df.head())
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download raw data for {name}",
            data=csv_data,
            file_name=f"{name}.csv",
            mime='text/csv'
        )

def plot_histograms(df: pd.DataFrame):
    """
    Let the user select a numeric column and plot its histogram.
    """
    st.subheader("Histogram of Numeric Feature")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not len(numeric_cols):
        st.write("No numeric columns available")
        return
    col = st.selectbox("Select numeric column", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), bins=20, ax=ax)
    ax.set_title(f"Histogram of {col}")
    st.pyplot(fig)

def show_summary(df: pd.DataFrame):
    st.subheader("Dataframe Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.subheader("Statistical Summary")
    st.write(df.describe())

def main():
    st.title("Agritech Pest EDA & Preprocessing App")
    st.sidebar.title("Navigation & Download")
    mode = st.sidebar.radio("Choose page:", [
        "Capture Chart Data",
        "Weather Data",
        "EDA & Visualization",
        "Preprocessing"
    ])

    # Allow downloading this app script
    with open(__file__, 'r') as f:
        script_bytes = f.read().encode('utf-8')
    st.sidebar.download_button(
        label="Download App Script",
        data=script_bytes,
        file_name="streamlit_app.py",
        mime="text/plain"
    )

    if mode == "Capture Chart Data":
        st.header("Capture Chart Datasets")
        dfs = load_capture_chart()
        show_dataframes(dfs)

    elif mode == "Weather Data":
        st.header("Historical Weather Datasets")
        dfs = load_weather_data()
        show_dataframes(dfs)

    elif mode == "EDA & Visualization":
        st.header("Explore any Dataset")
        cap = load_capture_chart()
        weath = load_weather_data()
        all_dfs = {**cap, **weath}
        name = st.selectbox("Select dataset for EDA", list(all_dfs.keys()))
        df = all_dfs[name]
        st.dataframe(df.head())
        if st.checkbox("Show summary info & stats"):
            show_summary(df)
        if st.checkbox("Show histogram plot"):
            plot_histograms(df)
        if st.checkbox("Show correlation heatmap"):
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
            st.pyplot(fig)

    elif mode == "Preprocessing":
        st.header("Preprocessing Steps")
        cap = load_capture_chart()
        weath = load_weather_data()
        all_dfs = {**cap, **weath}
        name = st.selectbox("Select dataset to preprocess", list(all_dfs.keys()))
        df = all_dfs[name]
        if st.checkbox("Drop missing values"):
            df = df.dropna()
            st.write("Dropped missing values.")
        if st.checkbox("Keep only numeric columns"):
            df = df.select_dtypes(include=[np.number])
            st.write("Selected numeric columns.")
        st.subheader("Processed Data Preview")
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download processed data",
            data=csv,
            file_name=f'processed_{name}.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
