
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(page_title="Agritech Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Raw Data", "Visualizations", "Statistics"])

# Automatically find CSV files in the current directory
@st.cache_data
def list_csv_files():
    return [f for f in os.listdir() if f.endswith(".csv")]

csv_files = list_csv_files()

selected_file = None
df = pd.DataFrame()

if csv_files:
    selected_file = st.sidebar.selectbox("Select a CSV file", csv_files)
    df = pd.read_csv(selected_file)

# Tabs by section
if section == "Overview":
    st.title("🌿 Agritech Pest & Weather Dashboard")
    st.markdown("This dashboard gives insights into pest captures and weather conditions across various regions.")
    st.markdown("Use the sidebar to select a file and explore raw data, visualizations, and statistics.")

elif section == "Raw Data":
    st.header("📄 Raw Dataset")
    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("No data loaded.")

elif section == "Visualizations":
    st.header("📊 Data Visualizations")
    if not df.empty:
        st.subheader("Insect Captures Over Time")
        fig, ax = plt.subplots()
        if 'Date' in df.columns and 'Catture' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values("Date")
            ax.plot(df['Date'], df['Catture'])
            ax.set_xlabel("Date")
            ax.set_ylabel("Catture")
            ax.set_title("Insect Catture Over Time")
            st.pyplot(fig)
        else:
            st.warning("The dataset must contain 'Date' and 'Catture' columns.")

elif section == "Statistics":
    st.header("📈 Basic Statistics")
    if not df.empty:
        st.write(df.describe())
    else:
        st.info("No data available to describe.")
