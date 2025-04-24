
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Agritech Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Raw Data", "Visualizations", "Statistics"])

# Load data
@st.cache_data
def load_data():
    try:
        final_df = pd.read_csv("Final_df.csv")
        return final_df
    except Exception as e:
        st.error("Failed to load data. Make sure 'Final_df.csv' is in the repo.")
        return pd.DataFrame()

df = load_data()

# Tabs by section
if section == "Overview":
    st.title("🌿 Agritech Pest & Weather Dashboard")
    st.markdown("This dashboard gives insights into pest captures and weather conditions across various regions.")
    st.markdown("Use the sidebar to explore raw data, visualizations, and basic statistics.")

elif section == "Raw Data":
    st.header("📄 Raw Dataset")
    st.dataframe(df)

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
