
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# --- Page Config ---
st.set_page_config(page_title="Agritech Pest EDA and Preprocessing", layout="wide")
st.title("🚜 Agritech Pest - EDA and Preprocessing Showcase")

st.markdown("""
This app runs the full EDA (Exploratory Data Analysis) and preprocessing steps 
on the agricultural pest dataset. All results are computed live and displayed below.
""")

# --- Load Data ---
st.header("1. Load Dataset")

# Replace this line with your actual data loading
# Example: df = pd.read_csv('path/to/your/file.csv')
try:
    df = pd.read_csv('train.csv')  # Assuming the data is called 'train.csv' in same repo
    st.success("Data loaded successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading data: {e}")

# --- Data Overview ---
st.header("2. Data Overview")
st.write("### Dataset Information:")
buffer = []
df.info(buf=buffer)
info_str = '\n'.join(buffer)
st.text(info_str)

st.write("### Basic Statistics:")
st.dataframe(df.describe())

# --- Checking Null Values ---
st.header("3. Missing Values")
st.write("Checking missing values in the dataset:")
missing = df.isnull().sum()
st.dataframe(missing[missing > 0])

# --- Visualizations ---
st.header("4. Visualizations")

st.subheader("Class Distribution")
plt.figure(figsize=(10,5))
sns.countplot(x='Class', data=df)
plt.xticks(rotation=45)
st.pyplot(plt)

st.subheader("Feature Correlation Heatmap")
corr = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot(plt)

# --- Preprocessing ---
st.header("5. Preprocessing Steps")

st.subheader("Encoding Labels")
le = LabelEncoder()
df['Class_encoded'] = le.fit_transform(df['Class'])
st.dataframe(df[['Class', 'Class_encoded']].head())

st.subheader("Handling Missing Values")
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)
st.dataframe(df_imputed.head())

st.subheader("Train-Test Split")
X = df_imputed.drop('Class_encoded', axis=1)
y = df_imputed['Class_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training set size: {X_train.shape}")
st.write(f"Testing set size: {X_test.shape}")

# --- Footer ---
st.success("EDA and Preprocessing Completed Successfully! 🚀")
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
