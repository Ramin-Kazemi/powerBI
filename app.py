
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Agritech Pest EDA and Preprocessing", layout="wide")
st.title("🚜 Agritech Pest - EDA and Preprocessing Showcase")

# Load dataset
st.header("1. Load Dataset")
try:
    df = pd.read_csv('train.csv')  # assuming train.csv is in the repo
    st.success("Data loaded successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading dataset: {e}")

# Overview
st.header("2. Dataset Overview")
st.subheader("Info:")
buffer = []
df.info(buf=buffer)
info_str = '\n'.join(buffer)
st.text(info_str)

st.subheader("Describe:")
st.dataframe(df.describe())

# Checking for null values
st.header("3. Missing Values")
missing_values = df.isnull().sum()
st.dataframe(missing_values[missing_values > 0])

# Visualization
st.header("4. Visualizations")

st.subheader("Class Distribution")
plt.figure(figsize=(10,5))
sns.countplot(x='Class', data=df)
plt.xticks(rotation=45)
st.pyplot(plt.gcf())

st.subheader("Correlation Heatmap")
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot(plt.gcf())

# Preprocessing
st.header("5. Preprocessing Steps")

st.subheader("Label Encoding")
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

st.write("Training set shape:", X_train.shape)
st.write("Testing set shape:", X_test.shape)

# Completion
st.success("EDA and Preprocessing Completed Successfully! 🚀")
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
