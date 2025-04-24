import streamlit as st
st.title('Agritech Pest Data Analysis Dashboard')

# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Cicalino
Capture_Chart_Cicalino_1 = pd.read_csv("Capture_Chart(Cicalino_1).csv")
Capture_Chart_Cicalino_2 = pd.read_csv("Capture_Chart(Cicalino_2).csv")
Historical_Weather_Data_Cicalino_1 = pd.read_csv("Historical_Weather_Data(Cicalino_1).csv")
Historical_Weather_Data_Cicalino_2 = pd.read_csv("Historical_Weather_Data(Cicalino_2).csv")

# Imola
Capture_Chart_Imola_1 = pd.read_csv("Capture_Chart(Imola_1).csv")
Capture_Chart_Imola_2 = pd.read_csv("Capture_Chart(Imola_2).csv")
Capture_Chart_Imola_3 = pd.read_csv("Capture_Chart(Imola_3).csv")
Historical_Weather_Data_Imola_1 = pd.read_csv("Historical_Weather_Data(Imola_1).csv")
Historical_Weather_Data_Imola_2 = pd.read_csv("Historical_Weather_Data(Imola_2).csv")
Historical_Weather_Data_Imola_3 = pd.read_csv("Historical_Weather_Data(Imola_3).csv")

# Showing first rows of each file for Cicalino
st.write("** Capture Chart Cicalino 1 ** \n",Capture_Chart_Cicalino_1.head(), "\n")
st.write("** Capture Chart Cicalino 2 ** \n",Capture_Chart_Cicalino_2.head(), "\n")
st.write("** Historical Weather Data Cicalino 1 ** \n",Historical_Weather_Data_Cicalino_1.head(), "\n")
st.write("** Historical Weather Data Cicalino 2 ** \n",Historical_Weather_Data_Cicalino_2.head(), "\n")

# Showing first rows of each file for Imola
st.write("** Capture Chart Imola 1 ** \n",Capture_Chart_Imola_1.head(), "\n")
st.write("** Capture Chart Imola 2 ** \n",Capture_Chart_Imola_2.head(), "\n")
st.write("** Capture Chart Imola 3 ** \n",Capture_Chart_Imola_3.head(), "\n")
st.write("** Historical Weather Data Imola 1 ** \n",Historical_Weather_Data_Imola_1.head(), "\n")
st.write("** Historical Weather Data Imola 2 ** \n",Historical_Weather_Data_Imola_2.head(), "\n")
st.write("** Historical Weather Data Imola 3 ** \n",Historical_Weather_Data_Imola_3.head(), "\n")

# Function to check the summay of dataset
def show_datasets_info(datasets: dict):
    """
    Prints information about multiple datasets.

    Parameters:
    datasets (dict): A dictionary where keys are dataset names and values are pandas DataFrames.
    """
    for name, df in datasets.items():
        st.write(f"\n{'='*40}\nDataset: {name}\n{'='*40}")
        st.write(df.info())

# Store datasets in a dictionary
datasets = {
    "Capture_Chart_Cicalino_1": Capture_Chart_Cicalino_1,
    "Capture_Chart_Cicalino_2": Capture_Chart_Cicalino_2,
    "Capture_Chart_Imola_1": Capture_Chart_Imola_1,
    "Capture_Chart_Imola_2": Capture_Chart_Imola_2,
    "Capture_Chart_Imola_3": Capture_Chart_Imola_3,
    "Historical_Weather_Data_Cicalino_1": Historical_Weather_Data_Cicalino_1,
    "Historical_Weather_Data_Cicalino_2": Historical_Weather_Data_Cicalino_2,
    "Historical_Weather_Data_Imola_1": Historical_Weather_Data_Imola_1,
    "Historical_Weather_Data_Imola_2": Historical_Weather_Data_Imola_2,
    "Historical_Weather_Data_Imola_3": Historical_Weather_Data_Imola_3
}

# Call function for get dataset info
show_datasets_info(datasets)

# Function to show statistic summary of all datasets
def show_datasets_summary(datasets: dict):
    """
    Prints summary statistics for multiple datasets.

    Parameters:
    datasets (dict): A dictionary where keys are dataset names and values are pandas DataFrames.
    """
    for name, df in datasets.items():
        st.write(f"\n{'='*40}\nDataset: {name}\n{'='*40}")
        st.write(df.describe())
        #st.write(df.head())


# Call function to gest statistical summary
show_datasets_summary(datasets)

# Fix Capture Chart data for Cicalino 1
CaptureChart_Cicalino1 = Capture_Chart_Cicalino_1
CaptureChart_Cicalino1.columns = CaptureChart_Cicalino1.iloc[0]  # Set first row as header
CaptureChart_Cicalino1 = CaptureChart_Cicalino1[1:].reset_index(drop=True)  # Remove first row

# Fix Capture Chart data for Cicalino 2
CaptureChart_Cicalino2 = Capture_Chart_Cicalino_2
CaptureChart_Cicalino2.columns = CaptureChart_Cicalino2.iloc[0]  # Set first row as header
CaptureChart_Cicalino2 = CaptureChart_Cicalino2[1:].reset_index(drop=True)  # Remove first row

# Fix Historical Weather Data Cicalino 1
HistoricalWeather_Cicalino1 = Historical_Weather_Data_Cicalino_1
HistoricalWeather_Cicalino1.columns = HistoricalWeather_Cicalino1.iloc[0]
HistoricalWeather_Cicalino1 = HistoricalWeather_Cicalino1[2:].reset_index(drop=True)  # Remove first two rows

# Fix Historical Weather Data Cicalino 2
HistoricalWeather_Cicalino2 = Historical_Weather_Data_Cicalino_2
HistoricalWeather_Cicalino2.columns = HistoricalWeather_Cicalino2.iloc[0]
HistoricalWeather_Cicalino2 = HistoricalWeather_Cicalino2[2:].reset_index(drop=True)  # Remove first two rows

# # Fix Historical Weather Data for Cicalino
# CaptureChart_Cicalino1.head(), CaptureChart_Cicalino2.head(), HistoricalWeather_Cicalino1.head(), HistoricalWeather_Cicalino2.head()

# Fix Capture Chart data for Imola 1
CaptureChart_Imola1 = Capture_Chart_Imola_1
CaptureChart_Imola1.columns = CaptureChart_Imola1.iloc[0]  # Set first row as header
CaptureChart_Imola1 = CaptureChart_Imola1[1:].reset_index(drop=True)  # Remove first row

# Fix Capture Chart data for Imola 2
CaptureChart_Imola2 = Capture_Chart_Imola_2
CaptureChart_Imola2.columns = CaptureChart_Imola2.iloc[0]  # Set first row as header
CaptureChart_Imola2 = CaptureChart_Imola2[1:].reset_index(drop=True)  # Remove first row

# Fix Capture Chart data for Imola 2
CaptureChart_Imola3 = Capture_Chart_Imola_3
CaptureChart_Imola3.columns = CaptureChart_Imola3.iloc[0]  # Set first row as header
CaptureChart_Imola3 = CaptureChart_Imola3[1:].reset_index(drop=True)

# Fix Historical Weather Data Imola 1
HistoricalWeather_Imola1 = Historical_Weather_Data_Imola_1
HistoricalWeather_Imola1.columns = HistoricalWeather_Imola1.iloc[0]
HistoricalWeather_Imola1 = HistoricalWeather_Imola1[2:].reset_index(drop=True)  # Remove first two rows

# Fix Historical Weather Data Imola 2
HistoricalWeather_Imola2 = Historical_Weather_Data_Imola_2
HistoricalWeather_Imola2.columns = HistoricalWeather_Imola2.iloc[0]
HistoricalWeather_Imola2 = HistoricalWeather_Imola2[2:].reset_index(drop=True)  # Remove first two rows

# Fix Historical Weather Data Imola 3
HistoricalWeather_Imola3 = Historical_Weather_Data_Imola_3
HistoricalWeather_Imola3.columns = HistoricalWeather_Imola3.iloc[0]
HistoricalWeather_Imola3 = HistoricalWeather_Imola3[2:].reset_index(drop=True)  # Remove first two rows

# # Fix Historical Weather Data for Imola
# CaptureChart_Imola1.head(), CaptureChart_Imola2.head(), CaptureChart_Imola3.head(), HistoricalWeather_Imola1.head(), HistoricalWeather_Imola2.head(), HistoricalWeather_Imola3.head()

# Standardize column names for Capture Chart Cicalion
CaptureChart_Cicalino1.columns = ["DateTime", "Number of Insects", "New Catches", "Reviewed", "Event"]
CaptureChart_Cicalino2.columns = ["DateTime", "Number of Insects", "New Catches", "Reviewed", "Event"]

# Standardize column names for Historical Weather Data Cicalion
HistoricalWeather_Cicalino1.columns = ["DateTime", "Average Temperature", "Temp_low", "Temp_high", "Average Humidity"]
HistoricalWeather_Cicalino2.columns = ["DateTime", "Average Temperature", "Temp_low", "Temp_high", "Average Humidity"]

# st.write("** Capture Chart Cicalino 1 ** \n",CaptureChart_Cicalino1.head(), "\n")
# st.write("** Capture Chart Cicalino 2 ** \n",CaptureChart_Cicalino2.head(), "\n")
# st.write("** Historical Weather Data Cicalino 1 ** \n",HistoricalWeather_Cicalino1.head(), "\n")
# st.write("** Historical Weather Data Cicalino 2 ** \n",HistoricalWeather_Cicalino2.head(), "\n")

# Standardize column names for Capture Chart Imola
CaptureChart_Imola1.columns = ["DateTime", "Number of Insects", "New Catches", "Reviewed", "Event"]
CaptureChart_Imola2.columns = ["DateTime", "Number of Insects", "New Catches", "Reviewed", "Event"]
CaptureChart_Imola3.columns = ["DateTime", "Number of Insects", "New Catches", "Reviewed", "Event"]

# Standardize column names for Historical Weather Data Imola
HistoricalWeather_Imola1.columns = ["DateTime", "Average Temperature", "Temp_low", "Temp_high", "Average Humidity"]
HistoricalWeather_Imola2.columns = ["DateTime", "Average Temperature", "Temp_low", "Temp_high", "Average Humidity"]
HistoricalWeather_Imola3.columns = ["DateTime", "Average Temperature", "Temp_low", "Temp_high", "Average Humidity"]

# st.write("** Capture Chart Imola 1 ** \n",CaptureChart_Imola1.head(), "\n")
# st.write("** Capture Chart Imola 2 ** \n",CaptureChart_Imola2.head(), "\n")
# st.write("** Historical Weather Data Imola 1 ** \n",HistoricalWeather_Imola1.head(), "\n")
# st.write("** Historical Weather Data Imola 2 ** \n",HistoricalWeather_Imola2.head(), "\n")
# st.write("** Historical Weather Data Imola 3 ** \n",HistoricalWeather_Imola3.head(), "\n")

# Function to process DateTime, add Location, and reorder columns
def process_dataset(df, location, col="DateTime"):
    df[col] = pd.to_datetime(df[col], format="%d.%m.%Y %H:%M:%S", errors="coerce")  # Ensure datetime format
    df["Date"] = df[col].dt.strftime("%Y-%m-%d")  # Extract date in YYYY-MM-DD format
    df["Time"] = df[col].dt.strftime("%H:%M:%S")  # Extract time in HH:MM:SS format
    df["Location"] = location  # Add Location column

    return df

# Dictionary mapping datasets to their respective locations
datasets_with_locations = {
    "Cicalino1": [CaptureChart_Cicalino1, HistoricalWeather_Cicalino1],
    "Cicalino2": [CaptureChart_Cicalino2, HistoricalWeather_Cicalino2],
    "Imola1": [CaptureChart_Imola1, HistoricalWeather_Imola1],
    "Imola2": [CaptureChart_Imola2, HistoricalWeather_Imola2],
    "Imola3": [CaptureChart_Imola3, HistoricalWeather_Imola3],
}

# Apply processing to each dataset
for location, dfs in datasets_with_locations.items():
    for i in range(len(dfs)):
        dfs[i] = process_dataset(dfs[i], location)

# Drop "DateTime" column
CaptureChart_Cicalino1.drop(columns=["DateTime"], inplace=True)
CaptureChart_Cicalino2.drop(columns=["DateTime"], inplace=True)
HistoricalWeather_Cicalino1.drop(columns=["DateTime"], inplace=True)
HistoricalWeather_Cicalino2.drop(columns=["DateTime"], inplace=True)
CaptureChart_Imola1.drop(columns=["DateTime"], inplace=True)
CaptureChart_Imola2.drop(columns=["DateTime"], inplace=True)
CaptureChart_Imola3.drop(columns=["DateTime"], inplace=True)
HistoricalWeather_Imola1.drop(columns=["DateTime"], inplace=True)
HistoricalWeather_Imola2.drop(columns=["DateTime"], inplace=True)
HistoricalWeather_Imola3.drop(columns=["DateTime"], inplace=True)


# Reorder "Date" and "Time" columns as first and seconf column from the left
CaptureChart_Cicalino1 = CaptureChart_Cicalino1[["Date", "Time"] + [col for col in CaptureChart_Cicalino1 if col not in ["Date", "Time"]]]
CaptureChart_Cicalino2 = CaptureChart_Cicalino2[["Date", "Time"] + [col for col in CaptureChart_Cicalino2 if col not in ["Date", "Time"]]]
HistoricalWeather_Cicalino1 = HistoricalWeather_Cicalino1[["Date", "Time"] + [col for col in HistoricalWeather_Cicalino1 if col not in ["Date", "Time"]]]
HistoricalWeather_Cicalino2 = HistoricalWeather_Cicalino2[["Date", "Time"] + [col for col in HistoricalWeather_Cicalino2 if col not in ["Date", "Time"]]]
CaptureChart_Imola1 = CaptureChart_Imola1[["Date", "Time"] + [col for col in CaptureChart_Imola1 if col not in ["Date", "Time"]]]
CaptureChart_Imola2 = CaptureChart_Imola2[["Date", "Time"] + [col for col in CaptureChart_Imola2 if col not in ["Date", "Time"]]]
CaptureChart_Imola3 = CaptureChart_Imola3[["Date", "Time"] + [col for col in CaptureChart_Imola3 if col not in ["Date", "Time"]]]
HistoricalWeather_Imola1 = HistoricalWeather_Imola1[["Date", "Time"] + [col for col in HistoricalWeather_Imola1 if col not in ["Date", "Time"]]]
HistoricalWeather_Imola2 = HistoricalWeather_Imola2[["Date", "Time"] + [col for col in HistoricalWeather_Imola2 if col not in ["Date", "Time"]]]
HistoricalWeather_Imola3 = HistoricalWeather_Imola3[["Date", "Time"] + [col for col in HistoricalWeather_Imola3 if col not in ["Date", "Time"]]]

# # Display output
# st.write("** Capture Chart Cicalino 1 ** \n", CaptureChart_Cicalino1.head(), "\n")
# st.write("** Capture Chart Cicalino 2 ** \n", CaptureChart_Cicalino2.head(), "\n")
# st.write("** Historical Weather Data Cicalino 1 ** \n", HistoricalWeather_Cicalino1.head(), "\n")
# st.write("** Historical Weather Data Cicalino 2 ** \n", HistoricalWeather_Cicalino2.head(), "\n")
# st.write("** Capture Chart Imola 1 ** \n", CaptureChart_Imola1.head(), "\n")
# st.write("** Capture Chart Imola 2 ** \n", CaptureChart_Imola2.head(), "\n")
# st.write("** Capture Chart Imola 3 ** \n", CaptureChart_Imola3.head(), "\n")
# st.write("** Historical Weather Data Imola 1 ** \n", HistoricalWeather_Imola1.head(), "\n")
# st.write("** Historical Weather Data Imola 2 ** \n", HistoricalWeather_Imola2.head(), "\n")
# st.write("** Historical Weather Data Imola 3 ** \n", HistoricalWeather_Imola3.head(), "\n")

# # Convert DateTime column to proper datetime format for Cicalion
# CaptureChart_Cicalino1["DateTime"] = CaptureChart_Cicalino1["Date"].dt.strftime("%Y-%m-%d")
# CaptureChart_Cicalino2["DateTime"] = CaptureChart_Cicalino2["Date"].dt.strftime("%Y-%m-%d")
# HistoricalWeather_Cicalino1["DateTime"] = HistoricalWeather_Cicalino1["Date"].dt.strftime("%Y-%m-%d")
# HistoricalWeather_Cicalino2["DateTime"] = HistoricalWeather_Cicalino2["Date"].dt.strftime("%Y-%m-%d")

# st.write("** Capture Chart Cicalino 1 ** \n",CaptureChart_Cicalino1.head(), "\n")
# st.write("** Capture Chart Cicalino 2 ** \n",CaptureChart_Cicalino2.head(), "\n")
# st.write("** Historical Weather Data Cicalino 1 ** \n",HistoricalWeather_Cicalino1.head(), "\n")
# st.write("** Historical Weather Data Cicalino 2 ** \n",HistoricalWeather_Cicalino2.head(), "\n")

# # Convert DateTime column to proper datetime format for Imola
# CaptureChart_Imola1["DateTime"] = CaptureChart_Imola1["DateTime"].dt.date
# CaptureChart_Imola2["DateTime"] = CaptureChart_Imola2["DateTime"].dt.date
# CaptureChart_Imola3["DateTime"] = CaptureChart_Imola3["DateTime"].dt.date
# HistoricalWeather_Imola1["DateTime"] = HistoricalWeather_Imola1["DateTime"].dt.date
# HistoricalWeather_Imola2["DateTime"] = HistoricalWeather_Imola2["DateTime"].dt.date
# HistoricalWeather_Imola3["DateTime"] = HistoricalWeather_Imola3["DateTime"].dt.date

# st.write("** Capture Chart Imola 1 ** \n",CaptureChart_Imola1.head(), "\n")
# st.write("** Capture Chart Imola 2 ** \n",CaptureChart_Imola2.head(), "\n")
# st.write("** Capture Chart Imola 3 ** \n",CaptureChart_Imola3.head(), "\n")
# st.write("** Historical Weather Data Imola 1 ** \n",HistoricalWeather_Imola1.head(), "\n")
# st.write("** Historical Weather Data Imola 2 ** \n",HistoricalWeather_Imola2.head(), "\n")
# st.write("** Historical Weather Data Imola 3 ** \n",HistoricalWeather_Imola3.head(), "\n")

# Convert numeric columns (replace commas with dots and convert to float)
CaptureChart_NumericalColumns = ["Number of Insects", "New Catches"]
HistoricalWeather_NumericalColumns = ["Average Temperature", "Temp_low", "Temp_high", "Average Humidity"]

# Convert the numerical columns to integer type
for col in CaptureChart_NumericalColumns:
    CaptureChart_Cicalino1[col] = CaptureChart_Cicalino1[col].astype(float, errors="ignore")
    CaptureChart_Cicalino2[col] = CaptureChart_Cicalino2[col].astype(float, errors="ignore")
    CaptureChart_Imola1[col] = CaptureChart_Imola1[col].astype(float, errors="ignore")
    CaptureChart_Imola2[col] = CaptureChart_Imola2[col].astype(float, errors="ignore")
    CaptureChart_Imola3[col] = CaptureChart_Imola3[col].astype(float, errors="ignore")

# Convert the numerical columns to float type
for col in HistoricalWeather_NumericalColumns:
    HistoricalWeather_Cicalino1[col] = HistoricalWeather_Cicalino1[col].str.replace(",", ".").astype(float, errors="ignore")
    HistoricalWeather_Cicalino2[col] = HistoricalWeather_Cicalino2[col].str.replace(",", ".").astype(float, errors="ignore")
    HistoricalWeather_Imola1[col] = HistoricalWeather_Imola1[col].str.replace(",", ".").astype(float, errors="ignore")
    HistoricalWeather_Imola2[col] = HistoricalWeather_Imola2[col].str.replace(",", ".").astype(float, errors="ignore")
    HistoricalWeather_Imola3[col] = HistoricalWeather_Imola3[col].str.replace(",", ".").astype(float, errors="ignore")


# st.write("** Capture Chart Cicalino 1 ** \n", CaptureChart_Cicalino1.head(), "\n")
# st.write("** Capture Chart Cicalino 2 ** \n", CaptureChart_Cicalino2.head(), "\n")
# st.write("** Historical Weather Data Cicalino 1 ** \n", HistoricalWeather_Cicalino1.head(), "\n")
# st.write("** Historical Weather Data Cicalino 2 ** \n", HistoricalWeather_Cicalino2.head(), "\n")
# st.write("** Capture Chart Imola 1 ** \n", CaptureChart_Imola1.head(), "\n")
# st.write("** Capture Chart Imola 2 ** \n", CaptureChart_Imola2.head(), "\n")
# st.write("** Capture Chart Imola 3 ** \n", CaptureChart_Imola3.head(), "\n")
# st.write("** Historical Weather Data Imola 1 ** \n", HistoricalWeather_Imola1.head(), "\n")
# st.write("** Historical Weather Data Imola 2 ** \n", HistoricalWeather_Imola2.head(), "\n")
# st.write("** Historical Weather Data Imola 3 ** \n", HistoricalWeather_Imola3.head(), "\n")

# Check for missing values
st.write("Missing Values for CaptureChart_Cicalino1: \n", CaptureChart_Cicalino1.isnull().sum(), "\n")
st.write("Missing Values for CaptureChart_Cicalino2: \n", CaptureChart_Cicalino2.isnull().sum(), "\n")
st.write("Missing Values for HistoricalWeather_Cicalino1: \n", HistoricalWeather_Cicalino1.isnull().sum(), "\n")
st.write("Missing Values for HistoricalWeather_Cicalino2: \n", HistoricalWeather_Cicalino2.isnull().sum(), "\n")
st.write("Missing Values for CaptureChart_Imola1: \n", CaptureChart_Imola1.isnull().sum(), "\n")
st.write("Missing Values for CaptureChart_Imola2: \n", CaptureChart_Imola2.isnull().sum(), "\n")
st.write("Missing Values for CaptureChart_Imola3: \n", CaptureChart_Imola3.isnull().sum(), "\n")
st.write("Missing Values for HistoricalWeather_Imola1: \n", HistoricalWeather_Imola1.isnull().sum(), "\n")
st.write("Missing Values for HistoricalWeather_Imola: \n", HistoricalWeather_Imola2.isnull().sum(), "\n")
st.write("Missing Values for HistoricalWeather_Imola: \n", HistoricalWeather_Imola3.isnull().sum(), "\n")

# # Check for missing values
# missing_values = {
#     "CaptureChart_Cicalino1": CaptureChart_Cicalino1.isnull().sum(),
#     "CaptureChart_Cicalino2": CaptureChart_Cicalino2.isnull().sum(),
#     "HistoricalWeather_Cicalino1": HistoricalWeather_Cicalino1.isnull().sum(),
#     "HistoricalWeather_Cicalino2": HistoricalWeather_Cicalino2.isnull().sum(),
#     "CaptureChart_Imola1": CaptureChart_Imola1.isnull().sum(),
#     "CaptureChart_Imola2": CaptureChart_Imola2.isnull().sum(),
#     "CaptureChart_Imola3": CaptureChart_Imola3.isnull().sum(),
#     "HistoricalWeather_Imola1": HistoricalWeather_Imola1.isnull().sum(),
#     "HistoricalWeather_Imola2": HistoricalWeather_Imola2.isnull().sum(),
#     "HistoricalWeather_Imola3": HistoricalWeather_Imola3.isnull().sum(),
# }

# missing_values = pd.DataFrame(missing_values)
# st.write(missing_values)

# Drop columns from dataset in Capture Chart datasets
CaptureChart_Cicalino1.drop(columns=["Reviewed"], inplace=True)
CaptureChart_Cicalino2.drop(columns=["Reviewed"], inplace=True)
CaptureChart_Imola1.drop(columns=["Reviewed"], inplace=True)
CaptureChart_Imola2.drop(columns=["Reviewed"], inplace=True)
CaptureChart_Imola3.drop(columns=["Reviewed"], inplace=True)

# # Fill the missing value with 0 if missing value exists
# CaptureChart_Cicalino1.fillna(0, inplace=True)
# CaptureChart_Cicalino2.fillna(0, inplace=True)
# CaptureChart_Imola1.fillna(0, inplace=True)
# CaptureChart_Imola2.fillna(0, inplace=True)
# CaptureChart_Imola3.fillna(0, inplace=True)

# Store datasets in a dictionary
datasets = {
    "CaptureChart_Cicalino1": CaptureChart_Cicalino1,
    "CaptureChart_Cicalino2": CaptureChart_Cicalino2,
    "Capture_Chart_Imola_1": CaptureChart_Imola1,
    "Capture_Chart_Imola_2": CaptureChart_Imola2,
    "Capture_Chart_Imola_3": CaptureChart_Imola3,
    "HistoricalWeather_Cicalino1": HistoricalWeather_Cicalino1,
    "HistoricalWeather_Cicalino2": HistoricalWeather_Cicalino2,
    "Historical_Weather_Data_Imola_1": HistoricalWeather_Imola1,
    "Historical_Weather_Data_Imola_2": HistoricalWeather_Imola2,
    "Historical_Weather_Data_Imola_3": HistoricalWeather_Imola3
}

# Call function
show_datasets_info(datasets)

st.write("** Capture Chart Cicalino 1 ** \n", CaptureChart_Cicalino1.head(), "\n")
st.write("** Capture Chart Cicalino 2 ** \n", CaptureChart_Cicalino2.head(), "\n")
st.write("** Historical Weather Data Cicalino 1 ** \n", HistoricalWeather_Cicalino1.head(), "\n")
st.write("** Historical Weather Data Cicalino 2 ** \n", HistoricalWeather_Cicalino2.head(), "\n")
st.write("** Capture Chart Imola 1 ** \n", CaptureChart_Imola1.head(), "\n")
st.write("** Capture Chart Imola 2 ** \n", CaptureChart_Imola2.head(), "\n")
st.write("** Capture Chart Imola 3 ** \n", CaptureChart_Imola3.head(), "\n")
st.write("** Historical Weather Data Imola 1 ** \n", HistoricalWeather_Imola1.head(), "\n")
st.write("** Historical Weather Data Imola 2 ** \n", HistoricalWeather_Imola2.head(), "\n")
st.write("** Historical Weather Data Imola 3 ** \n", HistoricalWeather_Imola3.head(), "\n")

# Function to concatenate multiple datasets
def combine_datasets(dfs):
    return pd.concat(dfs, ignore_index=True)  # Combine and reset index

# Combine CaptureChart datasets
CaptureChart_Cicalino = combine_datasets([CaptureChart_Cicalino1, CaptureChart_Cicalino2])
CaptureChart_Imola = combine_datasets([CaptureChart_Imola1, CaptureChart_Imola2, CaptureChart_Imola3])

# Combine HistoricalWeather datasets
HistoricalWeather_Cicalino = combine_datasets([HistoricalWeather_Cicalino1, HistoricalWeather_Cicalino2])
HistoricalWeather_Imola = combine_datasets([HistoricalWeather_Imola1, HistoricalWeather_Imola2, HistoricalWeather_Imola3])

# Display outputs
st.write("** Combined Capture Chart Cicalino ** \n", CaptureChart_Cicalino.head(), "\n")
st.write("** Combined Historical Weather Cicalino ** \n", HistoricalWeather_Cicalino.head(), "\n")

st.write("** Combined Capture Chart Imola ** \n", CaptureChart_Imola.head(), "\n")
st.write("** Combined Historical Weather Imola ** \n", HistoricalWeather_Imola.head(), "\n")


# Save datasets to CSV files
CaptureChart_Cicalino.to_csv('CaptureChart_Cicalino.csv', index=False)
CaptureChart_Imola.to_csv('CaptureChart_Imola.csv', index=False)
HistoricalWeather_Cicalino.to_csv('HistoricalWeather_Cicalino.csv', index=False)
HistoricalWeather_Imola.to_csv('HistoricalWeather_Imola.csv', index=False)

st.write("Datasets saved successfully")

# Store datasets in a dictionary
Combined_DS = {
    "CaptureChart_Cicalino": CaptureChart_Cicalino,
    "CaptureChart_Imola": CaptureChart_Imola,
    "HistoricalWeather_Cicalino": HistoricalWeather_Cicalino,
    "HistoricalWeather_Imola": HistoricalWeather_Imola
}

# Function to show statistic summary of all datasets
def show_datasets_summary(datasets: dict):
    """
    Prints summary statistics for multiple datasets.

    Parameters:
    datasets (dict): A dictionary where keys are dataset names and values are pandas DataFrames.
    """
    for name, df in datasets.items():
        st.write(f"\n{'='*40}\nDataset: {name}\n{'='*40}")
        st.write(df.describe())
        #st.write(df.head())


# Call function to gest statistical summary
show_datasets_summary(Combined_DS)

# Function to plot distributions
def plot_distributions(df, dataset_name):
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        plt.figure(figsize=(12, 4))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], bins=30, kde=True, color='blue')
        plt.title(f"Histogram of {col} - {dataset_name}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col], color='green')
        plt.title(f"Boxplot of {col} - {dataset_name}")

        plt.tight_layout()
        st.pyplot()

# Plot distributions for each dataset
plot_distributions(CaptureChart_Cicalino, "CaptureChart_Cicalino")
plot_distributions(CaptureChart_Imola, "CaptureChart_Imola")
plot_distributions(HistoricalWeather_Cicalino, "HistoricalWeather_Cicalino")
plot_distributions(HistoricalWeather_Imola, "HistoricalWeather_Imola")

# Function to calculate missing values percentage
def missing_values(df, dataset_name):
    return pd.DataFrame({'Dataset': dataset_name, 'Column': df.columns, 'Missing %': df.isnull().mean() * 100})

# Calculate missing values for each dataset
missing_weather_cicalino = missing_values(HistoricalWeather_Cicalino, 'HistoricalWeather_Cicalino')
missing_weather_imola = missing_values(HistoricalWeather_Imola, 'HistoricalWeather_Imola')
missing_capture_cicalino = missing_values(CaptureChart_Cicalino, 'CaptureChart_Cicalino')
missing_capture_imola = missing_values(CaptureChart_Imola, 'CaptureChart_Imola')

# Combine all missing values
missing_data = pd.concat([missing_weather_cicalino, missing_weather_imola,
                          missing_capture_cicalino, missing_capture_imola])

# Plot missing values using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Column', y='Missing %', hue='Dataset', data=missing_data)
plt.xticks(rotation=45, ha='right')
plt.title('Missing Values Percentage in Each Dataset')
plt.ylabel('Percentage of Missing Values')
plt.xlabel('Columns')
plt.legend(title='Dataset')
st.pyplot()

st.write("\n")

# Plot missing values using a pie chart (aggregated by dataset)
missing_summary = missing_data.groupby('Dataset')['Missing %'].mean()

plt.figure(figsize=(6, 6))
plt.pie(missing_summary, labels=missing_summary.index, autopct='%1.1f%%', colors=['blue', 'orange', 'green', 'red'])
plt.title('Average Missing Values Percentage per Dataset')
st.pyplot()

# Fill the Nan-Values in each dataset with 0 if missing value exists
CaptureChart_Cicalino.fillna(0, inplace=True)
HistoricalWeather_Cicalino.fillna(0, inplace=True)

CaptureChart_Imola.fillna(0, inplace=True)
HistoricalWeather_Imola.fillna(0, inplace=True)

# Check for missing values
st.write("Missing Values for CaptureChart_Cicalino: \n", CaptureChart_Cicalino.isnull().sum(), "\n")
st.write("Missing Values for HistoricalWeather_Cicalino: \n", HistoricalWeather_Cicalino.isnull().sum(), "\n")
st.write("Missing Values for CaptureChart_Imola: \n", CaptureChart_Imola.isnull().sum(), "\n")
st.write("Missing Values for HistoricalWeather_Imola: \n", HistoricalWeather_Imola.isnull().sum(), "\n")

# # Convert Date and Time columns to datetime format
# for df in [HistoricalWeather_Cicalino, HistoricalWeather_Imola, CaptureChart_Cicalino, CaptureChart_Imola]:
#     df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

# # Merge datasets on Datetime for comparison
# merg_cicalino = pd.merge(HistoricalWeather_Cicalino, CaptureChart_Cicalino, on="Datetime", how="left")
# merg_imola = pd.merge(HistoricalWeather_Imola, CaptureChart_Imola, on="Datetime", how="left")

# # Function to plot relationships
# def plot_relationships(merged_df, location):
#     # Temperature vs. Insect Captures
#     plt.figure(figsize=(8, 5))
#     sns.scatterplot(x=merged_df["Average Temperature"], y=merged_df["Number of Insects"], color="red")
#     plt.title(f"Temperature vs. Insect Captures - {location}")
#     plt.xlabel("Average Temperature (°C)")
#     plt.ylabel("Number of Insects")
#     st.pyplot()

#     # Humidity vs. Insect Captures
#     plt.figure(figsize=(8, 5))
#     sns.scatterplot(x=merged_df["Average Humidity"], y=merged_df["Number of Insects"], color="blue")
#     plt.title(f"Humidity vs. Insect Captures - {location}")
#     plt.xlabel("Average Humidity (%)")
#     plt.ylabel("Number of Insects")
#     st.pyplot()

#     # Temperature vs. Humidity (with regression line)
#     plt.figure(figsize=(8, 5))
#     sns.regplot(x=merged_df["Average Temperature"], y=merged_df["Average Humidity"], scatter_kws={'alpha':0.5}, line_kws={"color":"green"})
#     plt.title(f"Temperature vs. Humidity - {location}")
#     plt.xlabel("Average Temperature (°C)")
#     plt.ylabel("Average Humidity (%)")
#     st.pyplot()


# # Generate plots for both locations
# plot_relationships(merg_cicalino, "Cicalino")
# plot_relationships(merg_imola, "Imola")

# Function to calculate som features
def process_weather_data(df):
    """
    Process a dataset to compute daily temperature and humidity statistics.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Date', 'Temperature', and 'Humidity' columns.
        - 'Date' should be in datetime or string format (convertible to datetime).

    Returns:
        pd.DataFrame: Original DataFrame with additional columns:
            - 'Day Avg_temp': Mean temperature for the day
            - 'Day Min_temp': Minimum temperature for the day
            - 'Day Max_temp': Maximum temperature for the day
            - 'Day Avg_Humidity': Mean humidity for the day
            - 'Temp_change': Difference between max and min temperature for the day
    """

    # Ensure Date column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate daily mean of "Average Temperature"
    daily_avg_temp = df.groupby('Date')['Average Temperature'].mean().rename("Day Avg_temp").round(2)

    # Find daily min of "Temp_low" and max of "Temp_high"
    daily_min_temp = df.groupby('Date')['Temp_low'].min().rename("Day Min_temp")
    daily_max_temp = df.groupby('Date')['Temp_high'].max().rename("Day Max_temp")

    # Calculate daily mean of "Average Humidity"
    daily_avg_humidity = df.groupby('Date')['Average Humidity'].mean().rename("Day Avg_Humidity").round(2)

    # Calculate temperature change (Max minus Min of "Temp_low" and "Temp_high")
    temp_change = (daily_max_temp - daily_min_temp).rename("Temp_change").round(2)

    # Merge calculated values back into the original dataframe
    df = df.merge(daily_avg_temp, on='Date', how='left')
    df = df.merge(daily_min_temp, on='Date', how='left')
    df = df.merge(daily_max_temp, on='Date', how='left')
    df = df.merge(daily_avg_humidity, on='Date', how='left')
    df = df.merge(temp_change, on='Date', how='left')

    return df


# Call process_weather_data function for Historical Weather data
Cicalino_HistoricalWeather_NewFeature = process_weather_data(HistoricalWeather_Cicalino)
Imola_HistoricalWeather_NewFeature = process_weather_data(HistoricalWeather_Imola)

# Display output
st.write("** Cicalino Historical Weather Data ** \n", Cicalino_HistoricalWeather_NewFeature.head(), "\n")
st.write("** Imola Historical Weather Data ** \n", Imola_HistoricalWeather_NewFeature.head(), "\n")

# Cicalino_HistoricalWeather_NewFeature.to_csv('Cicalino_HistoricalWeather_NewFeature.csv', index=False)
# Imola_HistoricalWeather_NewFeature.to_csv('Imola_HistoricalWeather_NewFeature.csv', index=False)

# Load datasets
weather_cicalino = Cicalino_HistoricalWeather_NewFeature
capture_cicalino = CaptureChart_Cicalino

# Convert Date and Time to datetime format
weather_cicalino['Datetime'] = pd.to_datetime(weather_cicalino['Date'].astype(str) + ' ' + weather_cicalino['Time'].astype(str), errors='coerce')
capture_cicalino['Datetime'] = pd.to_datetime(capture_cicalino['Date'].astype(str) + ' ' + capture_cicalino['Time'].astype(str), errors='coerce')

# Extract only Date and Hour (ignore minutes & seconds)
weather_cicalino['Date_Hour'] = weather_cicalino['Datetime'].dt.strftime('%Y-%m-%d %H')
capture_cicalino['Date_Hour'] = capture_cicalino['Datetime'].dt.strftime('%Y-%m-%d %H')

# Select only required columns from weather dataset
cicalino_weather_selected = weather_cicalino[['Location','Date_Hour', 'Average Temperature', 'Temp_low', 'Temp_high', 'Average Humidity',
                                              "Day Avg_temp", "Day Min_temp", "Day Max_temp", "Day Avg_Humidity", "Temp_change"]]

# Merge datasets based on Date_Hour and Location
Merged_Cicalino = pd.merge(capture_cicalino, cicalino_weather_selected, on=['Date_Hour', 'Location'], how='left')

# Drop the intermediate Date_Hour column
Merged_Cicalino.drop(columns=['Date_Hour', 'Datetime'], inplace=True)

# Display the first few rows
st.write(Merged_Cicalino.head())

# # Save the merged dataset if needed
# Merged_Cicalino.to_csv("Merged_Cicalino.csv", index=False)

# Load datasets
weather_imola = Imola_HistoricalWeather_NewFeature
capture_imola = CaptureChart_Imola

# Convert Date and Time to datetime format
weather_imola['Datetime'] = pd.to_datetime(weather_imola['Date'].astype(str) + ' ' + weather_imola['Time'].astype(str), errors='coerce')
capture_imola['Datetime'] = pd.to_datetime(capture_imola['Date'].astype(str) + ' ' + capture_imola['Time'].astype(str), errors='coerce')

# Extract only Date and Hour (ignore minutes & seconds)
weather_imola['Date_Hour'] = weather_imola['Datetime'].dt.strftime('%Y-%m-%d %H')
capture_imola['Date_Hour'] = capture_imola['Datetime'].dt.strftime('%Y-%m-%d %H')

# Select only required columns from weather dataset
imola_weather_selected = weather_imola[['Location','Date_Hour', 'Average Temperature', 'Temp_low', 'Temp_high', 'Average Humidity',
                                        "Day Avg_temp", "Day Min_temp", "Day Max_temp", "Day Avg_Humidity", "Temp_change"]]

# Merge datasets based on Date_Hour
Merged_Imola = pd.merge(capture_imola, imola_weather_selected, on=['Date_Hour', 'Location'], how='left')

# Drop the intermediate Date_Hour column
Merged_Imola.drop(columns=['Date_Hour', 'Datetime'], inplace=True)

# Display the first few rows
st.write(Merged_Imola.head())

# # Save the merged dataset if needed
# Merged_Imola.to_csv("Merged_Imola.csv", index=False)

df = Merged_Cicalino

# Identify rows where Event is "Cleaning"
cleaning_rows = df[df["Event"] == "Cleaning"]

# Iterate over cleaning rows to update corresponding rows
for _, row in cleaning_rows.iterrows():
    df.loc[(df["Date"] == row["Date"]) & (df["Location"] == row["Location"]) & (df["Event"] != "Cleaning"), "Event"] = "1"

# Drop rows where Event is "Cleaning"
Updated_Merged_Cicalino = df[df["Event"] != "Cleaning"]

# # Save the modified file
# Updated_Merged_Cicalino.to_csv("Updated_Merged_Cicalino.csv", index=False)

# Return the path to the updated file
Updated_Merged_Cicalino.head()

df = Merged_Imola

# Identify rows where Event is "Cleaning"
cleaning_rows = df[df["Event"] == "Cleaning"]

# Iterate over cleaning rows to update corresponding rows
for _, row in cleaning_rows.iterrows():
    df.loc[(df["Date"] == row["Date"]) & (df["Location"] == row["Location"]) & (df["Event"] != "Cleaning"), "Event"] = "1"

# Drop rows where Event is "Cleaning"
Updated_Merged_Imola = df[df["Event"] != "Cleaning"]

# # Save the modified file
# Updated_Merged_Imola.to_csv("Updated_Merged_Imola.csv", index=False)

# Return the path to the updated file
Updated_Merged_Imola.head()

# Concatenate the datasets
Final_Merged_Dataset = pd.concat([Updated_Merged_Cicalino, Updated_Merged_Imola], ignore_index=True)


Final_Merged_Dataset.to_csv("Final_Merged_Dataset.csv", index=False)

# Return the path to the merged file
Final_Merged_Dataset.head()

#######################
# # Load dataset
# df = pd.read_csv("Final_Merged_Dataset.csv")

# # Combine 'Date' and 'Time' columns into a single datetime column
# df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# # Set as index for time-based operations
# df.set_index('Datetime', inplace=True)

# # # Drop original 'Date' and 'Time' columns
# # df.drop(columns=['Date', 'Time'], inplace=True)

# # Check for missing values
# missing_values = df.isnull().sum()
# st.write("Missing Values per Column:")
# st.write(missing_values)

# # Apply time-based interpolation if missing values exist
# if missing_values.sum() > 0:
#     df.interpolate(method='time', inplace=True)
#     df.round(2)
#     st.write("Missing values have been interpolated using time-based method.")
# else:
#     st.write("No missing values detected.")

# # Save the cleaned dataset
# df.to_csv("Final_Merged_Dataset_Cleaned.csv")

# Function to estimate and fill missing values
def interpolate_missing_values(df, location_col, target_cols):
    """
    Fill missing values using time-based interpolation for a specific location.

    Parameters:
    df (pd.DataFrame): The dataset containing missing values.
    location_col (str): Column name representing the location.
    target_cols (list): List of columns to interpolate.

    Returns:
    pd.DataFrame: Dataset with missing values filled.
    """
    df = df.copy()

    # Filter data for Cicalino2
    cicalino2_data = df[df[location_col] == "Cicalino2"].sort_values(by="Time")

    #st.write(cicalino2_data)

    # Interpolate missing values for each target column
    for col in target_cols:
        cicalino2_data[col] = cicalino2_data[col].interpolate(method='linear', limit_direction='both')

    # Merge back into the main dataframe
    df.update(cicalino2_data)

    return df

# Function to Identify missing values and send to "interpolate_missing_values" each in time
def handle_missing_values(df, location_col, target_cols):
    """
    Identify missing values, send them to interpolation, and return the updated dataset.

    Parameters:
    df (pd.DataFrame): The dataset containing missing values.
    location_col (str): Column name representing the location.
    target_cols (list): List of columns to check for missing values.

    Returns:
    pd.DataFrame: Updated dataset with missing values handled.
    """
    # Check for missing values
    missing_mask = df[target_cols].isnull().any(axis=1)
    missing_rows = df[missing_mask]

    if missing_rows.empty:
        st.write("No missing values detected.")
        return df

    st.write(f"Found {len(missing_rows)} rows with missing values. Applying interpolation... \n")

    # Apply interpolation function
    df = interpolate_missing_values(df, location_col, target_cols)

    return df

# Use Time-based interpolation to handle missing values
Final_Merged_Dataset_Cleaned = handle_missing_values(Final_Merged_Dataset, location_col="Location", target_cols=["Average Temperature", "Temp_low", "Temp_high", "Average Humidity",
                                                                                       "Day Avg_temp", "Day Min_temp", "Day Max_temp", "Day Avg_Humidity", "Temp_change"])

Final_Merged_Dataset_Cleaned.to_csv("Final_Merged_Dataset_Cleaned.csv", index=False)

st.write("The final dataset: \n")
Final_Merged_Dataset_Cleaned.head()

# Summary of final dataset
Final_Merged_Dataset_Cleaned.info()

# Summay of final dataset
st.write(Final_Merged_Dataset_Cleaned.describe())

# Check for missing values in final dataset
st.write(Final_Merged_Dataset_Cleaned.isnull().sum())

# Copy dataset fto a new dataframe
Final_df = Final_Merged_Dataset_Cleaned.copy()

# Convert Date to datetime format
Final_df['Date'] = pd.to_datetime(Final_df['Date'])

st.write(Final_df.head())

# Assuming Final_df is already defined
numerical_cols = Final_df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(12, 12))
for i, col in enumerate(numerical_cols):
    plt.subplot(4, 3, i + 1)  # Adjust subplot layout as needed
    sns.histplot(Final_df[col], bins=20)
    plt.title(col)
plt.suptitle('Histograms of Numerical Features')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap
st.pyplot()

# Identify outliers using boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=Final_df[numerical_cols])
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Features")
st.pyplot()

# Get unique locations
locations = Final_df['Location'].unique()

# Plot the trend for each location
plt.figure(figsize=(10, 6))
for location in locations:
    location_data = Final_df[Final_df['Location'] == location]
    plt.plot(location_data['Date'], location_data['Number of Insects'], marker='o', linestyle='-', alpha=0.7, label=location)

plt.xlabel("Date")
plt.ylabel("Number of Insects")
plt.title("Trend of Insect Count Over Time by Location")
plt.xticks(rotation=45)
plt.legend(title="Location")
plt.grid(True)
plt.tight_layout()
st.pyplot()

# Get unique locations
locations = Final_df['Location'].unique()

# Plot the trends for each location in separate subplots
num_locations = len(locations)
fig, axes = plt.subplots(num_locations, 1, figsize=(10, 5 * num_locations), sharex=True)
fig.suptitle("Trend of Insect Count and New Catches Over Time by Location", fontsize=16)

for i, location in enumerate(locations):
    location_data = Final_df[Final_df['Location'] == location]

    ax = axes[i]
    ax.plot(location_data['Date'], location_data['Number of Insects'], marker='o', linestyle='-', alpha=0.7, label='Number of Insects', color='skyblue')
    ax.plot(location_data['Date'], location_data['New Catches'], marker='s', linestyle='--', alpha=0.7, label='New Catches', color='salmon')
    ax.set_ylabel("Count")
    ax.set_title(f"Location: {location}")
    ax.legend()
    # ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7) # Customize major grid lines
    # ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5) # Add minor grid lines
    ax.grid(True, alpha=0.5)
    ax.set_ylim(0, 6)

axes[-1].set_xlabel("Date")
plt.xticks(rotation=45)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
st.pyplot()

# Group by Date and sum 'Number of Insects' and 'New Catches'
daily_summary = Final_df.groupby('Date')[['Number of Insects', 'New Catches']].sum().reset_index()

# Plotting the trends
plt.figure(figsize=(12, 6))

plt.plot(daily_summary['Date'], daily_summary['Number of Insects'], marker='o', linestyle='-', color='skyblue', label='Total Number of Insects')
plt.plot(daily_summary['Date'], daily_summary['New Catches'], marker='s', linestyle='--', color='salmon', label='Total New Catches')

plt.xlabel("Date")
plt.ylabel("Total Count")
plt.title("Trend of Total Insect Count and New Catches Over Time (All Locations)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot()

# Group by Time and sum 'Number of Insects' and 'New Catches'
time_summary = Final_df.groupby('Time')[['Number of Insects', 'New Catches']].sum().reset_index()

# Sort by time for proper plotting order
time_summary['Time_dt'] = pd.to_datetime(time_summary['Time'], format='%H:%M:%S')
time_summary = time_summary.sort_values(by='Time_dt')

# Plotting the trends
plt.figure(figsize=(12, 6))
plt.plot(time_summary['Time'], time_summary['Number of Insects'], marker='o', linestyle='-', color='skyblue', label='Total Number of Insects')
plt.plot(time_summary['Time'], time_summary['New Catches'], marker='s', linestyle='--', color='salmon', label='Total New Catches')
plt.xlabel("Time")
plt.ylabel("Total Count")
plt.title("Trend of Total Insect Count and New Catches Over Time (All Locations)")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot()

# Group by 'Location'
grouped = Final_df.groupby('Location')

# Calculate the number of locations
num_locations = len(grouped)

# Determine the number of rows and columns for subplots
num_rows = (num_locations + 1) // 2  # Adjust as needed for layout
num_cols = min(num_locations, 2)  # Display up to 2 plots per row

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()  # Flatten the axes array for easy indexing

# Iterate through each location and plot
for i, (location, group) in enumerate(grouped):
    # Sort the group by date
    group = group.sort_values(by='Date')

    # Calculate the 7-day rolling average
    group['Rolling_Avg'] = group['Number of Insects'].rolling(window=7, min_periods=1).mean()

    # Plot the original data and the moving average
    ax = axes[i]
    ax.plot(group['Date'], group['Number of Insects'], label='Original Data', alpha=0.6)
    ax.plot(group['Date'], group['Rolling_Avg'], label='7-Day Moving Average', color='red')
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Insects")
    ax.set_title(f"Moving Average Trend - {location}")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

# Remove any unused subplots
if num_locations < len(axes):
    for i in range(num_locations, len(axes)):
        fig.delaxes(axes[i])

plt.tight_layout()
st.pyplot()

# Group by 'Location'
grouped = Final_df.groupby('Location')

# Create subplots for each location
num_locations = len(grouped)
num_rows = (num_locations + 1) // 2
num_cols = min(num_locations, 2)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()

# Iterate through each location and plot
for i, (location, group) in enumerate(grouped):
    ax = axes[i]
    ax.plot(group['Date'], group['Number of Insects'], label='Number of Insects')

    # Identify event dates for the current location
    event_dates_location = group[group['Event'] != 0]['Date']

    # Plot vertical lines for event dates at this location
    for date in event_dates_location:
        ax.axvline(x=date, color='r', linestyle='--', label='Event' if date == event_dates_location.iloc[0] else "")

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Insects")
    ax.set_title(f"Insects Over Time - {location}")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

# Remove any unused subplots
if num_locations < len(axes):
    for j in range(num_locations, len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
st.pyplot()

# Get unique locations for the legend
locations = Final_df['Location'].unique()

# Create a single plot
plt.figure(figsize=(15, 7))

# Iterate through each location and plot its insect count
for location in locations:
    group = Final_df[Final_df['Location'] == location].sort_values(by='Date')
    plt.plot(group['Date'], group['Number of Insects'], label=f'Insects - {location}', marker='o', alpha=0.7)

    # Identify event dates for the current location
    event_dates_location = group[group['Event'] != 0]['Date']

    # Plot vertical lines for event dates at this location
    for date in event_dates_location:
        plt.axvline(x=date, color='r', linestyle='--', alpha=0.5)

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Number of Insects")
plt.title("Number of Insects Over Time Across All Locations with Event Markers")
plt.legend(title='Location', loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
st.pyplot()

# Compare average number of insects on days with vs. without events ---
has_event = Final_df[Final_df['Event'] != 0]['Number of Insects']
no_event = Final_df[Final_df['Event'] == 0]['Number of Insects']

avg_with_event = has_event.mean() if not has_event.empty else 0
avg_without_event = no_event.mean() if not no_event.empty else 0

st.write(f"\nAverage number of insects on days with events: {avg_with_event:.2f}")
st.write(f"Average number of insects on days without events: {avg_without_event:.2f}")

# Create a bar plot for comparison
event_categories = ['With Event', 'Without Event']
average_insects = [avg_with_event, avg_without_event]

plt.figure(figsize=(6, 4))
plt.bar(event_categories, average_insects, color=['skyblue', 'lightcoral'])
plt.ylabel("Average Number of Insects")
plt.title("Comparison of Average Insects (With vs. Without Events)")
plt.tight_layout()
st.pyplot()

# Event-based comparison
plt.figure(figsize=(8, 6))
sns.boxplot(x=Final_df['Event'], y=Final_df['Number of Insects'])
plt.title("Insect Count Distribution by Event")
st.pyplot()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(Final_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
st.pyplot()

# 7. Scatter plots: Temperature vs Insects, Humidity vs Insects
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=Final_df['Average Temperature'], y=Final_df['Number of Insects'], ax=axes[0])
axes[0].set_title("Temperature vs Insect Count")
sns.scatterplot(x=Final_df['Average Humidity'], y=Final_df['Number of Insects'], ax=axes[1])
axes[1].set_title("Humidity vs Insect Count")
st.pyplot()

# Melt the dataframe to separate 'Number of Insects' and 'New Catches'
df_melted = pd.melt(Final_df,
                    id_vars=['Date', 'Time', 'Event', 'Location'],
                    value_vars=['Number of Insects', 'New Catches'],
                    var_name='Insect_Type',
                    value_name='Count')

# Location-wise comparison with color separation
plt.figure(figsize=(10, 6))
sns.barplot(x='Location', y='Count', hue='Insect_Type', data=df_melted, estimator=sum, errorbar=None)
plt.xticks(rotation=45, ha='right')
plt.title("Total Insects and New Catches per Location")
plt.xlabel("Location")
plt.ylabel("Count")
plt.tight_layout()
st.pyplot()

# Line plot: Date vs Number of Insects
plt.figure(figsize=(10, 5))
sns.lineplot(x=Final_df['Date'], y=Final_df['Number of Insects'])
plt.xticks(rotation=45)
plt.title("Number of Insects Over Time")
st.pyplot()

# Line plot: Time vs Number of Insects
plt.figure(figsize=(10, 5))
# Sort by time for the line plot to make sense
Final_df['Time_dt'] = pd.to_datetime(Final_df['Time'], format='%H:%M:%S')
Final_df_sorted_by_time = Final_df.sort_values(by='Time_dt')
sns.lineplot(x=Final_df_sorted_by_time['Time'].astype(str), y=Final_df_sorted_by_time['Number of Insects'])
plt.xticks(rotation=45, ha='right')
plt.xlabel("Time")
plt.ylabel("Number of Insects")
plt.title("Number of Insects Over Time")
st.pyplot()

# Define weather-related variables
weather_vars = ['Average Temperature', 'Temp_low', 'Temp_high', 'Average Humidity',
                'Day Avg_temp', 'Day Min_temp', 'Day Max_temp', 'Day Avg_Humidity', 'Temp_change']

# 2. Boxplots for weather variables to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=Final_df[weather_vars])
plt.xticks(rotation=90)
plt.title("Boxplot of Weather Features")
st.pyplot()

# Group by date and calculate the mean for each weather variable
daily_weather = Final_df.groupby('Date')[weather_vars].mean()

# Line plots for daily average of weather variables over time
plt.figure(figsize=(12, 6))
for var in weather_vars:
    plt.plot(daily_weather.index, daily_weather[var], label=var, alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Average Values")
plt.title("Daily Trends of Weather Variables Over Time")
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
st.pyplot()

# 4. Scatter plots: Weather variables vs Insect count
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, var in enumerate(weather_vars):
    sns.scatterplot(x=df[var], y=df['Number of Insects'], ax=axes[i])
    axes[i].set_title(f"{var} vs Insect Count")
plt.tight_layout()
st.pyplot()

# 6. Boxplots: Compare temperature and humidity across months
Final_df['Month'] = Final_df['Date'].dt.month
plt.figure(figsize=(10, 5))
sns.boxplot(x=Final_df['Month'], y=Final_df['Average Temperature'])
plt.title("Monthly Distribution of Average Temperature")
plt.xlabel("Month")
plt.ylabel("Average Temperature")
st.pyplot()

st.write("\n")

plt.figure(figsize=(10, 5))
sns.boxplot(x=Final_df['Month'], y=Final_df['Average Humidity'])
plt.title("Monthly Distribution of Average Humidity")
plt.xlabel("Month")
plt.ylabel("Average Humidity")
st.pyplot()
