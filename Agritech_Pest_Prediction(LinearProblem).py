def main():
    #!/usr/bin/env python
    # coding: utf-8

    # <a href="https://colab.research.google.com/github/raz0208/Agritech-Pest-Prediction/blob/main/Agritech_Pest_Prediction(LinearProblem).ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

    # ## Agritech Pest Prediction and Classification

    # ### Preprocessed Dataset Overview:
    # - Total Rows: 153
    # - Total Columns: 15
    # - Data Types:
    #   - 3 Object (String) Columns: `Date`, `Time`, `Location`
    #   - 11 Float Columns: `Number of Insects`, `New Catches`, `Average Temperature`, `Temp_low`, `Temp_high`, `Average Humidity`, `Day Avg_temp`, `Day Min_temp`, `Day Max_temp`, `Day Avg_Humidity`, and `Temp_change`
    #   - 1 Integer Column: `Event`, Binary indicator (0 or 1), representing an event occurrence.
    # 
    # ### Column Breakdown:
    # 1. Date & Time: Represent the timestamp of each record.
    # 2. Number of Insects & New Catches: Key target variables for regression and classification.
    # 3. Event: Binary indicator, possibly for significant occurrences.
    # 4. Location: Specifies data collection sites (Cicalino1, etc.).
    # 5. Temperature & Humidity Variables:
    #   - Includes averages, daily minimum/maximum, and changes.
    # 6. Temp_change: Measures temperature variation.
    # 
    # ### Observations:
    # - No missing values.
    # - The dataset is structured with meteorological and pest count variables, ideal for predictive modeling.

    # ### Import required libraries and read the data

    # In[1]:


    # Import required libraries
    import os
    import pandas as pd
    import numpy as np

    ## Libraries for visualization
    import matplotlib.pyplot as plt
    import seaborn as sns


    # In[3]:


    # Read data
    Final_Agritech_Pest_DS = pd.read_csv('/content/Final_Merged_Dataset_Cleaned.csv')

    # Showing first rows of preprocessed Agritech Pest dataset
    print(Final_Agritech_Pest_DS.head())


    # In[4]:


    # Summary of dataset
    Final_Agritech_Pest_DS.info()


    # In[5]:


    # Summary of dataset
    print(Final_Agritech_Pest_DS.describe())


    # In[6]:


    # Check for missing values
    print(Final_Agritech_Pest_DS.isnull().sum())


    # ## Feature Extracting

    # In[7]:


    # Copy dataset in a dataframe
    df = Final_Agritech_Pest_DS.copy()

    # Convert "Date" to datetime format and extract useful time-based features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday  # Monday=0, Sunday=6

    # Convert "Time" to datetime format and extract hour and minute
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time

    # Create lag features (previous days' insect counts)
    df = df.sort_values(by=["Location", "Date"])  # Ensure sorting for lag features
    df["Lag_1"] = df.groupby("Location")["Number of Insects"].shift(1)
    df["Lag_2"] = df.groupby("Location")["Number of Insects"].shift(2)
    df["Lag_3"] = df.groupby("Location")["Number of Insects"].shift(3)

    # Fill missing values in lag features with 0 (assuming no prior data)
    df[["Lag_1", "Lag_2", "Lag_3"]] = df[["Lag_1", "Lag_2", "Lag_3"]].fillna(0)

    # Save new dataset in a csv file
    df.to_csv("FeatureExtracted_dataset.csv", index=False)

    # Display the updated dataset with new features
    print(df)


    # In[8]:


    # Set plot style
    sns.set_style("whitegrid")

    # --- Time-Based Features Visualization ---

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Insect counts by month
    sns.boxplot(x="Month", y="Number of Insects", data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Monthly Variation in Insect Counts")

    # Insect counts by weekday
    sns.boxplot(x="Weekday", y="Number of Insects", data=df, ax=axes[0, 1])
    axes[0, 1].set_title("Weekday Variation in Insect Counts")

    # Insect counts over time
    df_sorted = df.sort_values(by=["Date"])
    sns.lineplot(x="Date", y="Number of Insects", hue="Location", data=df_sorted, ax=axes[1, 0])
    axes[1, 0].set_title("Trend of Insect Counts Over Time")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Insect counts by year (if applicable)
    if df["Year"].nunique() > 1:
        sns.boxplot(x="Year", y="Number of Insects", data=df, ax=axes[1, 1])
        axes[1, 1].set_title("Yearly Variation in Insect Counts")

    plt.tight_layout()
    plt.show()

    print("\n")
    # --- Lag Features Visualization ---

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[["Number of Insects", "Lag_1", "Lag_2", "Lag_3"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Between Insect Counts and Lag Features")

    plt.show()


    # ## Model Implementation: Regression Problem
    # - Linear Regression, Ridge Regression, and Lasso Regression.
    # - Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost regressors.
    # - Support Vector Regression (SVR) and K-Nearest Neighbors (KNN) Regression.

    # In[9]:


    #!pip uninstall catboost


    # In[10]:


    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    #from catboost import CatBoostRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


    # ### Linear Regression, Ridge Regression, and Lasso Regression

    # In[27]:


    # Copy dataset to nw dataframe
    FeatureExtracted_dataset = pd.read_csv('/content/FeatureExtracted_dataset.csv')
    df0 = FeatureExtracted_dataset.copy()

    # Features and Target Variable
    X0 = df0.drop(columns=["Number of Insects", "Date", "Time", "Location"])  # Drop non-numeric columns
    y0 = df0["Number of Insects"]

    # Split into train and test sets
    X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2, random_state=42)

    # Initialize models
    lin_reg = LinearRegression()
    ridge_reg = Ridge(alpha=1.0)  # Default alpha = 1.0 (can be tuned)
    lasso_reg = Lasso(alpha=0.1)  # Default alpha = 0.1 (can be tuned)

    # Train models
    lin_reg.fit(X0_train, y0_train)
    ridge_reg.fit(X0_train, y0_train)
    lasso_reg.fit(X0_train, y0_train)

    # Predict on test set
    y0_pred_lin = lin_reg.predict(X0_test)
    y0_pred_ridge = ridge_reg.predict(X0_test)
    y0_pred_lasso = lasso_reg.predict(X0_test)

    # Evaluation function
    def evaluate_model(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{model_name} Results:")
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}\n")
        return mae, mse, r2

    # Evaluate all models
    results0 = {}
    results0["Linear Regression"] = evaluate_model(y0_test, y0_pred_lin, "Linear Regression")
    results0["Ridge Regression"] = evaluate_model(y0_test, y0_pred_ridge, "Ridge Regression")
    results0["Lasso Regression"] = evaluate_model(y0_test, y0_pred_lasso, "Lasso Regression")


    # In[28]:


    # Visualization of Model Performance
    models0 = list(results0.keys())
    mae_values0 = [results0[m][0] for m in models0]
    mse_values0 = [results0[m][1] for m in models0]
    r2_values0 = [results0[m][2] for m in models0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.barplot(x=models0, y=mae_values0, ax=axes[0], palette="Blues", hue=models0, legend=False)
    axes[0].set_title("Mean Absolute Error (MAE)")
    axes[0].set_ylabel("MAE")

    sns.barplot(x=models0, y=mse_values0, ax=axes[1], palette="Greens", hue=models0, legend=False)
    axes[1].set_title("Mean Squared Error (MSE)")
    axes[1].set_ylabel("MSE")

    sns.barplot(x=models0, y=r2_values0, ax=axes[2], palette="Reds", hue=models0, legend=False)
    axes[2].set_title("R² Score")
    axes[2].set_ylabel("R²")

    plt.tight_layout()
    plt.show()


    # In[29]:


    # Visualization of Predictions vs Actual
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].scatter(y0_test, y0_pred_lin, alpha=0.5, color='blue')
    ax[0].plot([y0_test.min(), y0_test.max()], [y0_test.min(), y0_test.max()], '--', color='red')
    ax[0].set_title("Linear Regression: Predictions vs Actual")
    ax[0].set_xlabel("Actual Values")
    ax[0].set_ylabel("Predicted Values")

    ax[1].scatter(y0_test, y0_pred_ridge, alpha=0.5, color='green')
    ax[1].plot([y0_test.min(), y0_test.max()], [y0_test.min(), y0_test.max()], '--', color='red')
    ax[1].set_title("Ridge Regression: Predictions vs Actual")
    ax[1].set_xlabel("Actual Values")
    ax[1].set_ylabel("Predicted Values")

    ax[2].scatter(y0_test, y0_pred_lasso, alpha=0.5, color='purple')
    ax[2].plot([y0_test.min(), y0_test.max()], [y0_test.min(), y0_test.max()], '--', color='red')
    ax[2].set_title("Lasso Regression: Predictions vs Actual")
    ax[2].set_xlabel("Actual Values")
    ax[2].set_ylabel("Predicted Values")

    plt.tight_layout()
    plt.show()


    # ### Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost regressors

    # In[30]:


    # Copy dataset to nw dataframe
    FeatureExtracted_dataset = pd.read_csv('/content/FeatureExtracted_dataset.csv')
    df1 = FeatureExtracted_dataset.copy()

    # Load the dataset (assuming 'df' is already preprocessed)
    X1 = df1.drop(columns=["Number of Insects", "Date", "Time", "Location"])  # Drop non-numeric columns
    y1 = df1["Number of Insects"]

    # Split into train and test sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

    # Initialize models
    dt_reg = DecisionTreeRegressor(random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    xgb_reg = XGBRegressor(n_estimators=100, random_state=42)
    #lgb_reg = LGBMRegressor(n_estimators=100, random_state=42)
    #cat_reg = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)

    models1 = {
        "Decision Tree": dt_reg,
        "Random Forest": rf_reg,
        "Gradient Boosting": gb_reg,
        "XGBoost": xgb_reg,
        #"LightGBM": lgb_reg,
        #"CatBoost": cat_reg
    }

    results1 = {}
    predictions1 = {}

    # Train and evaluate each model
    for name, model in models1.items():
        model.fit(X1_train, y1_train)
        y1_pred = model.predict(X1_test)
        predictions1[name] = y1_pred
        mae1 = mean_absolute_error(y1_test, y1_pred)
        mse1 = mean_squared_error(y1_test, y1_pred)
        r21 = r2_score(y1_test, y1_pred)
        results1[name] = (mae1, mse1, r21)
        print(f"{name} Results:")
        print(f"MAE: {mae1:.4f}, MSE: {mse1:.4f}, R²: {r21:.4f}\n")


    # In[31]:


    # Visualization of Predictions vs Actual
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (name, y1_pred) in enumerate(predictions1.items()):
        axes[i].scatter(y1_test, y1_pred, alpha=0.5)
        axes[i].plot([y1_test.min(), y1_test.max()], [y1_test.min(), y1_test.max()], '--', color='red')
        axes[i].set_title(f"{name}: Predictions vs Actual")
        axes[i].set_xlabel("Actual Values")
        axes[i].set_ylabel("Predicted Values")

    plt.tight_layout()
    plt.show()


    # In[32]:


    # Visualization of Model Performance
    models_list1 = list(results1.keys())
    mae_values1 = [results1[m][0] for m in models_list1]
    mse_values1 = [results1[m][1] for m in models_list1]
    r2_values1 = [results1[m][2] for m in models_list1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x=models_list1, y=mae_values1, ax=axes[0], hue=models_list1, legend=False, palette="Blues")
    axes[0].set_title("Mean Absolute Error (MAE)")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(x=models_list1, y=mse_values1, ax=axes[1], hue=models_list1, legend=False, palette="Greens")
    axes[1].set_title("Mean Squared Error (MSE)")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis='x', rotation=45)

    sns.barplot(x=models_list1, y=r2_values1, ax=axes[2], hue=models_list1, legend=False, palette="Reds")
    axes[2].set_title("R² Score")
    axes[2].set_ylabel("R²")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


    # ## Support Vector Regression (SVR) and K-Nearest Neighbors (KNN) Regression

    # In[33]:


    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler


    # In[34]:


    # Copy dataset to nw dataframe
    FeatureExtracted_dataset = pd.read_csv('/content/FeatureExtracted_dataset.csv')
    df2 = FeatureExtracted_dataset.copy()

    # Load the dataset (assuming 'df' is already preprocessed)
    X2 = df2.drop(columns=["Number of Insects", "Date", "Time", "Location"])  # Drop non-numeric columns
    y2 = df2["Number of Insects"]

    # Split into train and test sets
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    # Standardize the data for SVR and KNN
    scaler = StandardScaler()
    X2_train_scaled = scaler.fit_transform(X2_train)
    X2_test_scaled = scaler.transform(X2_test)

    # Initialize models
    svr_reg = SVR(kernel='rbf', C=100, gamma=0.1)
    knn_reg = KNeighborsRegressor(n_neighbors=5)

    models2 = {
        "Support Vector Regression": svr_reg,
        "K-Nearest Neighbors": knn_reg
    }

    results2 = {}
    predictions2 = {}

    # Train and evaluate each model
    for name, model in models2.items():
        model.fit(X2_train_scaled, y2_train)
        y2_pred = model.predict(X2_test_scaled)
        predictions2[name] = y2_pred
        mae2 = mean_absolute_error(y2_test, y2_pred)
        mse2 = mean_squared_error(y2_test, y2_pred)
        r22 = r2_score(y2_test, y2_pred)
        results2[name] = (mae2, mse2, r22)
        print(f"{name} Results:")
        print(f"MAE: {mae2:.4f}, MSE: {mse2:.4f}, R²: {r22:.4f}\n")


    # In[35]:


    # Visualization of Predictions vs Actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (name, y2_pred) in enumerate(predictions2.items()):
        axes[i].scatter(y2_test, y2_pred, alpha=0.5)
        axes[i].plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], '--', color='red')
        axes[i].set_title(f"{name}: Predictions vs Actual")
        axes[i].set_xlabel("Actual Values")
        axes[i].set_ylabel("Predicted Values")

    plt.tight_layout()
    plt.show()


    # In[36]:


    # Visualization of Model Performance
    models_list2 = list(results2.keys())
    mae_values2 = [results2[m][0] for m in models_list2]
    mse_values2 = [results2[m][1] for m in models_list2]
    r2_values2 = [results2[m][2] for m in models_list2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.barplot(x=models_list2, y=mae_values2, ax=axes[0], hue=models_list2, legend=False, palette="Blues")
    axes[0].set_title("Mean Absolute Error (MAE)")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(x=models_list2, y=mse_values2, ax=axes[1], hue=models_list2, legend=False, palette="Greens")
    axes[1].set_title("Mean Squared Error (MSE)")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis='x', rotation=45)

    sns.barplot(x=models_list2, y=r2_values2, ax=axes[2], hue=models_list2, legend=False, palette="Reds")
    axes[2].set_title("R² Score")
    axes[2].set_ylabel("R²")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


    # ## Model Impelementaion: Regression with time component

    # In[37]:


    # Required libararies for Time Series Forcasting models
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import STL


    # In[43]:


    # Load dataset
    FeatureExtracted_dataset = pd.read_csv('/content/FeatureExtracted_dataset.csv')
    TS_df = FeatureExtracted_dataset.copy()

    # Convert Date column to datetime
    TS_df["Date"] = pd.to_datetime(TS_df["Date"])  # Convert Date column to datetime
    TS_df.set_index("Date", inplace=True)  # Set Date as index
    TS_df = TS_df.sort_index()  # Ensure data is sorted by time

    # Plot Time Series
    plt.figure(figsize=(12, 5))
    plt.plot(TS_df.index, TS_df["Number of Insects"], label="Number of Insects")
    plt.xlabel("Date")
    plt.ylabel("Insect Count")
    plt.title("Time Series Plot of Insect Counts")
    plt.legend()
    plt.show()

    print("\n")

    # Seasonal Decomposition
    decomposition = seasonal_decompose(TS_df["Number of Insects"], model='additive', period=30)  # Adjust period based on data
    decomposition.plot()
    plt.show()

    print("\n")

    # ACF and PACF Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(TS_df["Number of Insects"], ax=axes[0], lags=50)
    plot_pacf(TS_df["Number of Insects"], ax=axes[1], lags=50)
    plt.show()

    print("\n")

    # Fourier Transform (Spectral Analysis)
    fft_vals = np.fft.fft(TS_df["Number of Insects"])
    frequencies = np.fft.fftfreq(len(fft_vals))

    plt.figure(figsize=(10, 5))
    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_vals[:len(frequencies)//2]))
    plt.title("Fourier Transform - Frequency Domain Analysis")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()


    # In[44]:


    # Augmented Dickey-Fuller (ADF) Test for Stationarity
    def adf_test(series):
        result = adfuller(series)
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        print("Critical Values:", result[4])
        if result[1] <= 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is non-stationary.")

    print("Augmented Dickey-Fuller Test:")
    adf_test(TS_df["Number of Insects"])

    print("\n")

    # Seasonal-Trend Decomposition using LOESS (STL)
    period = 5  # Adjust based on your time series frequency
    stl = STL(df["Number of Insects"], period=period)
    result3 = stl.fit()

    # Plot STL decomposition
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    result3.trend.plot(ax=axes[0], title="Trend")
    result3.seasonal.plot(ax=axes[1], title="Seasonality")
    result3.resid.plot(ax=axes[2], title="Residuals")
    plt.tight_layout()
    plt.show()


    # ### Regression Models with Time-Based Features
    # - Linear Regression
    # - Ridge Regression
    # - Lasso Regression
    # - Random Forest
    # - Gradient Boosting

    # In[46]:


    # Load dataset
    Time_df = pd.read_csv('/content/FeatureExtracted_dataset.csv')

    # Convert Date column to datetime
    Time_df["Date"] = pd.to_datetime(Time_df["Date"])  # Convert Date to datetime
    Time_df["Time"] = pd.to_datetime(Time_df["Time"], format='%H:%M:%S').dt.hour  # Convert Time to numeric (hour of the day)
    Time_df["Year"] = Time_df["Date"].dt.year
    Time_df["Month"] = Time_df["Date"].dt.month
    Time_df["Day"] = Time_df["Date"].dt.day
    Time_df["Weekday"] = Time_df["Date"].dt.weekday

    # Drop original Date column
    X4 = Time_df.drop(columns=["Number of Insects", "Date", "Location"])  # Drop target and non-numeric columns
    y4 = Time_df["Number of Insects"]


    # In[47]:


    # Split into train and test sets
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.20, random_state=42)

    # Initialize models
    models4 = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Train and evaluate each model
    results4 = {}
    predictions4 = {}
    for name, model in models4.items():
        model.fit(X4_train, y4_train)
        y4_pred = model.predict(X4_test)
        predictions4[name] = y4_pred
        mae4 = mean_absolute_error(y4_test, y4_pred)
        mse4 = mean_squared_error(y4_test, y4_pred)
        r24 = r2_score(y4_test, y4_pred)
        results4[name] = (mae4, mse4, r24)
        print(f"{name} Results:")
        print(f"MAE: {mae4:.4f}, MSE: {mse4:.4f}, R²: {r24:.4f}\n")


    # In[48]:


    # Visualization of Predictions vs Actual
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (name, y4_pred) in enumerate(predictions4.items()):
        axes[i].scatter(y4_test, y4_pred, alpha=0.5)
        axes[i].plot([y4_test.min(), y4_test.max()], [y4_test.min(), y4_test.max()], '--', color='red')
        axes[i].set_title(f"{name}: Predictions vs Actual")
        axes[i].set_xlabel("Actual Values")
        axes[i].set_ylabel("Predicted Values")

    plt.tight_layout()
    plt.show()


    # In[49]:


    # Visualization of Model Performance
    models_list4 = list(results4.keys())
    mae_values4 = [results4[m][0] for m in models_list4]
    mse_values4 = [results4[m][1] for m in models_list4]
    r2_values4 = [results4[m][2] for m in models_list4]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x=models_list4, y=mae_values4, ax=axes[0], palette="Blues", legend=False, hue=models4.keys())
    axes[0].set_title("Mean Absolute Error (MAE)")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(x=models_list4, y=mse_values4, ax=axes[1], palette="Greens", legend=False, hue=models4.keys())
    axes[1].set_title("Mean Squared Error (MSE)")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis='x', rotation=45)

    sns.barplot(x=models_list4, y=r2_values4, ax=axes[2], palette="Reds", legend=False, hue=models4.keys())
    axes[2].set_title("R² Score")
    axes[2].set_ylabel("R²")
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
