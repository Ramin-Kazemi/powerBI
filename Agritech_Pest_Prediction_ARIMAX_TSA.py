def main():
    #!/usr/bin/env python
    # coding: utf-8

    # <a href="https://colab.research.google.com/github/raz0208/Agritech-Pest-Prediction/blob/main/Agritech_Pest_Prediction_ARIMAX_TSA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

    # ## Time Series Analysis For Agritech Pest Prediction

    # ## **ARIMAX** Model Implementation:

    # In[4]:


    # # Uninstall potentially conflicting versions first
    # !pip uninstall -y pmdarima numpy

    # # Install the latest NumPy 1.x version (e.g., 1.26.4)
    # !pip install numpy==1.26.4

    # # Now install pmdarima (hopefully it links against NumPy 1.26.4)
    # # Use --no-cache-dir just to be safe
    # !pip install --no-cache-dir pmdarima==2.0.4


    # In[5]:


    # import required libaraies
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from pmdarima import auto_arima
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


    # In[6]:


    # Load datasets
    FeatureExtracted_df = pd.read_csv('/content/FeatureExtracted_dataset.csv')


    # In[7]:


    # Display basic info for  datasets FeatureExtracted_dataset
    print(FeatureExtracted_df.head(), '\n')
    print(FeatureExtracted_df.info())


    # In[8]:


    # Convert 'Date' column to datetime format
    FeatureExtracted_df['Date'] = pd.to_datetime(FeatureExtracted_df['Date'])


    # In[9]:


    # Drop some columns
    FeatureExtracted_df.drop(['Lag_1', 'Lag_2', 'Lag_3'], axis=1, inplace=True)

    # Group and aggregate
    aggregated_df = FeatureExtracted_df.groupby('Date').agg({
        'Number of Insects': 'sum',
        'New Catches': 'sum',
        'Event': 'max',
        'Average Temperature': 'mean',
        'Temp_low': 'mean',
        'Temp_high': 'mean',
        'Average Humidity': 'mean',
        'Day Avg_temp': 'mean',
        'Day Min_temp': 'mean',
        'Day Max_temp': 'mean',
        'Day Avg_Humidity': 'mean',
        'Temp_change': 'mean',
        'Year': 'first',
        'Month': 'first',
        'Day': 'first',
        'Weekday': 'first',
        # 'Lag_1': 'mean',
        # 'Lag_2': 'mean',
        # 'Lag_3': 'mean'
    }).reset_index()

    print(aggregated_df.head(10))

    # # save to csv file
    # aggregated_df.to_csv('aggregated_dataset.csv', index=False)


    # In[10]:


    # Convert "Date" to datetime format
    aggregated_df["Date"] = pd.to_datetime(aggregated_df["Date"])

    # Set "Date" as the index
    aggregated_df.set_index("Date", inplace=True)

    # Check for Nan_Value
    print(aggregated_df.isna().sum())


    # ### Visualization trends

    # In[11]:


    # Features to plot
    features_to_plot = [
        'Number of Insects',
        'Average Temperature',
        'Average Humidity',
        'Day Avg_temp',
        'Day Avg_Humidity',
        'Temp_change'
    ]

    # Plotting
    plt.figure(figsize=(12, 5))
    for feature in features_to_plot:
        plt.plot(aggregated_df.index, aggregated_df[feature], label=feature)

    plt.title("Number of Insects and Meteorological Factors Over Time")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


    # In[12]:


    # Plot ACF and PACF to examine autocorrelation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plot_acf(aggregated_df['Number of Insects'], lags=40, ax=ax1)
    plot_pacf(aggregated_df['Number of Insects'], lags=25, ax=ax2)
    plt.show()


    # ### Check for Stationarity by Augmented Dickey-Fuller (ADF Test)

    # In[13]:


    # Perform Augmented Dickey-Fuller (ADF) test to check for stationarity
    def check_stationarity(series):
        """
        Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

        Parameters:
            series (pd.Series): Time series data (e.g., insect count indexed by date).

        Returns:
            dict: Contains ADF statistic, p-value, critical values, and interpretation.
        """
        adf_result = adfuller(series.dropna())

        result = {
            "ADF Statistic": adf_result[0],
            "p-value": adf_result[1],
            "# Lags Used": adf_result[2],
            "# Observations": adf_result[3],
            "Critical Values": adf_result[4]
        }

        # Interpretation
        result["Stationary"] = adf_result[1] < 0.05

        print("ADF Statistic:", result["ADF Statistic"])
        print("p-value:", result["p-value"])
        print("# Lags Used:", result["# Lags Used"])
        print("# Observations:", result["# Observations"])
        print("Critical Values:")
        for key, value in result["Critical Values"].items():
            print(f"   {key}: {value}")

        if result["Stationary"]:
            print("✅ Series is stationary (p < 0.05)")
        else:
            print("⚠️ Series is not stationary (p >= 0.05)")

        #return result

    check_stationarity(aggregated_df['Number of Insects'])


    # ### Apply Differencing (if needed) and check Stationarity

    # In[14]:


    # ##--- ### Method 1 to make stationary ### ---##

    # # Make stationary function
    # def make_stationary(series, max_diff=100, verbose=True):
    #     """
    #     Automatically differences a series until it becomes stationary (ADF p-value <= 0.05).

    #     Parameters:
    #         series (pd.Series): The time series to make stationary
    #         max_diff (int): Maximum differencing order to try
    #         verbose (bool): Whether to print ADF results

    #     Returns:
    #         stationary_series (pd.Series): Differenced series
    #         d (int): The order of differencing applied
    #     """
    #     for d in range(max_diff + 1):
    #         if d > 0:
    #             differenced = series.diff(d).dropna()
    #         else:
    #             differenced = series.dropna()

    #         result = adfuller(differenced, autolag='AIC')
    #         p_value = result[1]

    #         if verbose:
    #             print(f'Differencing Order: {d}, ADF p-value: {p_value}')

    #         if p_value <= 0.05:
    #             if verbose:
    #                 print(f"✅ Series is stationary with differencing order d={d}\n")
    #             return differenced, d

    #     print("⚠️ Series is still not stationary after maximum differencing.")
    #     return differenced, max_diff

    # # Call function
    # df_diff, num_diffs = make_stationary(DateAgg_df['Number of Insects'])


    # In[15]:


    # # Plot differenced data
    # plt.figure(figsize=(14, 5))
    # plt.plot(df_diff, marker='o', linestyle='-')
    # plt.title(f'Differenced Insect Count (Differencing {num_diffs})')
    # plt.xlabel('Date')
    # plt.ylabel('Differenced Value')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()


    # In[16]:


    ##--- ### Method 2 to make stationary ### ---##

    # Function to apply Differencing and check Stationarity
    def make_stationary(ts, max_diff=6, verbose=True):
        d = 0
        result = adfuller(ts.dropna())
        if verbose:
            print(f"ADF Test p-value (d={d}): {result[1]:.4f}")
        while result[1] > 0.05 and d < max_diff:
            ts = ts.diff().dropna()
            d += 1
            result = adfuller(ts)
            if verbose:
                print(f"ADF Test p-value (d={d}): {result[1]:.4f}")
        if verbose and result[1] <= 0.05:
            print(f"✅ Series is stationary at d={d}")
        return ts, d

    # Call the function
    df_diff, num_diffs = make_stationary(aggregated_df['Number of Insects'])


    # In[17]:


    # Plot differenced data
    plt.figure(figsize=(14, 5))
    plt.plot(df_diff, marker='o', linestyle='-')
    plt.grid(True)
    plt.title(f'Differenced Insect Count (Differencing {num_diffs})')
    plt.xlabel('Date')
    plt.ylabel('Differenced Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # In[18]:


    # Choose meteorological features for ARIMAX
    exog_features = [
        'New Catches',
        'Event',
        'Average Temperature',
        'Temp_low',
        'Temp_high',
        'Average Humidity',
        'Day Min_temp',
        'Day Max_temp',
        'Day Avg_temp',
        'Day Avg_Humidity',
        'Temp_change',
        # 'Year',
        # 'Month',
        # 'Day',
        # 'Weekday'
    ]

    exogenous_data = aggregated_df[exog_features]

    # Optional: Check for NaNs in exogenous data
    print("\nMissing values in exogenous features:\n")
    print(exogenous_data.isnull().sum())


    # In[19]:


    # Make target stationary and determine best differencing order
    target_series = aggregated_df['Number of Insects']

    # # Drop initial rows to align target and exogenous data after differencing
    # aligned_exog = exogenous_data.iloc[num_diffs:]

    # Use auto_arima to find best ARIMAX configuration
    arimax_model = auto_arima(
        y=target_series,  # Use original series, auto_arima will apply differencing internally
        exogenous=exogenous_data,
        seasonal=False,
        d=num_diffs,  # Let auto_arima determine differencing
        #max_d=2,
        start_p=1,
        start_q=1,
        max_p=5,
        max_q=5,
        stepwise=True,
        trace=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    # Model summary
    print(arimax_model.summary())


    # In[20]:


    from datetime import timedelta

    # Forecasting next X days
    n_periods = 7

    # Average over the last 7 days of training
    recent_avg = exogenous_data.tail(7).mean()

    future_exogenous_data = pd.DataFrame(
        [recent_avg] * 7,
        columns=exogenous_data.columns,
        index=pd.date_range(start=aggregated_df.index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
    )

    # Create future exogenous variables for 7 days (this is essential!)
    future_exog = future_exogenous_data[:n_periods]

    # Forecast with confidence intervals
    forecast, conf_int = arimax_model.predict(n_periods=n_periods, exogenous=future_exog, return_conf_int=True)

    # Create forecast index
    last_date = aggregated_df.index[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=n_periods, freq='D')

    # Build forecast DataFrame
    forecast_df = pd.DataFrame({
        'Forecast': forecast,
        'Lower_CI': conf_int[:, 0],
        'Upper_CI': conf_int[:, 1]
    }, index=forecast_index)

    # Plotting
    plt.figure(figsize=(14, 5))
    plt.plot(aggregated_df.index, aggregated_df['Number of Insects'], label='Historical', marker='o')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green', marker='o')
    plt.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='lightgreen', alpha=0.4)
    plt.title('Forecast of Insect Counts (Next 7 Days)')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ## ARIMAX model performance
    # 
    # ### Calculate Error Metrics

    # In[21]:


    # Actual and predicted values
    actual = aggregated_df['Number of Insects']
    predicted = pd.Series(arimax_model.predict_in_sample(), index=aggregated_df.index)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual, predicted)

    # Root Mean Squared Error (RMSE)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100

    # Print results
    print(f"📊 Model Performance Metrics:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAPE = {mape:.2f}%")


    # ### Train vs. Fitted Plot
    # This shows how closely the model’s fitted values match the actual data during training.

    # In[22]:


    # Get in-sample predictions
    fitted_values = pd.Series(arimax_model.predict_in_sample(), index=aggregated_df.index)

    # Plot actual vs fitted
    plt.figure(figsize=(14, 5))
    plt.plot(aggregated_df.index, aggregated_df['Number of Insects'], label='Actual', marker='o')
    plt.plot(fitted_values.index, fitted_values, label='Fitted', color='orange', linestyle='--', marker='x')
    plt.title('Actual vs Fitted Insect Counts (Training Performance)')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ### Residual Plot (Error Over Time)
    # This shows the difference between actual and predicted values. Ideally, residuals should look like white noise (no pattern).

    # In[23]:


    # Calculate residuals
    residuals = aggregated_df['Number of Insects'] - fitted_values

    # Plot residuals
    plt.figure(figsize=(14, 4))
    plt.plot(residuals.index, residuals, label='Residuals', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ### Histogram + Q-Q Plot of Residuals (Check Normality)
    # This is to check if residuals are normally distributed — a good sign for model assumptions.

    # In[24]:


    import scipy.stats as stats

    # Histogram of residuals
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title('Histogram of Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n")

    # Q-Q Plot
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals.dropna(), dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ## ARIMAX with train and test

    # In[27]:


    # Define target and exogenous features
    target = 'Number of Insects'
    exog_features = [
        'Average Temperature',
        'Average Humidity',
        'Day Avg_temp',
        'Day Avg_Humidity',
        'Temp_change'
    ]

    # Drop NA values due to differencing (if needed)
    data = aggregated_df[[target] + exog_features].dropna()


    # In[28]:


    # Split into train and test (90/10 split)
    split_index = int(len(data) * 0.9)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    # Define endogenous and exogenous variables
    y_train = train[target]
    X_train = train[exog_features]
    y_test = test[target]
    X_test = test[exog_features]


    # In[30]:


    # Fit ARIMAX model using auto_arima
    arimax_model = auto_arima(
        y_train,
        exogenous=X_train,
        seasonal=False,
        d=num_diffs,  # Let auto_arima determine differencing
        max_p=5,
        max_q=5,
        stepwise=True,
        suppress_warnings=True,
        trace=True,
        error_action='ignore'
    )

    print(arimax_model.summary())


    # In[31]:


    # Forecast on test set
    n_periods = len(y_test)
    forecast = arimax_model.predict(n_periods=n_periods, exogenous=X_test)

    # Convert forecast to a pandas Series for easy plotting
    forecast_series = pd.Series(forecast, index=y_test.index)

    # Evaluation
    mae = mean_absolute_error(y_test, forecast)
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    r2 = r2_score(y_test, forecast)

    print(f"\nModel Evaluation:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")


    # In[33]:


    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual', marker='o')
    plt.plot(forecast_series, label='Forecast', linestyle='--', marker='x')
    plt.title('ARIMAX: Actual vs Forecasted Insect Counts')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


    # In[40]:


    # Create time axis
    forecast_index = y_test.index

    # Plot actual vs forecast
    plt.figure(figsize=(12, 5))
    plt.plot(y_train.index, y_train, label='Train', color='blue')
    plt.plot(y_test.index, y_test, label='Test (Actual)', color='black')
    plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='--')
    plt.title('Historical Insect Count and Forecast')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


    # ### Train vs. Fitted Plot
    # This shows how closely the model’s fitted values match the actual data during training.

    # In[53]:


    # Get in-sample predictions for training data
    fitted_values = arimax_model.predict_in_sample(exogenous=X_train)

    # Plot actual vs fitted
    plt.figure(figsize=(14, 5))
    plt.plot(aggregated_df.index, aggregated_df['Number of Insects'], label='Actual', marker='o')
    plt.plot(fitted_values.index, fitted_values, label='Fitted', color='orange', linestyle='--', marker='x')
    plt.plot(y_test.index, y_test, label='Test (Actual)', color='skyblue', marker='o')
    plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='--', marker='x')
    plt.title('Actual vs Fitted Insect Counts (Training Performance)')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ### Residual Plot (Error Over Time)

    # In[55]:


    # Calculate residuals
    residuals = y_train - fitted_values

    plt.figure(figsize=(14, 4))
    plt.plot(residuals.index, residuals, marker='o')
    plt.title('Residuals (Train Set)')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.axhline(0, color='red', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()


    # ### Histogram + Q-Q Plot of Residuals (Normality Check)

    # In[64]:


    # Histogram
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, bins=20)
    plt.title('Histogram of Residuals')
    plt.grid(True)

    # Q-Q Plot
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
