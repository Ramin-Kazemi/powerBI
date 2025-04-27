def main():
    #!/usr/bin/env python
    # coding: utf-8

    # <a href="https://colab.research.google.com/github/raz0208/Agritech-Pest-Prediction/blob/main/Agritech_Pest_Prediction_SARIMAX_TSA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

    # # Time Series Analysis For Agritech Pest Prediction

    # ### **SARIMA** Model Implementation:

    # In[58]:


    # # Uninstall potentially conflicting versions first
    # !pip uninstall -y pmdarima numpy

    # # Install the latest NumPy 1.x version (e.g., 1.26.4)
    # !pip install numpy==1.26.4

    # # Now install pmdarima (hopefully it links against NumPy 1.26.4)
    # # Use --no-cache-dir just to be safe
    # !pip install --no-cache-dir pmdarima==2.0.4


    # In[59]:


    # import required libaraies
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from scipy import stats
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from pmdarima import auto_arima
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


    # In[60]:


    # Load datasets
    FeatureExtracted_df = pd.read_csv('/content/FeatureExtracted_dataset.csv')


    # In[61]:


    # Display basic info for  datasets FeatureExtracted_dataset
    print(FeatureExtracted_df.head(), '\n')
    print(FeatureExtracted_df.info())


    # In[62]:


    # Convert 'Date' column to datetime format
    FeatureExtracted_df['Date'] = pd.to_datetime(FeatureExtracted_df['Date'])


    # In[63]:


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


    # In[64]:


    # Convert "Date" to datetime format
    aggregated_df["Date"] = pd.to_datetime(aggregated_df["Date"])

    # Set "Date" as the index
    aggregated_df.set_index("Date", inplace=True)

    # Check for Nan_Value
    print(aggregated_df.isna().sum())


    # ### Visualization trends

    # In[65]:


    ##-- ### Visualization trends ### --##

    # Set plot style
    sns.set(style="whitegrid")

    # Plot the time series
    plt.figure(figsize=(14, 5))
    plt.plot(aggregated_df.index, aggregated_df['Number of Insects'], marker='o', linestyle='-')
    plt.title('Daily Insect Counts Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # In[66]:


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


    # In[67]:


    # Plot ACF and PACF to examine autocorrelation
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(aggregated_df['Number of Insects'], lags=30, ax=ax[0])
    plot_pacf(aggregated_df['Number of Insects'], lags=25, ax=ax[1])
    plt.tight_layout()
    plt.show()


    # ### Check for Stationarity (ADF Test)

    # In[68]:


    # Perform Augmented Dickey-Fuller (ADF) test to check for stationarity
    def check_stationarity(series):
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
            print("⚠️ Series is not stationary (p >= 0.05)", "\n")

        # Plot rolling mean and std
        rolling_mean = series.rolling(window=7).mean()
        rolling_std = series.rolling(window=7).std()

        plt.figure(figsize=(14,6))
        plt.plot(series, label='Original')
        plt.plot(rolling_mean, label='Rolling Mean (7 days)', color='red')
        plt.plot(rolling_std, label='Rolling Std (7 days)', color='black')
        plt.legend()
        plt.title("Rolling Mean & Standard Deviation - Non-seasonal Check")
        plt.show()

        #return result

    check_stationarity(aggregated_df['Number of Insects'])


    # In[69]:


    from statsmodels.tsa.seasonal import seasonal_decompose

    insects_series = aggregated_df['Number of Insects']

    # Set seaborn style for better aesthetics
    sns.set(style="whitegrid")

    # Seasonal decomposition
    decomposition = seasonal_decompose(insects_series, model='additive', period=7)  # Weekly seasonality

    # Extract components
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot original series
    axs[0].plot(insects_series, label='Observed', color='black')
    axs[0].set_title('Observed', fontsize=13)
    axs[0].legend(loc='upper left')

    # Plot trend
    axs[1].plot(trend, label='Trend', color='blue')
    axs[1].set_title('Trend', fontsize=13)
    axs[1].legend(loc='upper left')

    # Plot seasonal
    axs[2].plot(seasonal, label='Seasonal', color='green')
    axs[2].set_title('Seasonal', fontsize=13)
    axs[2].legend(loc='upper left')

    # Plot residual
    axs[3].plot(residual, label='Residual', color='red')
    axs[3].set_title('Residual', fontsize=13)
    axs[3].legend(loc='upper left')

    # General layout adjustments
    plt.suptitle("Seasonal Decomposition of Number of Insects", fontsize=16, y=1.02)
    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


    # ### Split train and test set

    # In[70]:


    # Define target and exogenous features
    target = 'Number of Insects'

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

    # Drop NA values due to differencing (if needed)
    exogenous_data = aggregated_df[[target] + exog_features].dropna()

    # Optional: Check for NaNs in exogenous data
    print("\nMissing values in exogenous features:\n")
    print(exogenous_data.isnull().sum())


    # In[71]:


    # Ensure datetime index if not already
    exogenous_data.index = pd.to_datetime(exogenous_data.index)

    # Split into train and test (90/10 split)
    split_index = int(len(exogenous_data) * 0.9)
    train = exogenous_data.iloc[:split_index]
    test = exogenous_data.iloc[split_index:]

    # Define endogenous and exogenous variables
    y_train = train[target]
    X_train = train[exog_features]
    y_test = test[target]
    X_test = test[exog_features]


    # In[72]:


    # Optional: visualize the split
    plt.figure(figsize=(14, 6))
    plt.plot(y_train.index, y_train, label='Train')
    plt.plot(y_test.index, y_test, label='Test')
    plt.axvline(y_train.index[-1], color='black', linestyle='--', label='Train/Test Split')
    plt.title('Train-Test Split')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.show()


    # ### Fit the SARIMA model

    # In[73]:


    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Step 1: Fit SARIMA model
    model = auto_arima(
        y_train,
        exogenous=X_train,
        seasonal=True,
        m=7,
        d=None,
        D=1,
        start_p=0, max_p=3,   # Range of p values
        start_q=0, max_q=3,   # Range of q values
        start_P=0, max_P=2,   # Seasonal AR order
        start_Q=0, max_Q=2,   # Seasonal MA order
        stepwise=True,
        enforce_stationarity=False,
        enforce_invertibility=False,
        suppress_warnings=True,
        trace=True,           # Print the search process
        error_action='ignore'
    )

    # Summary of the best model
    print(model.summary())


    # ### Forcast by SARIMA model

    # In[74]:


    # Step 2: Forecast for the test period
    forecast = model.predict(n_periods=len(y_test), exogenous=X_test)
    forecast = pd.Series(forecast, index=y_test.index)


    # Step 3: Plot forecast vs actual for train and test
    plt.figure(figsize=(14, 6))
    plt.plot(y_train.index, y_train, label='Train')
    plt.plot(y_train.index, model.predict_in_sample(exogenous=X_train), label='Fitted', marker='x', linestyle='--', color='orange')
    plt.plot(y_test.index, y_test, label='Test')
    plt.plot(forecast.index, forecast, label='SARIMA Forecast', color='red', linestyle='--')
    plt.axvline(y_train.index[-1], color='black', linestyle='--', label='Train/Test Split')
    plt.title("SARIMA Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Number of Insects")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ### Apply differencing (if needed) and check Stationarity for non_seasonal data

    # In[75]:


    # Function
    def handle_nonseasonal_stationarity(series, max_diff=6, verbose=True):
        diff_series = series.copy()
        d = 0

        for i in range(max_diff + 1):
            result = adfuller(diff_series.dropna())
            p_value = result[1]
            if verbose:
                print(f"Non-seasonal differencing d={d}, ADF p-value: {p_value:.4f}")
            if p_value < 0.05:
                if verbose:
                    print(f"✅ Stationary after {d} non-seasonal difference(s)\n")
                return diff_series, d
            diff_series = diff_series.diff()
            d += 1

        print(f"⚠️ Series may not be stationary after {max_diff} non-seasonal differences.\n")
        return diff_series, d


    # ### Apply differencing (if needed) and check stationarity for seasonal data

    # In[76]:


    def handle_seasonal_stationarity(series, seasonal_period=7, max_diff=6, verbose=True):
        diff_series = series.copy()
        D = 0

        for i in range(max_diff + 1):
            result = adfuller(diff_series.dropna())
            p_value = result[1]
            if verbose:
                print(f"Seasonal differencing D={D}, ADF p-value: {p_value:.4f}")
            if p_value < 0.05:
                if verbose:
                    print(f"✅ Stationary after {D} seasonal difference(s)\n")
                return diff_series, D
            diff_series = diff_series.diff(seasonal_period)
            D += 1

        print(f"⚠️ Series may not be seasonally stationary after {max_diff} seasonal differences.\n")
        return diff_series, D


    # In[77]:


    series = aggregated_df['Number of Insects']

    # First handle seasonal stationarity (D)
    seasonally_diffed, D = handle_seasonal_stationarity(series, seasonal_period=7)

    # Then handle non-seasonal stationarity (d)
    final_stationary_series, d = handle_nonseasonal_stationarity(seasonally_diffed)

    print(f"Final differencing parameters: d = {d}, D = {D}")


    # ### Split train and test affter handle non-stationary problem

    # In[78]:


    # Define target and exogenous features
    target = 'Number of Insects'

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

    # Drop NA values due to differencing (if needed)
    exogenous_data = aggregated_df[[target] + exog_features].dropna()

    # Optional: Check for NaNs in exogenous data
    print("\nMissing values in exogenous features:\n")
    print(exogenous_data.isnull().sum())


    # In[79]:


    # Ensure datetime index if not already
    exogenous_data.index = pd.to_datetime(exogenous_data.index)

    # Split into train and test (90/10 split)
    split_index = int(len(exogenous_data) * 0.9)
    train = exogenous_data.iloc[:split_index]
    test = exogenous_data.iloc[split_index:]

    # Define endogenous and exogenous variables
    y_train = train[target]
    X_train = train[exog_features]
    y_test = test[target]
    X_test = test[exog_features]


    # In[80]:


    # Optional: visualize the split
    plt.figure(figsize=(14, 6))
    plt.plot(y_train.index, y_train, label='Train')
    plt.plot(y_test.index, y_test, label='Test')
    plt.axvline(y_train.index[-1], color='black', linestyle='--', label='Train/Test Split')
    plt.title('Train-Test Split')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.show()


    # ### Fit the SARIMAX model by the new parameters

    # In[81]:


    ##############################
    # Fit SARIMA model (using SARIMAX with no exogenous variables)
    sarimax_model = SARIMAX(
        y_train,
        exgogenous=X_train,
        order=(1, 0, 0),               # (p, d, q) — you can tune p & q manually or use prior auto_arima result
        seasonal_order=(2, 3, 0, 7),   # (P, D, Q, m) — m=7 for weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    sarimax_result = sarimax_model.fit(disp=False)
    print(sarimax_result.summary())


    # In[82]:


    # Forecast for the test period
    sarimax_forecast = sarimax_result.get_forecast(steps=len(y_test), exog=X_test)
    forecast_mean = sarimax_forecast.predicted_mean
    conf_int = sarimax_forecast.conf_int()

    # Plot forecast vs actual
    plt.figure(figsize=(14, 6))
    plt.plot(y_train.index, y_train, label='Train')
    plt.plot(y_train.index, sarimax_result.fittedvalues, label='Fitted', marker='x', linestyle='--', color='orange')
    plt.plot(y_test.index, y_test, label='Test')
    plt.plot(forecast_mean.index, forecast_mean, label='SARIMAX Forecast', color='red', linestyle='--')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')
    plt.axvline(y_train.index[-1], color='black', linestyle='--', label='Train/Test Split')
    plt.title("SARIMAX Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Number of Insects")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # In[83]:


    # Plot forecast vs actual
    plt.figure(figsize=(8, 6))
    plt.plot(y_test.index, y_test, label='Test', marker='o')
    plt.plot(forecast_mean.index, forecast_mean, label='SARIMA Forecast', color='red', linestyle='--', marker='x')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')
    plt.title("SARIMA Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Number of Insects")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # In[84]:


    mae = mean_absolute_error(y_test, forecast_mean)
    rmse = mean_squared_error(y_test, forecast_mean)
    r2 = r2_score(y_test, forecast_mean)

    print(f"\nForecasting Performance on Test Set:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {r2:.2f}")


    # ### Residual Plot (Error Over Time)

    # In[85]:


    # Get in-sample predictions for training data
    fitted_values = sarimax_result.fittedvalues
    in_sample_predictions = pd.Series(fitted_values, index=y_train.index)

    # Calculate residuals
    residuals = y_train - in_sample_predictions

    # Plot residuals
    plt.figure(figsize=(14, 5))
    plt.plot(residuals.index, residuals, label='Residuals', color='red', marker='o', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='--', label='Zero Line')
    plt.title("Residual Plot")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ### Histogram + Q-Q Plot of Residuals (Normality Check)

    # In[86]:


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
