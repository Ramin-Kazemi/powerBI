def main():
    #!/usr/bin/env python
    # coding: utf-8

    # <a href="https://colab.research.google.com/github/raz0208/Agritech-Pest-Prediction/blob/main/Agritech_Pest_Prediction_ARIMA_TSA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

    # ## Time Series Analysis For Agritech Pest Prediction

    # ## Model Implementation:

    # ## **ARIMA**

    # In[30]:


    # # Uninstall potentially conflicting versions first
    # !pip uninstall -y pmdarima numpy

    # # Install the latest NumPy 1.x version (e.g., 1.26.4)
    # !pip install numpy==1.26.4

    # # Now install pmdarima (hopefully it links against NumPy 1.26.4)
    # # Use --no-cache-dir just to be safe
    # !pip install --no-cache-dir pmdarima==2.0.4


    # In[31]:


    # import required libaraies
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from pmdarima import auto_arima


    # In[32]:


    # Load datasets
    Merged_Dataset_df = pd.read_csv('/content/Final_Merged_Dataset_Cleaned.csv')


    # In[33]:


    # Display basic info for  datasets Final_Merged_Dataset_Cleaned
    print(Merged_Dataset_df.head(), '\n')
    print(Merged_Dataset_df.info())


    # In[34]:


    # Convert 'Date' column to datetime format
    Merged_Dataset_df['Date'] = pd.to_datetime(Merged_Dataset_df['Date'])

    # Set Date column as index
    Merged_Dataset_df.set_index('Date', inplace=True)


    # In[35]:


    # Drop redundant columns for time series modeling
    DateAgg_df = Merged_Dataset_df[['Number of Insects']].copy()

    # Resample data by date (sum over locations for the same day)
    DateAgg_df = DateAgg_df.resample('D').sum()

    # Show the processed daily data
    print(DateAgg_df.head(10))


    # ### Visualization trends

    # In[36]:


    ##-- ### Visualization trends ### --##

    # Set plot style
    sns.set(style="whitegrid")

    # Plot the time series
    plt.figure(figsize=(14, 5))
    plt.plot(DateAgg_df.index, DateAgg_df['Number of Insects'], marker='o', linestyle='-')
    plt.title('Daily Insect Counts Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # In[37]:


    # Plot ACF and PACF to examine autocorrelation
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(DateAgg_df['Number of Insects'], lags=30, ax=ax[0])
    plot_pacf(DateAgg_df['Number of Insects'], lags=25, ax=ax[1])
    plt.tight_layout()
    plt.show()


    # ### Check for Stationarity (ADF Test)

    # In[38]:


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

    check_stationarity(DateAgg_df['Number of Insects'])


    # ### Apply Differencing (if needed) and check Stationarity

    # In[39]:


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


    # In[40]:


    # # Plot differenced data
    # plt.figure(figsize=(14, 5))
    # plt.plot(df_diff, marker='o', linestyle='-')
    # plt.title(f'Differenced Insect Count (Differencing {num_diffs})')
    # plt.xlabel('Date')
    # plt.ylabel('Differenced Value')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()


    # In[41]:


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
    df_diff, num_diffs = make_stationary(DateAgg_df['Number of Insects'])


    # In[42]:


    # Plot differenced data
    plt.figure(figsize=(14, 5))
    plt.plot(df_diff, marker='o', linestyle='-')
    plt.title(f'Differenced Insect Count (Differencing {num_diffs})')
    plt.xlabel('Date')
    plt.ylabel('Differenced Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    # ### Fit ARIMA with auto_arima

    # In[43]:


    # Implement ARIMA model with autoarima
    model = auto_arima(DateAgg_df['Number of Insects'],
                       d=num_diffs,
                       seasonal=False,
                       stepwise=True,
                       trace=True,
                       suppress_warnings=True)

    print(model.summary())


    # ### Forecast Future Insect Counts

    # In[44]:


    # Predicting number of insects for a specific time period
    n_periods = 7
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    forecast_index = pd.date_range(start=DateAgg_df.index[-1] + pd.Timedelta(days=1), periods=n_periods, freq='D')

    # Convert to DataFrame
    forecast_df = pd.DataFrame({
        'Forecast': forecast,
        'Lower_CI': conf_int[:, 0],
        'Upper_CI': conf_int[:, 1]
    }, index=forecast_index)

    # Plot
    plt.figure(figsize=(14, 5))
    plt.plot(DateAgg_df.index, DateAgg_df['Number of Insects'], label='Historical', marker='o')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green', marker='o')
    plt.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='lightgreen', alpha=0.4)
    plt.title('Forecast of Insect Counts (Next 14 Days)')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # ## ARIMA model performance
    # #### Train vs. Fitted Plot
    # This shows how closely the model’s fitted values match the actual data during training.

    # In[45]:


    # Get in-sample predictions
    fitted_values = pd.Series(model.predict_in_sample(), index=DateAgg_df.index)

    # Plot actual vs fitted
    plt.figure(figsize=(14, 5))
    plt.plot(DateAgg_df.index, DateAgg_df['Number of Insects'], label='Actual', marker='o')
    plt.plot(fitted_values.index, fitted_values, label='Fitted', color='orange', linestyle='--', marker='x')
    plt.title('Actual vs Fitted Insect Counts (Training Performance)')
    plt.xlabel('Date')
    plt.ylabel('Number of Insects')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # #### Residual Plot (Error Over Time)
    # This shows the difference between actual and predicted values. Ideally, residuals should look like white noise (no pattern).

    # In[46]:


    # Calculate residuals
    residuals = DateAgg_df['Number of Insects'] - fitted_values

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


    # #### Histogram + Q-Q Plot of Residuals (Check Normality)
    # This is to check if residuals are normally distributed — a good sign for model assumptions.

    # In[47]:


    import scipy.stats as stats

    # Histogram of residuals
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title('Histogram of Residuals')
    plt.tight_layout()
    plt.show()

    print("\n")

    # Q-Q Plot
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals.dropna(), dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.tight_layout()
    plt.show()


    # ### Calculate Error Metrics

    # In[48]:


    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Actual and predicted values
    actual = DateAgg_df['Number of Insects']
    predicted = pd.Series(model.predict_in_sample(), index=DateAgg_df.index)

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



if __name__ == '__main__':
    main()
