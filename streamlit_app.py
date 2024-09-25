# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit app title
st.title("India Unemployment Rate Analysis & Forecasting")

# File uploader
data = st.file_uploader("Upload a CSV file", type="csv")

# Data Preprocessing
if data is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(data)
    
    # Clean column names
    data.columns = data.columns.str.strip()
    
    # Convert 'Estimated Unemployment Rate (%)' to numeric, handling errors
    data['Estimated Unemployment Rate (%)'] = pd.to_numeric(data['Estimated Unemployment Rate (%)'], errors='coerce')
    
    # Drop rows with missing values
    data = data.dropna(subset=['Estimated Unemployment Rate (%)'])
    
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # Remove rows with invalid dates
    data = data.dropna(subset=['Date'])
    
    # Extract Month and Year for analysis
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    
    # Display the dataset in Streamlit
    st.subheader('Dataset')
    st.dataframe(data.head())
    
    # Visualize unemployment rate over time
    st.subheader('Unemployment Rate Over Time')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Date', y='Estimated Unemployment Rate (%)', marker='o', color='blue')
    plt.title('Unemployment Rate Over Time in India')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    
    # Grouping the unemployment rate by month
    monthly_data = data.groupby('Date').agg({'Estimated Unemployment Rate (%)': 'mean'})
    
    # Forecasting period selection
    st.subheader('Forecasting Options')
    forecast_period = st.slider('Select the number of months for forecasting:', min_value=1, max_value=12, value=6)
    
    # Split data into training and test sets (80% training, 20% testing)
    train_size = int(len(monthly_data) * 0.8)
    train, test = monthly_data.iloc[:train_size], monthly_data.iloc[train_size:]

    # Fit ARIMA model on training data
    model = ARIMA(train['Estimated Unemployment Rate (%)'], order=(1, 1, 1))
    model_fit = model.fit()
    
    # Forecast based on the user-selected period
    forecast = model_fit.forecast(steps=forecast_period)

    # Ensure test_subset is appropriately defined
    if len(test) < forecast_period:
        st.warning("Not enough data for the specified forecast period. Adjust the forecast period slider.")
        test_subset = test.copy()  # Keep actual test values for plotting
    else:
        test_subset = test.iloc[:forecast_period]  # Only take the forecast_period length from test

    # Calculate RMSE (Root Mean Squared Error) only if test_subset is valid
    if len(test_subset) == forecast_period:
        rmse = np.sqrt(mean_squared_error(test_subset['Estimated Unemployment Rate (%)'], forecast))
        st.write(f'RMSE for {forecast_period} months: {rmse:.2f}')

        # Calculate accuracy (MAPE)
        mape = np.mean(np.abs((test_subset['Estimated Unemployment Rate (%)'].values - forecast) / test_subset['Estimated Unemployment Rate (%)'].values)) * 100
        accuracy = 100 - mape
        st.write(f'Accuracy: {accuracy:.2f}%')

        # Combine training data and forecast for a smooth plot
        last_date = train.index[-1]  # Get the last date from training set
        # Create a date range for the forecast
        forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_period, freq='M')
        forecast_series = pd.Series(forecast, index=forecast_index)

        # Plotting historical data, forecast, and actual values
        plt.figure(figsize=(12, 6))
        # Plot training data
        plt.plot(train.index, train['Estimated Unemployment Rate (%)'], label='Training Data', color='blue')
        # Plot test data (actual values)
        plt.plot(test.index, test['Estimated Unemployment Rate (%)'], label='Actual Data', color='green')
        # Plot forecasted data
        plt.plot(forecast_series.index, forecast_series, label='Forecasted Data', color='red', linestyle='--')
        # Highlight the forecast starting point
        plt.axvline(x=train.index[-1], color='gray', linestyle='--', label='Forecast Start')
        # Add title, labels, and legend
        plt.title('Unemployment Rate Forecast vs Actual')
        plt.xlabel('Month')
        plt.ylabel('Unemployment Rate (%)')
        plt.legend()
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        # Display the plot in Streamlit
        st.pyplot(plt.gcf())
        
    else:
        st.write("Not enough data to calculate RMSE and Accuracy.")
else:
    st.write("Please upload a CSV file to analyze.")
