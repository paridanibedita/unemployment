##BUSINESS PROBLEM: 
'''Unemployment is measured by the unemployment rate which is the number of people
who are unemployed as a percentage of the total labour force. We have seen a sharp
increase in the unemployment rate during Covid-19, so analyzing the unemployment rate
can be a good data science project. '''

##BUSINESS OBJECTIVE : 
'''Accurately analyse the unemployment rate.'''

##BUSINESS CONSTRAINTS : 
'''Eassy interpretability'''

##SUCCESS CRITERIA: 
'''
1.Business success criteria:Minimize the cost of emplyment. 
2.Economic success criteria: Decreasing the unemployment by 20%.
3.ML success criteria : Achieve accuracy of 90%.'''

##load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.arima.model import ARIMA 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

##Load dataset
data = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Documents\Data science\project2\TASK2\Unemployment in India.csv")
data

##describe the data
data.describe()

##information about the dataset
data.info()

##columns and shape of the dataset
data.columns
data.shape

##check is there any null values
data.isnull().sum()

##remove null values
data = data.dropna()
data.isnull().sum()

## Remove spaces
data.columns = data.columns.str.strip() 

## Convert date column to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

## Drop rows with missing dates
data = data.dropna(subset=['Date'])

## Extract day, month, and year from the 'Date' column
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

## Show updated data
print(data.head())

##Visualize the unemployment rate over time
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Date', y='Estimated Unemployment Rate (%)', marker='o', color='blue')
plt.title('Unemployment Rate Over Time in India')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.xticks(rotation=45)
plt.show()

##Plot unemployment rate by region using a boxplot
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x='Region', y='Estimated Unemployment Rate (%)', palette='Set2')
plt.title('Unemployment Rate by Region')
plt.xticks(rotation=45)
plt.show()

##Distribution of the unemployment rate (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(data['Estimated Unemployment Rate (%)'], bins=15, kde=True, color='green')
plt.title('Distribution of Unemployment Rate')
plt.xlabel('Unemployment Rate')
plt.ylabel('Frequency')
plt.show()

##Scatter plot to explore relationships (e.g., Month vs Unemployment Rate)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Month', y='Estimated Unemployment Rate (%)', hue='Year', palette='coolwarm')
plt.title('Unemployment Rate Scatter Plot (Month vs Year)')
plt.xlabel('Month')
plt.ylabel('Unemployment Rate')
plt.show()

# Grouping the unemployment rate by month
monthly_data = data.groupby('Month').agg({'Estimated Unemployment Rate (%)': 'mean'})

# Split data into training and test sets (80% training, 20% testing)
train_size = int(len(monthly_data) * 0.8)
train, test = monthly_data.iloc[:train_size], monthly_data.iloc[train_size:]

# Fit ARIMA model on training data
model = ARIMA(train['Estimated Unemployment Rate (%)'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast for the test set (length of the test set)
forecast = model_fit.forecast(steps=len(test))

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(test['Estimated Unemployment Rate (%)'], forecast))
print(f'RMSE: {rmse}')

# Accuracy metric (Mean Absolute Percentage Error - MAPE)
mape = np.mean(np.abs((test['Estimated Unemployment Rate (%)'] - forecast) / test['Estimated Unemployment Rate (%)'])) * 100
accuracy = 100 - mape
print(f'Accuracy: {accuracy:.2f}%')

# Combine training data and forecast for a smooth plot
forecast_series = pd.Series(forecast, index=test.index)

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
# Display the plot
plt.show()













































