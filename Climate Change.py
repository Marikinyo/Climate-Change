#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt


# In[2]:


# # Walk through "data" folder, printing directories and files 
# # to use their names for file paths
# for root, dirs, files in os.walk("data"):
#     print("Current Directory:", root)
#     print("Subdirectories:", dirs)
#     print("Files:", files)


# # Load Data

# In[3]:


main_dir = 'data/processed/'
file_name = 'climate_cleaned.csv'
file_path = os.path.join(main_dir, file_name)


# In[4]:


df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
# print(df.head())


# In[5]:


# Basic data checks
print(f"Index data type: {df.index.dtype}")
print(f"Missing values in index: {df.index.isnull().sum()}")
df.dropna(how='any', inplace=True, ignore_index=False)
print(f"Number of duplicate rows: {df.duplicated().sum()}")


# # Correlation analysis

# In[6]:


# Plot heatmap of correlations
def plot_correlation_heatmap(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()


# In[7]:


plot_correlation_heatmap(df)


# In[8]:


# Plot specific correlation between Avg_Temp and other variables
def plot_avg_temp_correlation(df):
    correlation_matrix = df.corr()
    correlation_avg_temp = correlation_matrix[['Avg_Temp (°C)']].sort_values(by='Avg_Temp (°C)', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_avg_temp, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between Avg_Temp (°C) and Other Variables (Ordered)')
    plt.show()
    


# In[9]:


# Select relevant features for correlation analysis
df_tmp = df[['Year', 'Month', 'Avg_Temp (°C)', 'Precipitation (mm)', 'Humidity (%)', 'Wind_Speed (m/s)', 
             'Solar_Irradiance (W/m²)', 'Cloud_Cover (%)', 'CO2_Concentration (ppm)', 
             'Urbanization_Index', 'Vegetation_Index', 'ENSO_Index', 'Particulate_Matter (µg/m³)', 
             'Sea_Surface_Temp (°C)']]    

plot_avg_temp_correlation(df_tmp)


# In[10]:


# Re-select relevant features for correlation analysis
df_tmp = df_tmp[['Year', 'Month', 'Avg_Temp (°C)', 'Precipitation (mm)', 'Wind_Speed (m/s)', 'Solar_Irradiance (W/m²)', 
                 'Cloud_Cover (%)', 'CO2_Concentration (ppm)', 'Urbanization_Index', 'Vegetation_Index', 
                 'Sea_Surface_Temp (°C)']]

plot_avg_temp_correlation(df_tmp)


# # Machine Learning Models

# In[11]:


# Features and target variable
X = df_tmp.drop(columns=['Avg_Temp (°C)'])
y = df_tmp['Avg_Temp (°C)']


# In[12]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Models for evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42)
}


# In[14]:


# Dictionary for storing evaluation metrics
results = {
    "Model": [],
    "MAE": [],
    "MSE": [],
    "RMSE": [],
    "R2 Score": []
}


# In[15]:


# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results["Model"].append(model_name)
    results["MAE"].append(mae)
    results["MSE"].append(mse)
    results["RMSE"].append(rmse)
    results["R2 Score"].append(r2)


# In[16]:


results_df = pd.DataFrame(results)
print(results_df)


# In[17]:


# GridSearch for hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Initialize RandomForest model and GridSearchCV
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get best parameters and cross-validation score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation score: {best_score}")


# In[18]:


# Retrain model with best parameters and evaluate
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)


# In[19]:


# Evaluation metrics for the best Random Forest model after GridSearch
mae_best_rf = mean_absolute_error(y_test, y_pred_best_rf)
mse_best_rf = mean_squared_error(y_test, y_pred_best_rf)
rmse_best_rf = np.sqrt(mse_best_rf)
r2_best_rf = r2_score(y_test, y_pred_best_rf)


# In[20]:


# Update the results_df for the 'Random Forest Regressor'
results_df.loc[results_df['Model'] == 'Random Forest Regressor', 'MAE'] = mae_best_rf
results_df.loc[results_df['Model'] == 'Random Forest Regressor', 'MSE'] = mse_best_rf
results_df.loc[results_df['Model'] == 'Random Forest Regressor', 'RMSE'] = rmse_best_rf
results_df.loc[results_df['Model'] == 'Random Forest Regressor', 'R2 Score'] = r2_best_rf

# Updated results
print(results_df)


# In[21]:


# Print final evaluation metrics
print(f"Test MSE: {mean_squared_error(y_test, y_pred)}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")


# # Time Series

# In[22]:


# ADF test to check stationarity
result = adfuller(df_tmp['Avg_Temp (°C)'])
print(f"ADF Statistic: {result[0]:.2f}")
print(f"p-value: {result[1]:.2e}")


# Since p-value is NOT above 0.05, the data is stationary and does NOT require differencing

# ## Train-Test Split:

# In[23]:


# Split data into train and test
train_size = int(len(df_tmp) * 0.8)
train, test = df_tmp[:train_size], df_tmp[train_size:]


# In[24]:


# Visualize the train and test split
plt.figure(figsize=(10,6))
train['Avg_Temp (°C)'].plot(label='Train')
test['Avg_Temp (°C)'].plot(label='Test')
plt.legend()
plt.title("Train-Test Split of Temperature Data")
plt.show()


# ## Fit the ARIMA Model

# In[25]:


# Fit the ARIMA model to the training data
model = auto_arima(train['Avg_Temp (°C)'], seasonal=False, trace=True)

# Model summary
print(model.summary())


# ## Forecasting

# In[26]:


# Forecast the future values for the length of the test data
forecast = model.predict(n_periods=len(test))


# In[27]:


# # Plot the actual vs forecasted values
# plt.figure(figsize=(10,6))
# plt.plot(test.index, test['Avg_Temp (°C)'], label='Actual', color='blue')
# plt.plot(test.index, forecast, label='Forecast', linestyle='--', color='red')

# plt.legend()
# plt.title("Actual vs Forecasted Temperature")
# plt.xlabel('Date')
# plt.ylabel('Temperature (°C)')
# plt.xticks(rotation=45) 
# plt.show()


# ## Arima Model Evaluation

# In[28]:


# ARIMA metrics
arima_mae = mean_absolute_error(test['Avg_Temp (°C)'], forecast)
arima_mse = mean_squared_error(test['Avg_Temp (°C)'], forecast)
arima_rmse = np.sqrt(arima_mse)
arima_r2 = r2_score(test['Avg_Temp (°C)'], forecast)

# Dictionary for ARIMA evaluation metrics
arima_results = pd.DataFrame({
    'Model': ['ARIMA'],
    'MAE': [arima_mae],
    'MSE': [arima_mse],
    'RMSE': [arima_rmse],
    'R2 Score': [arima_r2]
})

print(arima_results)


# # All models

# In[29]:


# Add the ARIMA results to the Machine Learning models results
results_df = pd.concat([results_df, arima_results], ignore_index=True)

# Final results
print(results_df)


# # Conclusion

# - **Linear Regression:** It has the highest error values (MAE, MSE, RMSE) and the lowest R² score, indicating poor performance.

# - **Random Forest Regressor:** This model has lower error values (MAE, MSE, RMSE) and a better R² score then *Linear Regression*, though the R² score is still negative.

# - **Gradient Boosting Regressor:** It has lower errors than Linear Regression, but the R² score is still negative

# - **XGBoost Regressor:** This model performs the best in terms of MSE and RMSE, with the lowest errors, but the R² score is still negative.

# - **ARIMA:** This model has the lowest MAE, but its MSE and RMSE are higher compared to XGBoost, and the R² score is also quite low, so it’s not the best model overall.

# In[ ]:




