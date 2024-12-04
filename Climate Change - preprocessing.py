#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


# # Walk through "data" folder, printing directories and files 
# # to use their names for file paths

# for root, dirs, files in os.walk("data"):
#     print("Current Directory:", root)
#     print()
#     print("Subdirectories:", dirs)
#     print()
#     print("Files:", files)


# # Preprocessing Functions

# In[3]:


# Convert 'object' columns to numeric
def convert_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# In[4]:


# Replace outliers with NaN
def replace_outliers_with_nan(df):
    for column in df.select_dtypes(include=[np.number]):  
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = df[column].apply(lambda x: np.nan if x < lower_bound or x > upper_bound else x)
    return df


# In[5]:


# Fill missing values using linear interpolation for specified columns
def fill_missing_values_with_interpolation(df, columns):
    for col in columns:
        df[col] = df[col].interpolate(method='linear')
    return df


# In[6]:


# Fill missing categorical columns with the most frequent value
def fill_missing_with_mode(df, columns):
    for col in columns:
        most_frequent_value = df[col].mode()[0]
        df[col] = most_frequent_value
    return df


# In[7]:


# Feature Engineering Function with Column Reordering
def feature_engineering(df):
    # Combine 'Year' and 'Month' into a single datetime column
    df['Date'] = pd.to_datetime(df['Year'].astype('Int64').astype(str) + '-' +
                                 df['Month'].astype('Int64').astype(str), errors='coerce')

    # Calculate temperature differences (Max Temp - Min Temp)
    df['Temp_Range (°C)'] = df['Max_Temp (°C)'] - df['Min_Temp (°C)']

    # Reorder columns as per the given list
    column_order = [
        'Date', 'Year', 'Month', 'Avg_Temp (°C)', 'Max_Temp (°C)', 'Min_Temp (°C)', 'Temp_Range (°C)',
        'Precipitation (mm)', 'Humidity (%)', 'Wind_Speed (m/s)', 'Solar_Irradiance (W/m²)', 'Cloud_Cover (%)',
        'CO2_Concentration (ppm)', 'Latitude', 'Longitude', 'Altitude (m)', 'Proximity_to_Water (km)',
        'Urbanization_Index', 'Vegetation_Index', 'ENSO_Index', 'Particulate_Matter (µg/m³)', 'Sea_Surface_Temp (°C)'
    ]
    
    df = df[column_order]

    return df


# # Load and Clean Data

# In[8]:


# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# In[9]:


# Clean data
def clean_data(df):

    # Convert object columns to numeric values
    df = convert_to_numeric(df)

    # Replace outliers with NaN
    df = replace_outliers_with_nan(df)

    # Replace 'Unknown' values with NaN
    df.replace('Unknown', np.nan, inplace=True)

    # Fill year and month values
    years = [year for year in range(2020, 2024) for _ in range(12)] + [2024] * 5
    months = list(range(1, 13)) * 4 + list(range(1, 6))
    df['Year'] = years[:len(df)]  
    df['Month'] = months[:len(df)]

    # Fill missing values using linear interpolation
    continuous_columns = [
        'Avg_Temp (°C)', 'Max_Temp (°C)', 'Min_Temp (°C)', 'Precipitation (mm)',
        'Humidity (%)', 'Wind_Speed (m/s)', 'Solar_Irradiance (W/m²)', 'Cloud_Cover (%)',
        'CO2_Concentration (ppm)', 'Urbanization_Index', 'Vegetation_Index', 'ENSO_Index',
        'Particulate_Matter (µg/m³)', 'Sea_Surface_Temp (°C)'
    ]
    df = fill_missing_values_with_interpolation(df, continuous_columns)

    # Fill missing latitude, longitude, altitude, and proximity columns with mode
    mode_columns = ['Latitude', 'Longitude', 'Altitude (m)', 'Proximity_to_Water (km)']
    df = fill_missing_with_mode(df, mode_columns)
    df = feature_engineering(df)

    return df


# # Basic Data Validation Checks

# In[10]:


# Check for duplicates
def check_duplicates(df):
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Warning: There are {duplicates} duplicate rows in the dataset.")
    else:
        print("No duplicates found.")

# Check for null values
def check_null_values(df):
    null_values = df.isnull().sum()
    if null_values.any():
        print("\nWarning: Some columns have missing values:")
        print(null_values[null_values > 0])
    else:
        print("\nNo missing values found.")

# Check if data types match expected ones
def check_data_types(df):
    expected_types = {
        'Date': 'datetime64[ns]',
        'Year': 'int64',
        'Month': 'int64',
        'Avg_Temp (°C)': 'float64',
        'Max_Temp (°C)': 'float64',
        'Min_Temp (°C)': 'float64',
        'Temp_Range (°C)': 'float64',
        'Precipitation (mm)': 'float64',
        'Humidity (%)': 'float64',
        'Wind_Speed (m/s)': 'float64',
        'Solar_Irradiance (W/m²)': 'float64',
        'Cloud_Cover (%)': 'float64',
        'CO2_Concentration (ppm)': 'float64',
        'Latitude': 'float64',
        'Longitude': 'float64',
        'Altitude (m)': 'float64',
        'Proximity_to_Water (km)': 'float64',
        'Urbanization_Index': 'float64',
        'Vegetation_Index': 'float64',
        'ENSO_Index': 'float64',
        'Particulate_Matter (µg/m³)': 'float64',
        'Sea_Surface_Temp (°C)': 'float64'
    }
    
    incorrect_types = []
    for column, expected_type in expected_types.items():
        if column in df.columns:
            actual_type = df[column].dtype
            if str(actual_type) != expected_type:
                incorrect_types.append((column, str(actual_type), expected_type))
    
    if incorrect_types:
        print("\nWarning: The following columns have incorrect data types:")
        for col, actual, expected in incorrect_types:
            print(f"  {col}: Actual: {actual}, Expected: {expected}")
    else:
        print("\nAll columns have the correct data types.")


# Check for outliers using IQR method
def check_outliers(df):
    outliers_found = False
    for column in df.select_dtypes(include=[np.number]):  # Only check numerical columns
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if not outliers.empty:
            outliers_found = True
            print(f"Warning: Outliers detected in column '{column}'")
    if not outliers_found:
        print("No outliers detected in any numerical columns.")


# Perform all checks
def perform_data_checks(df):
    check_duplicates(df)
    print()
    print('******************************')
    check_null_values(df)
    print()
    print('******************************')
    check_data_types(df)
    print()
    print('******************************')
    check_outliers(df)


# # Perform Data Loading, Cleaning, and Validation
# 

# In[11]:


main_dir = 'data/raw/'
file_name = 'climate_change_dataset.csv'
file_path = os.path.join(main_dir, file_name)


# In[12]:


# Load data
raw_data = load_data(file_path)


# ## Perform Data Checks for Raw Data

# In[13]:


perform_data_checks(raw_data)


# In[14]:


# print(raw_data)


# In[15]:


# Clean data
cleaned_data = clean_data(raw_data)


# In[16]:


# print(cleaned_data)


# ## Perform Data Checks for Processed Data

# In[17]:


perform_data_checks(cleaned_data)


# # Save the Processed Data

# In[18]:


# Directory path where files will be saved
save_dir = 'data/processed/'
cleaned_file_name = 'climate_cleaned.csv'
cleaned_file_path = os.path.join(save_dir, cleaned_file_name)


# In[19]:


# Save the cleaned data
cleaned_data.to_csv(cleaned_file_path, index=False)
print(f"File saved successfully at {cleaned_file_path}.")


# In[ ]:




