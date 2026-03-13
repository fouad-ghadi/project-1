import pandas as pd
import numpy as np
# Assuming your comrades put their functions in src/data_processing.py
from src.data_processing import handle_missing_values, optimize_memory

def test_missing_values_handling():
    # 1. Create fake data with missing values
    df = pd.DataFrame({'age': [50, np.nan, 60], 'ejection_fraction': [30, 40, np.nan]})
    
    # 2. Run your comrade's cleaning function
    cleaned_df = handle_missing_values(df)
    
    # 3. Verify missing values handling 
    assert cleaned_df.isnull().sum().sum() == 0, "There are still missing values!"

def test_optimize_memory_function():
    # 1. Create fake data using 64-bit floats
    df = pd.DataFrame({'serum_creatinine': [1.1, 1.9, 2.0]}, dtype='float64')
    
    # 2. Run the optimization function
    optimized_df = optimize_memory(df)
    
    # 3. Verify optimize_memory(df) function  reduced it to 32-bit
    assert optimized_df['serum_creatinine'].dtype == 'float32', "Memory was not optimized to float32!"