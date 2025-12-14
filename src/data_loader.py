import pandas as pd
import numpy as np
import os

DATA_PATH = "archive"

def load_data(dataset="FD001"):
    """
    Loads the CMAPSS dataset.
    Args:
        dataset (str): One of "FD001", "FD002", "FD003", "FD004".
    Returns:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
        rul_df (pd.DataFrame): True RUL for test data
    """
    if dataset not in ["FD001", "FD002", "FD003", "FD004"]:
        raise ValueError("Invalid dataset name. Must be one of FD001, FD002, FD003, FD004")
    
    train_file = os.path.join(DATA_PATH, f"train_{dataset}.txt")
    test_file = os.path.join(DATA_PATH, f"test_{dataset}.txt")
    rul_file = os.path.join(DATA_PATH, f"RUL_{dataset}.txt")
    
    # Define column names
    index_names = ['unit_number', 'time_in_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)] 
    col_names = index_names + setting_names + sensor_names
    
    # Load data
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=col_names)
    test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=col_names)
    rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
    
    return train_df, test_df, rul_df

def add_rul(df):
    """
    Adds RUL column to training dataframe.
    RUL = Max Cycle - Current Cycle
    """
    # Get the max cycle for each unit
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df = df.drop(columns=['max_cycle'])
    return df
