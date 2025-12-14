import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_data(train_df, test_df, sensors_to_use=None):
    """
    Normalize sensor data and prepare for model.
    """
    if sensors_to_use is None:
        # Default sensors that are useful for FD001 according to literature
        # Some sensors are constant and can be dropped
        sensors_to_use = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
    
    train_data = train_df.copy()
    test_data = test_df.copy()

    # Feature Engineering
    # Calculate rolling mean and std for sensors
    window_size = 5
    sensor_cols = sensors_to_use
    
    for col in sensor_cols:
        train_data[f'{col}_mean'] = train_data.groupby('unit_number')[col].rolling(window=window_size).mean().reset_index(0, drop=True)
        train_data[f'{col}_std'] = train_data.groupby('unit_number')[col].rolling(window=window_size).std().reset_index(0, drop=True)
        
        test_data[f'{col}_mean'] = test_data.groupby('unit_number')[col].rolling(window=window_size).mean().reset_index(0, drop=True)
        test_data[f'{col}_std'] = test_data.groupby('unit_number')[col].rolling(window=window_size).std().reset_index(0, drop=True)
        
    # Drop NaNs created by rolling window
    train_data.dropna(inplace=True)
    test_data.fillna(0, inplace=True) # Fill NaNs in test to keep lengths consistent or drop
    
    # Update sensors_to_use to include new features
    new_features = [f'{col}_mean' for col in sensor_cols] + [f'{col}_std' for col in sensor_cols]
    sensors_to_use = sensors_to_use + new_features
    
    scaler = MinMaxScaler()
    train_data[sensors_to_use] = scaler.fit_transform(train_data[sensors_to_use])
    test_data[sensors_to_use] = scaler.transform(test_data[sensors_to_use])
    
    return train_data, test_data, sensors_to_use

def gen_sequence(df, sequence_length, columns):
    """
    Generates sequences for LSTM/GRU.
    Returns: (samples, sequence_length, features)
    """
    data_array = df[columns].values
    num_units = df['unit_number'].unique()
    
    sequences = []
    labels = []
    
    for unit in num_units:
        unit_data = df[df['unit_number'] == unit]
        data_matrix = unit_data[columns].values
        rul_array = unit_data['RUL'].values if 'RUL' in unit_data.columns else None
        
        for i in range(len(data_matrix) - sequence_length + 1):
            sequences.append(data_matrix[i : i + sequence_length])
            if rul_array is not None:
                labels.append(rul_array[i + sequence_length - 1])
                
    return np.array(sequences), np.array(labels)

def gen_test_sequence(test_df, sequence_length, columns):
    """
    Generates only the last sequence for each unit in test set for evaluation.
    """
    sequences = []
    num_units = test_df['unit_number'].unique()
    
    for unit in num_units:
        unit_data = test_df[test_df['unit_number'] == unit]
        
        if len(unit_data) >= sequence_length:
            # Take the last 'sequence_length' rows
            seq = unit_data[columns].values[-sequence_length:]
            sequences.append(seq)
        else:
            # Padding could be added here, but for now we skip short sequences 
            # (Note: in CMAPSS, almost all test units are long enough)
            seq = unit_data[columns].values
            # Pad with zeros
            pad_len = sequence_length - len(seq)
            padded = np.pad(seq, ((pad_len, 0), (0, 0)), 'constant')
            sequences.append(padded)
            
    return np.array(sequences)
