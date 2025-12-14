from src.data_loader import load_data
from src.preprocessing import process_data, gen_sequence
import torch

print("Testing data loading...")
try:
    train_df, test_df, rul_df = load_data("FD001")
    print(f"Loaded: Train {train_df.shape}, Test {test_df.shape}")
    
    print("Testing preprocessing...")
    train_data, test_data, features = process_data(train_df, test_df)
    print(f"Processed features: {len(features)}")
    
    print("Testing sequence gen...")
    X, y = gen_sequence(train_data, 50, features)
    print(f"Sequences: {X.shape}")
    
    print("All good!")
except Exception as e:
    print(f"Error: {e}")
