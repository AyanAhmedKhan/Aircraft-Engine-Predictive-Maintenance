import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from data_loader import load_data, add_rul
from preprocessing import process_data, gen_sequence
from models import RULModel

# Hyperparameters
SEQUENCE_LENGTH = 50
BATCH_SIZE = 64
EPOCHS = 10 
LEARNING_RATE = 0.001
HIDDEN_DIM = 100
NUM_LAYERS = 2

def train(dataset_name="FD001"):
    print(f"--- Training on {dataset_name} ---")
    print("Loading data...")
    try:
        train_df, test_df, rul_df = load_data(dataset_name)
    except Exception as e:
        print(f"Error loading data for {dataset_name}: {e}")
        return

    print("Adding RUL...")
    train_df = add_rul(train_df)
    # Clip RUL at 125
    train_df['RUL'] = train_df['RUL'].clip(upper=125)
    
    print("Preprocessing...")
    train_data, test_data, features = process_data(train_df, test_df)
    print(f"Using {len(features)} features.")
    
    print("Generating sequences...")
    X_train, y_train = gen_sequence(train_data, SEQUENCE_LENGTH, features)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model Setup
    input_dim = len(features)
    model = RULModel(input_dim, HIDDEN_DIM, NUM_LAYERS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")
        
    # Save model
    save_path = f"rul_model_{dataset_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RUL model")
    parser.add_argument('--dataset', type=str, default='all', 
                        help='Dataset to train on: FD001, FD002, FD003, FD004 or all')
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = ['FD001', 'FD002', 'FD003', 'FD004']
        for ds in datasets:
            train(ds)
    elif args.dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
        train(args.dataset)
    else:
        print("Invalid dataset. Choose from FD001...FD004 or all")
