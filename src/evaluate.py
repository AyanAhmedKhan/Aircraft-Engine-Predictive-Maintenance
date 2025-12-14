import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from data_loader import load_data
from preprocessing import process_data, gen_test_sequence
from models import RULModel

# Hyperparameters (must match training)
SEQUENCE_LENGTH = 50
HIDDEN_DIM = 100
NUM_LAYERS = 2

def evaluate():
    print("Loading data...")
    train_df, test_df, rul_df = load_data("FD001")
    
    print("Preprocessing...")
    # We need to process train_df just to fit the scaler (or use saved scaler, but here we re-fit for simplicity)
    # Ideally should save/load scaler
    _, test_data, features = process_data(train_df, test_df)
    
    print("Generating test sequences...")
    X_test = gen_test_sequence(test_data, SEQUENCE_LENGTH, features)
    y_true = rul_df['RUL'].values
    
    # Filter out units that were too short (if any)
    # In gen_test_sequence we padded, so len(X_test) should be equal to num_units
    # But let's verify
    if len(X_test) != len(y_true):
        print(f"Warning: Number of test sequences ({len(X_test)}) does not match number of RUL values ({len(y_true)}).")
        # This might happen if some units were shorter than sequence length and we didn't pad (but we did pad)
    
    # Convert to Tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Load Model
    input_dim = len(features)
    model = RULModel(input_dim, HIDDEN_DIM, NUM_LAYERS)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load("rul_model.pth"))
    model.to(device)
    model.eval()
    
    print("Predicting...")
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    # Clip predictions to be non-negative
    predictions = np.maximum(predictions, 0)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")

    # --- Classification Metrics ---
    # Define threshold for "Early Warning" (e.g. within 30 cycles)
    THRESHOLD = 30
    
    y_true_binary = (y_true <= THRESHOLD).astype(int)
    predictions_binary = (predictions <= THRESHOLD).astype(int)
    
    acc = accuracy_score(y_true_binary, predictions_binary)
    f1 = f1_score(y_true_binary, predictions_binary)
    prec = precision_score(y_true_binary, predictions_binary)
    rec = recall_score(y_true_binary, predictions_binary)
    cm = confusion_matrix(y_true_binary, predictions_binary)
    
    print("\nClassification Metrics (Threshold <= 30 cycles):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save metrics to text file
    with open("metrics.txt", "w") as f:
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
    print("Saved metrics.txt")
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='True RUL')
    plt.plot(predictions, label='Predicted RUL')
    plt.title(f'RUL Prediction on Test Set (RMSE={rmse:.2f})')
    plt.xlabel('Unit Index')
    plt.ylabel('RUL')
    plt.legend()
    plt.savefig('test_rul_prediction.png')
    print("Saved test_rul_prediction.png")
    
    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, predictions, alpha=0.5)
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.title('True vs Predicted RUL')
    plt.savefig('rul_scatter.png')
    print("Saved rul_scatter.png")

if __name__ == "__main__":
    if not __import__("os").path.exists("rul_model.pth"):
        print("Model file 'rul_model.pth' not found. Train the model first.")
    else:
        evaluate()
