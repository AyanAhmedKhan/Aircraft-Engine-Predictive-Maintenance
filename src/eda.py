import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_loader import load_data, add_rul

def run_eda():
    print("Loading data...")
    train_df, _, _ = load_data("FD001")
    train_df = add_rul(train_df)
    
    print("\nData Info:")
    print(train_df.info())
    
    print("\nData Head:")
    print(train_df.head())
    
    # Plot RUL for first unit
    plt.figure(figsize=(10, 6))
    unit_1 = train_df[train_df['unit_number'] == 1]
    plt.plot(unit_1['time_in_cycles'], unit_1['RUL'])
    plt.title('RUL of Unit 1')
    plt.xlabel('Time (Cycles)')
    plt.ylabel('RUL')
    plt.savefig('rul_unit_1.png')
    print("Saved rul_unit_1.png")
    
    # Correlation Heatmap
    plt.figure(figsize=(15, 10))
    # Drop constant columns or index columns for correlation
    cols_to_drop = ['unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'setting_3']
    # Also drop columns that are constant (check manually later or automating)
    # For now just plot all sensors
    sensor_cols = [c for c in train_df.columns if c.startswith('s_')]
    corr = train_df[sensor_cols + ['RUL']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    print("Saved correlation_matrix.png")

if __name__ == "__main__":
    if not os.path.exists("archive/train_FD001.txt"):
        print("Data not found. Please ensure data is downloaded.")
    else:
        run_eda()
