# Aircraft Engine Predictive Maintenance

This project implements a Deep Learning based approach to predict the Remaining Useful Life (RUL) of aircraft engines using the NASA CMAPSS dataset.

## Project Structure
- `src/data_loader.py`: Loads and parses the CMAPSS dataset.
- `src/preprocessing.py`: Feature engineering (Rolling Mean/Std) and normalization.
- `src/models.py`: PyTorch LSTM model definition.
- `src/train.py`: Training script.
- `src/evaluate.py`: Evaluation script (RMSE calculation and visualization).
- `src/eda.py`: Exploratory Data Analysis script.
- `requirements.txt`: Project dependencies.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. The dataset files should be in the `archive/` directory.

## Usage
### 1. Exploratory Data Analysis
Generate plots to understand the data:
```bash
python src/eda.py
```
Outputs: `rul_unit_1.png`, `correlation_matrix.png`

### 2. Train Model
Train the LSTM model:
```bash
python src/train.py
```
This will save the trained model to `rul_model.pth`.

### 3. Evaluate Model
Evaluate the model on the test set:
```bash
python src/evaluate.py
```
Outputs: 
- `metrics.txt`: Contains RMSE and Classification Metrics (Accuracy, F1, Precision, Recall).
- `test_rul_prediction.png`
- `rul_scatter.png`

## Metrics Explained
- **RMSE**: Root Mean Squared Error. (Lower is better). Measures how far off the predicted RUL is from the actual RUL on average.
- **Classification Metrics** (Threshold <= 30 cycles):
    - **Accuracy**: Percentage of correct "Fail soon" vs "Safe" predictions.
    - **F1 Score**: Harmonic mean of Precision and Recall. Good for imbalanced data.
    - **Precision**: When model predicts failure, how often is it correct?
    - **Recall**: Out of actual failures, how many did the model catch?

## Methodology
- **Data**: NASA CMAPSS FD001.
- **Features**: 14 sensors + Rolling Mean/Std (window=5).
- **Model**: 2-Layer LSTM with 100 hidden units.
- **Objective**: Predict RUL (Remaining Useful Life).
