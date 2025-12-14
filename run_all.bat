@echo off
echo ==========================================
echo Aircraft Engine Predictive Maintenance Run
echo ==========================================

echo [1/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/4] Running Exploratory Data Analysis...
python src/eda.py
if %errorlevel% neq 0 (
    echo Error running EDA.
    pause
    exit /b %errorlevel%
)

echo.
echo [3/4] Training Model (this may take a minute)...
python src/train.py
if %errorlevel% neq 0 (
    echo Error training model.
    pause
    exit /b %errorlevel%
)

echo.
echo [4/4] Evaluating Model...
python src/evaluate.py
if %errorlevel% neq 0 (
    echo Error evaluating model.
    pause
    exit /b %errorlevel%
)

echo.
echo ==========================================
echo All steps completed successfully!
echo Check the generated .png files and rul_model.pth
echo ==========================================
pause
