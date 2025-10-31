@echo off
REM run_training.bat - Windows training script
REM Run this from the project root directory

setlocal enabledelayedexpansion

echo ==========================================
echo AI Bottleneck Detection - Model Training
echo ==========================================
echo.

REM Check if we're in the right directory
if not exist "pipelines\main.py" (
    echo ERROR: Must run from project root directory
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo [OK] Running from project root
echo.

REM Check for required directories
echo Checking directories...
if not exist "out" mkdir out
if not exist "models" mkdir models
echo [OK] Directories ready
echo.

REM Check for cleaned data files
set LOGISTICS_DATA=out\cleaned_logistics.csv
set MANUFACTURING_DATA=out\cleaned_manufacturing.csv

if not exist "%LOGISTICS_DATA%" (
    echo WARNING: %LOGISTICS_DATA% not found
    echo Checking for raw data...
    if exist "data\logistics.csv" (
        echo Using data\logistics.csv instead
        set LOGISTICS_DATA=data\logistics.csv
    ) else (
        echo ERROR: No logistics data found!
        pause
        exit /b 1
    )
)

if not exist "%MANUFACTURING_DATA%" (
    echo WARNING: %MANUFACTURING_DATA% not found
    echo Checking for raw data...
    if exist "data\manufacturing.csv" (
        echo Using data\manufacturing.csv instead
        set MANUFACTURING_DATA=data\manufacturing.csv
    ) else (
        echo ERROR: No manufacturing data found!
        pause
        exit /b 1
    )
)

echo [OK] Data files located
echo.

REM Train logistics model
echo ==========================================
echo Training Logistics Models
echo ==========================================
python models\train_logistics_eta.py --train_csv "%LOGISTICS_DATA%" --models_dir models --pred_csv "%LOGISTICS_DATA%" --pred_out out\logistics_preds.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAILED] Logistics training failed
    pause
    exit /b 1
)

echo.
echo [OK] Logistics models trained successfully
echo.

REM Train manufacturing model
echo ==========================================
echo Training Manufacturing Models
echo ==========================================
python models\train_manufacturing_speed_and_error.py --train_csv "%MANUFACTURING_DATA%" --models_dir models --pred_csv "%MANUFACTURING_DATA%" --pred_out out\mf_preds.csv

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAILED] Manufacturing training failed
    pause
    exit /b 1
)

echo.
echo [OK] Manufacturing models trained successfully
echo.

REM Summary
echo ==========================================
echo Training Complete!
echo ==========================================
echo.
echo Generated Models:
if exist "models\*.joblib" (
    dir /b models\*.joblib
) else (
    echo No .joblib models found
)
echo.
echo Model Cards:
if exist "models\*_modelcard.json" (
    dir /b models\*_modelcard.json
) else (
    echo No model cards found
)
echo.
echo Predictions:
if exist "out\*_preds.csv" (
    dir /b out\*_preds.csv
) else (
    echo No prediction files found
)
echo.
echo Reports:
if exist "out\*_report.html" (
    dir /b out\*_report.html
) else (
    echo No HTML reports found
)
echo.
echo ==========================================
echo All training completed successfully!
echo ==========================================
echo.
pause