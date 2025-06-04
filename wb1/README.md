# WeatherBench1 (WB1) Analysis

This repository contains code for analyzing and comparing different weather forecast models using various meteorological metrics.

## Setup and Usage

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Download the required data by running:
```bash
./download_wb1.sh
```
This will download the necessary weather data files into the `./wb_data/` directory.

3. Compute the metrics for all models by running:
```bash
python compute_metrics.py
```
This will calculate various metrics (RMSE, CPA, PC, PCS, ACC) for each model and save the results in the `./metrics/` directory.

4. Generate visualizations by running:
```bash
python visualization.py
```
This will create a PDF figure showing metrics for different forecasts

## Directory Structure

- `./wb_data/`: Contains the downloaded weather data files
- `./metrics/`: Contains the computed metrics for each model
- `./download_wb1.sh`: Script to download required data
- `./compute_metrics.py`: Script to compute verification metrics
- `./visualization.py`: Script to generate visualization plots

## Models Analyzed

The analysis includes the following forecast models:
- CNN
- Persistence
- Linear Regression
- T42
- T63

## Metrics Computed

The following metrics are computed for each model:
- Root Mean Square Error (RMSE)
- Coefficient of Predictive Ability (CPA)
- Potential Continuous Ranked Probability Score (PC)
- Potential Continuous Ranked Probability Skill Score (PCS)
- Anomaly Correlation Coefficient (ACC) 
