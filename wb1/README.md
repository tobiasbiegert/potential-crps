# Weather Benchmark (WB) Analysis

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
python compute_metrics.py [--base_dir BASE_DIR] [--output_dir OUTPUT_DIR]
```
This will calculate various metrics (RMSE, CPA, PC, PCS, ACC) for each model and save the results in the `./metrics/` directory by default.

Optional arguments:
- `--base_dir`: Directory containing weather data (default: ./wb_data)
- `--output_dir`: Directory to save computed metrics (default: ./metrics)

4. Generate visualizations by running:
```bash
python visualization.py [--metrics_dir METRICS_DIR] [--output_dir OUTPUT_DIR]
```
This will create a PDF figure showing metrics for different forecasts.

Optional arguments:
- `--metrics_dir`: Directory containing metric files (default: ./metrics)
- `--output_dir`: Directory to save plots (default: metrics_dir/plots/)

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
- Centered Pattern Accuracy (CPA)
- Pattern Correlation (PC)
- Pattern Correlation Skill score (PCS)
- Anomaly Correlation Coefficient (ACC) 