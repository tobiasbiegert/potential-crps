# Potential Continuous Ranked Probability Score (PC) & PC Skill (PCS)

## Project Overview
This project implements and evaluates the **Potential Continuous Ranked Probability Score (PC)** and **PC Skill (PCS)** measures for different applications.

## Repository Structure
- **simulation_study/**: Python script for running the simulation study &rarr; `run_experiments.py`
- **wb1/**
- **wb2/**: Applycation of PC/PCS to WeatherBench2 forecasts
  - **pc/**: Python scripts and helper modules for computing PC &rarr; `compute_pc0.py`, `compute_pc.py`, `easyuq_helper.py`
  - **plots/**: Generated figures
  - `extract_results.py` &rarr; script for downloading CRPS results from GCP bucket and storing PC, PCS, and time measurements on disk
  - `create_plots.py` &rarr; script for visualizations
  - `test_pc.py` &rarr; script for testing
  - **requirements.txt**: Pinned Python packages for local development
  - **setup.py**: Installation requirements for running the evaluation on GCP with Apache Beam
