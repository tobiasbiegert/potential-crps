# Potential Continuous Ranked Probability Score (PC) & PC Skill (PCS)

## Project Overview
This project implements and evaluates the **Potential Continuous Ranked Probability Score (PC)** and **PC Skill (PCS)** measures for different applications.

## Repository Structure
- **simulation_study/**: Python script for running the simulation study &rarr; `run_experiments.py`
- **wb1/**
- **wb2/**: Applycation of PC/PCS to WeatherBench2 forecasts.
  - **pc/**: Python scripts and helper modules for computing PC &rarr; `compute_pc0.py`, `compute_pc.py`, `easyuq_helper.py`
  - **results/**: NetCDF outputs of computed PC metrics.
  - **plots/**: Generated figures.
  - **pc.ipynb**: Jupyter notebook for analysis and visualization.
  - **requirements.txt**: Pinned Python packages for local development.
  - **setup.py**: Installation requirements for running the evaluation on GCP with Apache Beam.
