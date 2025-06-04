# Potential Continuous Ranked Probability Score (PC)

## Project Overview
This repository provides tools to compute and evaluate the **Potential Continuous Ranked Probability Score (PC)** and **PC Skill (PCS)** measures, as introduced in Gneiting et al. (2025) (“Probabilistic measures afford fair comparisons of AIWP and NWP model output”). By applying EasyUQ, deterministic forecast outputs are converted into calibrated probabilistic distributions, and the resulting PC (mean CRPS of postprocessed forecasts) serves as a metric for comparing single‐valued forecasts across models.

Applications in this repo include:
- A simulation study (`simulation_study/`)
- WeatherBench 1 forecast datasets (`wb1/`)
- WeatherBench 2 forecast datasets (`wb2/`)

For detailed instructions, see each subdirectory’s README.
