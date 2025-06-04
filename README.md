# Potential Continuous Ranked Probability Score (PC) & PC Skill (PCS)

## Project Overview
This repository provides tools to compute and evaluate the **Potential Continuous Ranked Probability Score (PC)** and **PC Skill (PCS)** measures, as introduced in Gneiting et al. (2025) (“Probabilistic measures afford fair comparisons of AIWP and NWP model output”). By applying EasyUQ, deterministic forecast outputs are converted into calibrated probabilistic distributions, and the resulting PC (mean CRPS of postprocessed forecasts) serves as a metric for comparing single‐valued forecasts across models.

Applications in this repo include:
- A simulation study to illustrate PC/PCS properties (`simulation_study/`)
- WeatherBench 1 analyses (`wb1/`)
- WeatherBench 2 analyses (`wb2/`)

For detailed instructions, see each subdirectory’s README.
