# Potential Continuous Ranked Probability Score (PCRPS)

## Project Overview
This repository provides tools to compute and evaluate the **Potential Continuous Ranked Probability Score (PCRPS)** and **PCRPS Skill (PCRPS-S)** measures, as introduced in [Gneiting et al. (2026)](#paper) (“Probabilistic measures afford fair comparisons of AIWP and NWP model output”). By applying EasyUQ, deterministic forecast outputs are converted into calibrated probabilistic distributions, and the resulting PCRPS (mean CRPS of postprocessed forecasts) serves as a metric for comparing single‐valued forecasts across models.

Applications in this repo include:
- A simulation study (`simulation_study/`)
- WeatherBench 1 forecast datasets (`wb1/`)
- WeatherBench 2 forecast datasets (`wb2/`)

For detailed instructions, see each subdirectory’s README.

## Paper
**Gneiting et al. (2026).** Probabilistic measures afford fair comparisons of AIWP and NWP model output. 
Artificial Intelligence for the Earth Systems, 5(1), e250054. DOI: 10.1175/AIES-D-25-0054.1.

### Citation
```bibtex
@article{gneiting2026probabilistic,
  title={Probabilistic measures afford fair comparisons of AIWP and NWP model output},
  author={Gneiting, Tilmann and Biegert, Tobias and Kraus, Kristof and Walz, Eva-Maria and Jordan, Alexander I and Lerch, Sebastian},
  journal={Artificial Intelligence for the Earth Systems},
  volume={5},
  number={1},
  pages={e250054},
  year={2026},
  publisher={American Meteorological Society}
}
```
