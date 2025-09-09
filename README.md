# Potential Continuous Ranked Probability Score (PCRPS)

## Project Overview
This repository provides tools to compute and evaluate the **Potential Continuous Ranked Probability Score (PCRPS)** and **PCRPS Skill (PCRPS-S)** measures, as introduced in [Gneiting et al. (2025)](#paper) (“Probabilistic measures afford fair comparisons of AIWP and NWP model output”). By applying EasyUQ, deterministic forecast outputs are converted into calibrated probabilistic distributions, and the resulting PCRPS (mean CRPS of postprocessed forecasts) serves as a metric for comparing single‐valued forecasts across models.

Applications in this repo include:
- A simulation study (`simulation_study/`)
- WeatherBench 1 forecast datasets (`wb1/`)
- WeatherBench 2 forecast datasets (`wb2/`)

For detailed instructions, see each subdirectory’s README.

## Paper
**Gneiting et al. (2025).** Probabilistic measures afford fair comparisons of AIWP and NWP model output. Preprint available on [arXiv:2506.03744](https://arxiv.org/abs/2506.03744).

### Citation
```bibtex
@misc{gneiting2025probabilisticmeasuresaffordfair,
      title={Probabilistic measures afford fair comparisons of AIWP and NWP model output}, 
      author={Tilmann Gneiting and Tobias Biegert and Kristof Kraus and Eva-Maria Walz and Alexander I. Jordan and Sebastian Lerch},
      year={2025},
      eprint={2506.03744},
      archivePrefix={arXiv},
      primaryClass={stat.AP},
      url={https://arxiv.org/abs/2506.03744}, 
}
```
