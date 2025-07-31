# GAMM (Graph Attribute Missing Mechanisms)

A framework for generating missing data masks, evaluating, and comparing graph imputation methods based on our paper, "Drop the mask! GAMM - A Taxonomy for Graph Attributes Missing Mechanisms."

## Overview

This repository provides the official implementation of the GAMM framework. It introduces a comprehensive taxonomy for missingness mechanisms in attributed graphs and includes a robust experimental protocol for testing and comparing various imputation methods. The framework is designed for extensibility and reproducibility, supporting a wide range of popular graph datasets and state-of-the-art imputation algorithms.

The core components of this repository are:
- `GAMM_RUN.py`: The main command-line script for running experiments.
- `GAMM_UTILS.py`: Contains utility functions, including dataset loaders, mask generators, and imputation method implementations.
- `GAMM.ipynb`: A Jupyter Notebook for interactive exploration and running smaller-scale experiments.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
# Cloning may not work directly with anonymous.4open.science; a manual download may be required while the repository is anonymized.
git clone https://anonymous.4open.science/r/GAMM-B420

cd GAMM

pip install -r requirements.txt
```

## Usage

You can run experiments using either the command-line script (recommended for larger experiments) or the Jupyter Notebook (ideal for exploration).

### Command-Line Interface (CLI)

The primary way to run experiments is through the `GAMM_RUN.py` script.

#### Basic Example

Here is an example of how to run an experiment on the `Cora` and `CiteSeer` datasets with the `FP` and `GRIOT` imputers, using missing rates of 20% and 50%, repeated over 5 runs:

```bash
python3 GAMM_RUN.py \
    --datasets Cora,CiteSeer \
    --models FP,GRIOT \
    --missing-rates 0.2,0.5 \
    --runs 5
```

#### Full Options

For a complete list of all available command-line arguments, use the `--help` flag:

```bash
python3 GAMM_RUN.py --help
```

| Argument | Shorthand | Description | Default |
| :--- | :--- | :--- | :--- |
| `--datasets` | `-d` | Comma-separated list of datasets to use. | `Texas` |
| `--models` | `-m` | Comma-separated list of imputation models. If not set, all models are used. | `None` |
| `--missing-rates` | `--mr` | Comma-separated list of missing rates (float values between 0 and 1). | `0.2,0.5,0.8` |
| `--runs` | `-r` | Number of times to repeat the experiment for statistical significance. | `3` |
| `--output-path` | `-o` | Path to save experiment results. | `./GAMM` |
| `--start` | `-s` | Starting index for the experiment run number (e.g., for resuming). | `0` |
| `--save-output` | | If set, saves the imputed feature tensors and masks for each run. | `False` |
| `--keep-main-component`| | If set, only keeps the main connected component of each graph. | `True` |
| `--verbose` | | If set, prints detailed debug and environment information. | `True` |

---

### Available Datasets and Models

#### Datasets

| Category | Datasets |
| :--- | :--- |
| **Homophilic** | `Cora`, `CiteSeer`, `PubMed` |
| **Neutral** | `Actor`, `Chameleon`, `Squirrel` |
| **Heterophilic** | `Cornell`, `Texas`, `Wisconsin`, `Minesweeper`, `Roman-empire`, `Tolokers`, `Questions` |

#### Imputation Models

| Type | Models |
| :--- | :--- |
| **Baseline** | `Tabular_Avg`, `Random` |
| **Graph-Aware** | `Graph_1hop`, `FP` (Feature Propagation), `PCFI`, `GRIOT` |
| **Advanced Tabular** | `MICE`, `OT-tab` (Optimal Transport), `OT-tab-RR` |

---

### Jupyter Notebook

For interactive use, you can run experiments directly within the `GAMM.ipynb` notebook. The setup is similar to the CLI, allowing you to configure and run experiments cell by cell. This is useful for debugging, visualization, and smaller test runs.

## Output Structure

The results of the experiments are saved in the directory specified by the `--output-path` argument.

- **Aggregated Results**: A single JSON file named `aggregated_results_...json` is created at the end of all runs, containing a summary of performance metrics (MAE, RMSE, Accuracy, F1-score, etc.) for all configurations.
- **Per-Run Results**: A JSON file like `experimental_results_...json` is saved for each individual run.


## Citation

_currently under-review_

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)


