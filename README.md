# GAMM (Graph Attribute Missing Mechanisms)

A framework for generating masks, evaluating, and comparing graph imputation methods.

## Overview

This repository implements a comprehensive taxonomy for missingness mechanisms in attributed graphs and provides tools for testing various imputation methods. The framework supports multiple popular graph datasets and state-of-the-art imputation algorithms.

## Installation

```bash
git clone https://anonymous.4open.science/r/GAMM-B420   # clonning may not work with anonymous.4open.science, proceed manually while the repository is anonymized.
cd GAMM
pip install -r requirements.txt
```

## Quick Start

### Using the Jupyter Notebook

Open `a_new_taxonomy_for_attributed_graph_missingness_mechanisms.ipynb` and locate cell [3] with the `notebook_run` function:

```python
results = notebook_run(
    datasets=["Cornell"],
    models=["FP"],
    missing_rates=(0.2,),
    runs=1
)
```

### Parameters

- **datasets**: Available options
  - Citation networks: `Cora`, `CiteSeer`, `PubMed`
  - WebKB graphs: `Texas`, `Wisconsin`, `Cornell`
  - Social networks: `Minesweeper`, `Tolokers`, `Questions` (may leed to OOM error)

- **models**: Available imputers
  - Baseline methods: `Tabular_Avg`, `Random`, `Graph_1hop`
  - Graph-specific methods: `FP`, `OT-tab`, `PCFI`, `GRIOT`

- **missing_rates**: Any value between 0 and 1
- **runs**: Number of experimental repetitions (integer > 0)

### Command Line Interface

Alternatively, use the CLI script:

```bash
python3 a_new_taxonomy_for_attributed_graph_missingness_mechanisms_RUN.py \
    --datasets Cora,CiteSeer \
    --models FP,GRIOT \
    --missing-rates 0.2,0.5 \
    --runs 5
```

For full CLI options:
```bash
python3 a_new_taxonomy_for_attributed_graph_missingness_mechanisms_RUN.py --help
```

## Output

Results are saved in the specified output directory (default: `./a_new_taxonomy_for_attributed_graph_missingness_mechanisms/`) in CSV format.

## Citation

If you use this code in your research, please cite:
```bibtex
currently under review
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

![CC BY 4.0][cc-by-shield]

[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
