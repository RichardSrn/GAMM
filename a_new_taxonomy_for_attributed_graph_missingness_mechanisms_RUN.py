# general imports
import os
import sys
import time
from tqdm.auto import tqdm
import argparse
from datetime import datetime

# data manipulation
import numpy as np
import pandas as pd
import networkx as nx

# scipy
import scipy
from scipy import optimize

# torch
import torch
from torch_geometric.utils import degree
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_mean
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid, WebKB, HeterophilousGraphDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GatedGraphConv, GraphConv, JumpingKnowledge

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    confusion_matrix

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

path = "./a_new_taxonomy_for_attributed_graph_missingness_mechanisms"

# Ensure the results directory exists
if not os.path.exists(path):
    os.makedirs(path)

run_tests = False

# Import imputation methods, mask generators, aggregator functions, dataset loader and the new evaluation helpers.
from a_new_taxonomy_for_attributed_graph_missingness_mechanisms_UTILS import (
    impute_random,
    impute_column_average,
    impute_mice,
    impute_ot_tab,
    impute_ot_tab_RR,
    impute_neighbor_average,
    impute_feature_propagation,
    impute_pcfi,
    impute_griot,

    # Mask generators and aggregator functions.
    MissingMaskGenerator,
    GraphMissingMaskGenerator,
    DatasetsLoader,
    aggregator_mar_scenario2,
    aggregator_mar_scenario30,
    aggregator_mar_scenario310,
    aggregator_mnar_scenario3,
    aggregator_mnar_scenario31,
    aggregator_mnar_scenario4
)

# Import evaluation helpers
from sklearn.model_selection import train_test_split  # (if needed elsewhere)
from a_new_taxonomy_for_attributed_graph_missingness_mechanisms_UTILS import create_stratified_splits, compute_imputation_metrics, evaluate_classification

def run_experiment(selected_datasets=("Cora", "Texas", "Tolokers"),
                   missing_rates=(0.2, 0.5, 0.8),
                   output_path="./",
                   exp_indx=1,
                   imputer_names=None):
    """
    Run the full experiment for the given datasets.
    For each dataset, we:
      1. Create stratified node splits (60% train, 20% val, 20% test).
      2. For every missingness mechanism and imputation method:
           - Generate a mask (using either traditional or graph-aware procedure).
           - Run imputation.
           - Compute imputation metrics (MAE, RMSE) only on the test nodes.
           - Train a classifier using imputed features on training nodes,
             then evaluate classification (Accuracy, F1-score, ROC-AUC) on test nodes.
    :param selected_datasets: a list of dataset names.
    :param missing_rates: tuple, missing rates to test.
    :param output_path: directory for saving results (LaTeX and JSON).
    :param exp_indx: Experiment index (used to log the run).
    :param imputer_names: Optional string; if provided, filter the imputation_methods dictionary to run only that model.
    :return: a pandas DataFrame of results.
    """
    # Define the missing mask mechanisms.
    traditional_mechanisms = [
        ("MCAR", {}),
        ("MAR", {"proportion_observed": 0.25}),
        # ("MNAR_quantile_lower", {"option": "quantile", "cut": "lower", "q": 0.25, "p_params": 0.5, "mcar_extra": True}),
        # ("MNAR_quantile_upper", {"option": "quantile", "cut": "upper", "q": 0.25, "p_params": 0.5, "mcar_extra": True}),
        ("MNAR_quantile_both", {"option": "quantile", "cut": "both", "q": 0.25, "p_params": 0.5, "mcar_extra": True}),
        # ("MNAR_selfmasked", {"option": "selfmasked"})
    ]
    graph_mechanisms = [
        ("MAR_scenario3.0.25", {"scenario": 3, "aggregator_prob_funct": aggregator_mar_scenario30, "prop_obs": 0.25}),
        ("MAR_scenario3.1.0.25", {"scenario": 3, "aggregator_prob_funct": aggregator_mar_scenario310, "prop_obs": 0.25}),
        ("MAR_scenario3.0.75", {"scenario": 3, "aggregator_prob_funct": aggregator_mar_scenario30, "prop_obs": 0.75}),
        ("MAR_scenario3.1.0.75", {"scenario": 3, "aggregator_prob_funct": aggregator_mar_scenario310, "prop_obs": 0.75}),
        # ("MNAR_scenario3", {"scenario": 3, "aggregator_prob_funct": aggregator_mnar_scenario3}),
        # # ("MNAR_scenario3.1", {"scenario": 3, "aggregator_prob_funct": aggregator_mnar_scenario31}),
        # ("MNAR_scenario4", {"scenario": 4, "aggregator_prob_funct": aggregator_mnar_scenario4})
    ]
    # Instantiate mask generators.
    mmg = MissingMaskGenerator(seed=42+exp_indx)
    gmmg = GraphMissingMaskGenerator(seed=42+exp_indx)
    mechanisms = []
    for mech_name, params in traditional_mechanisms:
        mechanisms.append((mech_name, params, mmg.generate))
    for mech_name, params in graph_mechanisms:
        mechanisms.append((mech_name, params, gmmg.generate))

    # Define imputation methods.
    imputation_methods = {
        "Tabular_Avg": impute_column_average,
        "Random": impute_random,
        "Graph_1hop": impute_neighbor_average,
        "MICE": impute_mice,  # uncomment if desired
        "FP": impute_feature_propagation,
        "OT-tab": impute_ot_tab,
        "OT-tab-RR": impute_ot_tab_RR,  # will be skipped if feature dimension is too high (see below)
        "PCFI": impute_pcfi,
        "GRIOT": impute_griot,
    }
    # If a model parameter is provided, filter the dictionary.
    if imputer_names is not None:
        imputation_methods = {k:v for k,v in imputation_methods.items() if k in imputer_names}
        # if imputer_names in imputation_methods:
        #     imputation_methods = {imputer_names: imputation_methods[imputer_names]}
        assert len(imputation_methods) > 0, f"Warning: Model '{imputer_names}' not found; using all models."
        assert len(imputation_methods) == len(imputer_names), f"One or multiple imputer(s) could not be loaded."

    results = []
    torch.manual_seed(42)
    np.random.seed(42)
    loader = DatasetsLoader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for graph_name in tqdm(selected_datasets, desc="Dataset Loop", leave=True, colour="red", position=0):
        dataset = loader.load_dataset(graph_name)
        data = dataset[graph_name]._data.to(device)
        original_features = data.x.clone().float()  # ground truth features

        num_nodes = original_features.shape[0]
        # Create stratified splits (based on data.y)
        train_idx, val_idx, test_idx = create_stratified_splits(num_nodes, data.y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42+exp_indx)

        # Loop over each missingness mechanism.
        for mech_name, params, mask_generator in tqdm(mechanisms, desc="Mechanism Loop", leave=False, colour="blue", position=1):
            for mr in missing_rates:
                t0 = time.time()
                # Use the mechanism name's prefix for mask generation.
                base_mech = mech_name.split("_")[0]
                mask = mask_generator(data, base_mech, missing_rate=mr, **params)
                gen_time = time.time() - t0
                missing_prop = (mask == False).sum().item() / mask.numel()

                for impute_name, impute_func in tqdm(imputation_methods.items(), desc="Imputer Loop", leave=False, colour="green", position=2):
                    # Safety check: skip OT-tab-RR if the feature dimension is above threshold.
                    if impute_name == "OT-tab-RR" and original_features.shape[1] > 10:
                        continue
                    t_impute = time.time()
                    imputed_features = impute_func(data, mask, train_idx=train_idx, val_idx=val_idx, seed=42+exp_indx)
                    imp_time = time.time() - t_impute

                    # Compute imputation metrics only on test nodes.
                    mae, rmse = compute_imputation_metrics(original_features, imputed_features, mask, test_idx)

                    # Classification evaluation using the same splits.
                    try:
                        acc, f1_macro, f1_micro, precision_macro, recall_macro, roc_auc, class_report = evaluate_classification(
                        imputed_features, data, device, train_idx, val_idx, test_idx, num_epochs=200, lr=0.01, seed=42+exp_indx
                    )
                    except Exception as e:
                        acc, f1_macro, f1_micro, precision_macro, recall_macro, roc_auc, class_report = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                        print(f"Error in classification evaluation for {graph_name} with {impute_name}: {e}")


                    results.append({
                        "exp_indx": exp_indx,
                        "Dataset": graph_name,
                        "Mechanism": mech_name,
                        "MissingRate": mr,
                        "ActualMissingProp": missing_prop,
                        "Imputer": impute_name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "Acc": acc,
                        "F1_Macro": f1_macro,
                        "F1_Micro": f1_micro,
                        "Precision_Macro": precision_macro,
                        "Recall_Macro": recall_macro,
                        "ROC_AUC": roc_auc,
                        "MaskGenTime": gen_time,
                        "ImputeTime": imp_time,
                        # "Classification_Report": class_report
                    })

    df = pd.DataFrame(results)
    print("Experimental Results:")
    print(df)

    ds_str = "_".join(selected_datasets)
    timestamp = str(time.time()).split('.')[0]

    # latex_table = df.to_latex(index=False, float_format="%.4f")
    # latex_filename = f"{output_path}/experimental_results_{ds_str}_{timestamp}.tex"
    # with open(latex_filename, "w") as f:
    #     f.write(latex_table)
    # print(f"\nLaTeX table written to {latex_filename}")

    json_filename = f"{output_path}/experimental_results_{ds_str}_{timestamp}_{exp_indx}.json"
    df.to_json(json_filename, orient="records", indent=4)
    print(f"\nJSON file written to {json_filename}")

    return df

#####################################
# MAIN execution block
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run graph data imputation experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset argument
    parser.add_argument(
        "-d", "--datasets", type=str, default="Texas",
        help="Dataset(s) to use, comma-separated with no spaces. "
             "Available options: Cora, CiteSeer, PubMed, Texas, Wisconsin, Cornell, "
             "Minesweeper, Tolokers, Questions"
    )

    # Imputation model argument
    parser.add_argument(
        "-m", "--models", type=str, default=None,
        help="Imputation model(s) to use, comma-separated with no spaces. "
             "Available options: Tabular_Avg, Random, Graph_1hop, MICE, FP, OT-tab, PCFI, GRIOT. "
             "If not specified, all models will be used."
    )

    # Missing rates argument
    parser.add_argument(
        "--mr", "--missing-rates", type=str, default="0.2,0.5,0.8",
        dest="missing_rates",
        help="Missing rate(s) to use, comma-separated with no spaces. "
             "Should be values between 0 and 1 (e.g., 0.2,0.5,0.8)"
    )

    # Output path argument
    parser.add_argument(
        "-o", "--output-path", type=str,
        default="./a_new_taxonomy_for_attributed_graph_missingness_mechanisms",
        help="Path to save experiment results"
    )

    # Number of experiment runs argument
    parser.add_argument(
        "-r", "--runs", type=int, default=3,
        help="Number of times to repeat the experiment"
    )

    # Start or exp index
    parser.add_argument(
        "-s", "--start", type=int, default=0,
        help="Start or exp index"
    )
    # Main component argument
    parser.add_argument(
        "--keep-main-component", action="store_true", default=True,
        help="Only keep the main connected component of each graph"
    )

    # Verbose/debug mode argument
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print verbose debugging information"
    )

    return parser.parse_args()


def print_debug_info(config):
    """Print debug and environment information for reproducibility."""
    print("\n" + "="*50)
    print(" EXPERIMENT CONFIGURATION ")
    print("="*50)

    # print full sys.argv
    print(f"sys.argv: {' '.join(sys.argv)}")

    # Print timestamp
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print configuration
    print("\nExperiment Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Print Python and PyTorch versions
    print("\nEnvironment:")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU(s): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    Device {i}: {torch.cuda.get_device_name(i)}")

    # Print key environment variables
    print("\nKey environment variables:")
    env_vars_to_check = [
        "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF", "PYTHONHASHSEED"
    ]
    for var in env_vars_to_check:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")

    print("="*50 + "\n")


def run_experiments_with_config(config):
    """
    Run experiments with the given configuration.

    This is the core function that can be called from both the script and notebook.

    Parameters:


----


    config : dict
        Dictionary containing all experiment parameters:
        - selected_datasets: list of dataset names
        - imputer_names: list of imputer names or None
        - missing_rates: tuple of missing rates
        - output_path: path to save results
        - runs: number of experiment runs
        - start: start index for exp.
        - keep_main_component: whether to keep only the main component
        - verbose: whether to print debug info

    Returns:


----


    pandas.DataFrame
        Aggregated results from all experiment runs
    """
    # Extract parameters from config
    selected_datasets = config["selected_datasets"]
    imputer_names = config["imputer_names"]
    missing_rates = config["missing_rates"]
    output_path = config["output_path"]
    runs = config["runs"]
    print(config)
    start = config["start"]
    verbose = config["verbose"]

    # Print experiment setup
    print("\nStarting experiments with:")
    print(f"  Datasets: {selected_datasets}")
    print(f"  Imputers: {imputer_names if imputer_names else 'All available'}")
    print(f"  Missing rates: {missing_rates}")
    print(f"  Output path: {output_path}")
    print(f"  Experiment runs: {runs}")
    print(f"  Start of exp index: {start}")
    print(f"  Keep main component: {config['keep_main_component']}")

    # Run experiments
    all_results = []
    for exp in range(start + 1, start + runs + 1):
        print(f"\nStarting experiment run {exp}/{runs}...")
        t_start = time.time()

        df_exp = run_experiment(
            selected_datasets=selected_datasets,
            missing_rates=missing_rates,
            output_path=output_path,
            exp_indx=exp,
            imputer_names=imputer_names
        )

        all_results.append(df_exp)
        elapsed = time.time() - t_start
        print(f"Run {exp} completed in {int(elapsed // 60)}min {int(elapsed % 60)}s")

    # Concatenate all results
    if len(all_results) > 0:
        final_df = pd.concat(all_results, ignore_index=True)
        print("\nFinal aggregated experimental results:")
        print(final_df)

        # Optionally save the aggregated results
        ds_str = "_".join(selected_datasets)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_path, f"aggregated_results_{ds_str}_{runs}runs_{timestamp}.json")
        final_df.to_json(results_file, orient="records", indent=4)
        print(f"\nAggregated results saved to: {results_file}")

        return final_df
    else:
        print("\nNo results were generated from the experiments.")
        return None


def process_args_to_config(args):
    """
    Process parsed arguments into a configuration dictionary.

    Parameters:


----


    args : argparse.Namespace
        Parsed command-line arguments

    Returns:


----


    dict
        Configuration dictionary for run_experiments_with_config
    """
    # Process datasets
    selected_datasets = [d.strip() for d in args.datasets.split(",")]
    valid_datasets = ["Cora", "CiteSeer", "PubMed", "Texas", "Wisconsin",
                      "Cornell", "Minesweeper", "Tolokers", "Questions"]

    for dataset in selected_datasets.copy():
        if dataset not in valid_datasets:
            print(f"Warning: Invalid dataset '{dataset}'. Skipping.")
            selected_datasets.remove(dataset)

    if not selected_datasets:
        print("No valid datasets specified. Using default 'Texas'.")
        selected_datasets = ["Texas"]

    # Process imputation models
    imputer_names = None
    if args.models:
        imputer_names = [m.strip() for m in args.models.split(",")]
        valid_imputers = ["Tabular_Avg", "Random", "Graph_1hop", "MICE",
                          "FP", "OT-tab", "PCFI", "GRIOT"]

        for imputer in imputer_names.copy():
            if imputer not in valid_imputers:
                print(f"Warning: Invalid imputer '{imputer}'. Skipping.")
                imputer_names.remove(imputer)

    # Process missing rates
    missing_rates = tuple(float(mr.strip()) for mr in args.missing_rates.split(","))

    # Create config dictionary
    config = {
        "selected_datasets": selected_datasets,
        "imputer_names": imputer_names,
        "missing_rates": missing_rates,
        "output_path": args.output_path,
        "runs": args.runs,
        "start": args.start,
        "keep_main_component": args.keep_main_component,
        "verbose": args.verbose
    }

    return config


def main():
    """Main function to run the experiments."""
    # Parse command-line arguments
    args = parse_arguments()

    # Process arguments into a configuration dictionary
    config = process_args_to_config(args)

    # Print debug information if requested
    if config["verbose"]:
        print_debug_info(config)

    # Run experiments with the configuration
    return run_experiments_with_config(config)


if __name__ == "__main__":
    main()









# # Read the first command-line argument as the selected dataset.
# # If no argument is given, use "Cora" as default.
# if len(sys.argv) > 1:
#     if sys.argv[1] in ["Cora", "CiteSeer", "PubMed", "Texas", "Wisconsin", "Cornell", "Minesweeper", "Tolokers", "Questions"]:
#         selected_dataset = sys.argv[1]
#     else:
#         # print warning
#         print(f"Invalid dataset name passed as argv[1]. dataset must be in [Cora, CiteSeer, PubMed, Texas, Wisconsin, Cornell, Minesweeper, Tolokers, Questions],  but got '{sys.argv[1]}'. Using default dataset 'Texas'.")
#         selected_dataset = "Texas"
# else:
#     selected_dataset = "Texas"
#
# # Read the second command-line argument for the imputer model (if available).
# if len(sys.argv) > 2:
#     if sys.argv[2] in ["Tabular_Avg", "Random", "Graph_1hop", "MICE", "FP", "OT-tab", "PCFI", "GRIOT",]:
#         imputer_names = sys.argv[2]
#     else:
#         # print warning
#         print(f"Invalid imputer name passed as argv[2]. imputer must be in [Tabular_Avg, Random, Graph_1hop, MICE, FP, OT-tab, PCFI, GRIOT],  but got '{sys.argv[2]}'. Using all imputers as default.")
#         imputer_names = None
# else:
#     imputer_names = None  # Use all imputation methods if not specified.
#
# # We expect a single dataset name, so we wrap it in a list.
# selected_datasets = [selected_dataset]
#
# all_results = []
# # Run the experiment 5 times.
# for exp in range(1, 2):  # 6):
#     t_start = time.time()
#     df_exp = run_experiment(selected_datasets=selected_datasets, missing_rates=(0.2,),# 0.5, 0.8),
#                             output_path="./a_new_taxonomy_for_attributed_graph_missingness_mechanisms",
#                             exp_indx=exp, imputer_names=imputer_names)
#     all_results.append(df_exp)
#     elapsed = time.time() - t_start
#     print(f"Run {exp} time: {int(elapsed // 60)}min {int(elapsed % 60)}s")
#
# # Optionally, concatenate all results into one dataframe
# final_df = pd.concat(all_results, ignore_index=True)
# print("Final aggregated experimental results:")
# print(final_df)