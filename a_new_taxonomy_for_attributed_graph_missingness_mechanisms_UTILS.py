from torch_geometric.utils import subgraph, to_networkx
# import random
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch_geometric.utils
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph
import sys
import os
from torch_geometric.datasets import Planetoid, WebKB, HeterophilousGraphDataset, WikipediaNetwork, Actor
import torch
import matplotlib.pyplot as plt
from torch import nn
from geomloss import SamplesLoss
from geomloss.sinkhorn_divergence import scaling_parameters, sinkhorn_loop, log_weights, sinkhorn_cost
from geomloss.sinkhorn_samples import softmin_tensorized
import torch.optim
import time
import numpy as np
from scipy import optimize
from torch_scatter import scatter_add
from tqdm.auto import tqdm
from typing import Optional
import torch.nn.functional as F
import ot
import networkx as nx
import pandas as pd
from torch_geometric.nn import GCNConv, JumpingKnowledge  # or any other conv layer you wish to use
from torch.nn import ModuleList, Linear

from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, classification_report


import warnings
warnings.filterwarnings("ignore",
    message="It is not recommended to directly access the internal storage format `data` of an 'InMemoryDataset'.*",
    category=UserWarning)



# -----------------------------------------------------------------------------------------------------------------
#                               ▗▖ ▗▖▗▄▄▄▖▗▄▄▄▖▗▖    ▗▄▄▖
#                               ▐▌ ▐▌  █    █  ▐▌   ▐▌
#                               ▐▌ ▐▌  █    █  ▐▌    ▝▀▚▖
#                               ▝▚▄▞▘  █  ▗▄█▄▖▐▙▄▄▖▗▄▄▞▘
# -----------------------------------------------------------------------------------------------------------------

def create_stratified_splits(num_nodes, labels, train_ratio=0.6, val_ratio=0.1, test_ratio=0.3, random_state=0):
    """
    Create a stratified split of node indices.
    labels: torch.Tensor (or array-like) of node labels.
    Returns: train_idx, val_idx, test_idx (as torch.tensor)
    """
    all_idx = np.arange(num_nodes)
    # First split: training (60%) and temporary pool (40%)
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=(1 - train_ratio), stratify=labels.cpu().numpy(), random_state=random_state
    )
    # Second split: split temporary pool equally into validation and test (each = 20% of total)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio/(1-train_ratio), stratify=labels[temp_idx].cpu().numpy(), random_state=random_state
    )

    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(val_idx, dtype=torch.long), torch.tensor(test_idx, dtype=torch.long)


def compute_imputation_metrics(true_features, imputed_features, mask, test_idx):
    """
    Compute MAE and RMSE on missing entries over the test nodes only.
    true_features, imputed_features, mask: torch.Tensor of shape (num_nodes,d)
    test_idx: torch.Tensor of indices representing test nodes.
    """
    # Restrict each tensor to test nodes.
    true_test = true_features[test_idx]
    imputed_test = imputed_features[test_idx]
    mask_test = mask[test_idx]

    # Compute only for missing entries (mask == False).
    missing = ~mask_test
    if missing.sum() == 0:
        return np.nan, np.nan
    diff = true_test[missing] - imputed_test[missing]
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    return mae, rmse


def evaluate_classification(imputed_features, data, device, train_idx, val_idx, test_idx, num_epochs=200, lr=0.01, seed=0):
    """
    Train the classifier (GNNClassifier) on the imputed features using the pre-defined
    train_idx (for training) and assess performance on test_idx.
    Returns: Accuracy, Macro F1 and ROC-AUC (if binary, else nan)
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    x = imputed_features.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    num_features = x.shape[1]
    num_classes = int(y.max().item()) + 1  # Assumes labels {0, ..., num_classes -1}

    model = GNNClassifier(num_features=num_features, num_classes=num_classes,
                          hidden_dim=64, num_layers=2, dropout=0.5, conv_type="GCN", seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    # Evaluate on test nodes
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.exp(logits)  # probabilities from log_softmax
        preds = logits.argmax(dim=1)
        test_true = y[test_idx].cpu().numpy()
        test_pred = preds[test_idx].cpu().numpy()
        test_probs = probs[test_idx].cpu().numpy()

    acc = accuracy_score(test_true, test_pred)
    f1_macro = f1_score(test_true, test_pred, average='macro')
    f1_micro = f1_score(test_true, test_pred, average='micro')
    precision_macro = precision_score(test_true, test_pred, average='macro', zero_division=0)
    recall_macro = recall_score(test_true, test_pred, average='macro')
    # Only compute ROC-AUC for binary tasks
    roc = roc_auc_score(test_true, test_probs[:, 1]) if num_classes == 2 else np.nan

    # Generate classification report
    class_report = classification_report(test_true, test_pred, zero_division=0)

    return acc, f1_macro, f1_micro, precision_macro, recall_macro, roc, class_report



# -----------------------------------------------------------------------------------------------------------------
#                               ▗▄▄▄   ▗▄▖▗▄▄▄▖▗▄▖
#                               ▐▌  █ ▐▌ ▐▌ █ ▐▌ ▐▌
#                               ▐▌  █ ▐▛▀▜▌ █ ▐▛▀▜▌
#                               ▐▙▄▄▀ ▐▌ ▐▌ █ ▐▌ ▐▌
# -----------------------------------------------------------------------------------------------------------------

class DatasetsLoader:
    def __init__(self, root=None, normalize=True, drop_small_classes=True, keep_main_component=True, verbose=False):
        """
        Initialize the DatasetsLoader.

        :param root: Base directory for storing datasets. If None, defaults to "{cwd}/data".
        :param normalize: If True, normalize features to be between 0 and 1.
        :param drop_small_classes: If True, drop nodes belonging to underrepresented classes.
        :param keep_main_component: If True, filter each graph to only keep its largest connected component.
        """
        if root is None:
            self.root = os.path.join(os.getcwd(), 'data')
        else:
            self.root = root

        # Create the base directory if it doesn't exist.
        os.makedirs(self.root, exist_ok=True)

        self.normalize = normalize
        self.drop_small_classes = drop_small_classes
        self.keep_main_component_flag = keep_main_component
        self.verbose=verbose

    def _load_planetoid(self, graphs='all'):
        """
        Load the Planetoid datasets.

        :param graphs: 'all', 'cora', 'citeseer', or 'pubmed'. Not case sensitive.
        :return: A dictionary of dataset objects.
        """
        name_map = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}

        datasets = {}
        if isinstance(graphs, str) and graphs.lower() == 'all':
            graphs_list = ['Cora', 'CiteSeer', 'PubMed']
        elif isinstance(graphs, str):
            # Capitalize first letter; special handling for PubMed
            graphs_list = [name_map[graphs.lower()]]
        elif isinstance(graphs, list):
            graphs_list = []
            for g in graphs:
                graphs_list.append(name_map[g.lower()])
        else:
            raise TypeError("Parameter 'graphs' must be a string or a list of strings.")

        for graph in graphs_list:
            dataset_root = os.path.join(self.root, graph)
            os.makedirs(dataset_root, exist_ok=True)
            datasets[graph] = Planetoid(root=dataset_root, name=graph)
        return datasets

    def _load_webkb(self, graphs='all'):
        """
        Load the WebKB datasets.

        :param graphs: 'all', 'cornell', 'texas', or 'wisconsin'. Not case sensitive.
        :return: A dictionary of dataset objects.
        """
        datasets = {}
        if isinstance(graphs, str) and graphs.lower() == 'all':
            graphs_list = ['Cornell', 'Texas', 'Wisconsin']
        elif isinstance(graphs, str):
            graphs_list = [graphs.capitalize()]
        elif isinstance(graphs, list):
            graphs_list = [g.capitalize() for g in graphs]
        else:
            raise TypeError("Parameter 'graphs' must be a string or a list of strings.")

        for graph in graphs_list:
            dataset_root = os.path.join(self.root, graph)
            os.makedirs(dataset_root, exist_ok=True)
            datasets[graph] = WebKB(root=dataset_root, name=graph)
        return datasets

    def _load_heterophilous(self, graphs="all"):
        """
        Load the HeterophilousGraphDataset datasets.

        :param graphs: "all", "Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions",
                       or a list containing any of these (case-insensitive).
        :return: A dictionary of dataset objects.
        """
        available_names = ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]
        datasets = {}

        def normalize(name):
            return next((an for an in available_names if an.lower() == name.lower()), None)

        if isinstance(graphs, str):
            if graphs.lower() == "all":
                selected_names = available_names
            else:
                proper_name = normalize(graphs)
                if proper_name is None:
                    raise ValueError(f"Dataset '{graphs}' not found. Available options are: {available_names}")
                selected_names = [proper_name]
        elif isinstance(graphs, list):
            selected_names = []
            for name in graphs:
                proper_name = normalize(name)
                if proper_name is None:
                    raise ValueError(f"Dataset '{name}' not found. Available options are: {available_names}")
                selected_names.append(proper_name)
        else:
            raise TypeError("Parameter 'graphs' must be a string or a list of strings.")

        for name in selected_names:
            dataset_root = os.path.join(self.root, name)
            os.makedirs(dataset_root, exist_ok=True)
            datasets[name] = HeterophilousGraphDataset(root=dataset_root, name=name)
        return datasets

    def _load_wikipedia_network(self, graphs='all'):
        """
        Load the WikipediaNetwork datasets.
        :param graphs: 'all', 'squirrel', 'chameleon', or a list containing any of these (case-insensitive).
        :return: A dictionary of dataset objects.
        """
        available_names = ['Squirrel', 'Chameleon']
        datasets = {}
        def normalize_name(name):
            return next((n for n in available_names if n.lower() == name.lower()), None)
        if isinstance(graphs, str):
            if graphs.lower() == 'all':
                selected_names = available_names
            else:
                proper_name = normalize_name(graphs)
                if proper_name is None:
                    raise ValueError(f"Dataset '{graphs}' not found. Available datasets are: {available_names}")
                selected_names = [proper_name]
        elif isinstance(graphs, list):
            selected_names = []
            for name in graphs:
                proper_name = normalize_name(name)
                if proper_name is None:
                    raise ValueError(f"Dataset '{name}' not found. Available datasets are: {available_names}")
                selected_names.append(proper_name)
        else:
            raise TypeError("Parameter 'graphs' must be a string or a list of strings.")
        for name in selected_names:
            dataset_root = os.path.join(self.root, name)
            os.makedirs(dataset_root, exist_ok=True)
            datasets[name] = WikipediaNetwork(root=dataset_root, name=name)
        return datasets

    def _load_actor(self):
        """
        Load the Actor dataset.
        :return: A dictionary containing the Actor dataset.
        """
        dataset_root = os.path.join(self.root, 'Actor')
        os.makedirs(dataset_root, exist_ok=True)
        return {'Actor': Actor(root=dataset_root)}

    @staticmethod
    def normalize_features(dataset):
        """
        Normalize the features of a dataset to be between 0 and 1.

        :param dataset: A PyG dataset object.
        :return: The dataset with normalized features.
        """
        if not hasattr(dataset, 'data'):
            raise ValueError("The provided dataset does not have a 'data' attribute.")
        if not hasattr(dataset._data, 'x'):
            raise ValueError("The provided dataset does not have a 'x' attribute in 'data'.")

        x = dataset._data.x
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values

        # check if any feature has a constant value (x_min == x_max)
        # where the features are constant, we don't normalize them
        if (x_min == x_max).any():
            constant_features = (x_min == x_max)
            x_min[constant_features] = 0
            x_max[constant_features] = 1

        x_norm = (x - x_min) / (x_max - x_min)

        return x_norm


    def drop_irrelevant(self, datasets):
        """
        Drop nodes in each dataset that belong to classes with fewer than 3 representatives.

        The method modifies each dataset's _data in place and reindexes the edge_index accordingly.

        :param datasets: dictionary mapping dataset names to dataset objects.
        :return: The modified datasets (dict).
        """
        # Iterate over each dataset.
        for key, dataset in datasets.items():
            data = dataset._data
            # Get counts for each unique class in data.y.
            unique_labels, counts = torch.unique(data.y, return_counts=True)
            # Identify labels that have at least 3 instances.
            valid_labels = unique_labels[counts >= 3]
            if valid_labels.numel() == 0:
                raise ValueError(f"All classes in dataset {key} are underrepresented.")

            # Build a mask for nodes that belong to valid classes.
            mask = torch.zeros_like(data.y, dtype=torch.bool)
            for valid in valid_labels:
                mask |= (data.y == valid)
            valid_idx = mask.nonzero(as_tuple=False).view(-1)

            # Use PyG's subgraph utility to retain only the valid nodes and update edge_index.
            new_edge_index, new_edge_attr = subgraph(valid_idx, data.edge_index, relabel_nodes=True, num_nodes=data.y.size(0))

            # Update the dataset's data.
            data.x = data.x[valid_idx]
            data.y = data.y[valid_idx]
            data.edge_index = new_edge_index
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = new_edge_attr
        return datasets

    def keep_main_component(self, datasets):
        """
        For each dataset, keep only the largest connected component.
        This method updates each dataset's _data in place.
        :param datasets: Dictionary mapping dataset names to dataset objects.
        :return: The modified datasets dictionary.
        """
        for key, dataset in datasets.items():
            data = dataset._data
            try:
                # Convert to an undirected NetworkX graph.
                G = to_networkx(data, to_undirected=True)
            except Exception as e:
                print(f"Error converting dataset {key} to NetworkX: {e}")
                continue

            try:
                # Find the largest connected component.
                largest_cc = max(nx.connected_components(G), key=len)
                # Convert node IDs to a tensor.
                main_nodes = torch.tensor(list(largest_cc), dtype=torch.long)
            except Exception as e:
                print(f"Error finding the largest connected component for {key}: {e}")
                continue

            # Use PyG's subgraph utility to extract the main component.
            new_edge_index, new_edge_attr = subgraph(main_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.x.size(0))
            data.x = data.x[main_nodes]
            data.y = data.y[main_nodes]
            data.edge_index = new_edge_index
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = new_edge_attr
            if self.verbose:
                print(f"Dataset {key}: kept main component with {len(main_nodes)} nodes.")
        return datasets

    # def load_dataset(self, graph_name):
    #     """
    #     Generalized method to load datasets based on the provided graph_name parameter.
    #     :param graph_name: A string or a list of strings specifying which dataset(s) to load.
    #     :return: A dictionary of dataset objects.
    #     """
    #     planetoid_names = {"cora", "citeseer", "pubmed"}
    #     webkb_names = {"cornell", "texas", "wisconsin"}
    #     hetero_names = {"roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}
    #
    #     datasets = {}
    #
    #     def process_single(name):
    #         name_lower = name.lower()
    #         if name_lower == "all":
    #             datasets.update(self._load_planetoid("all"))
    #             datasets.update(self._load_webkb("all"))
    #             datasets.update(self._load_heterophilous("all"))
    #         elif name_lower == "planetoid":
    #             datasets.update(self._load_planetoid("all"))
    #         elif name_lower == "webkb":
    #             datasets.update(self._load_webkb("all"))
    #         elif name_lower == "heterophilous":
    #             datasets.update(self._load_heterophilous("all"))
    #         elif name_lower in planetoid_names:
    #             datasets.update(self._load_planetoid(name))
    #         elif name_lower in webkb_names:
    #             datasets.update(self._load_webkb(name))
    #         elif name_lower in hetero_names:
    #             datasets.update(self._load_heterophilous(name))
    #         else:
    #             raise ValueError(f"Dataset '{name}' not recognized. "
    #                              "Please use 'all', 'planetoid', 'webkb', 'heterophilous', "
    #                              "or one of the available dataset names.")
    #
    #     if isinstance(graph_name, list):
    #         for name in graph_name:
    #             process_single(name)
    #     elif isinstance(graph_name, str):
    #         process_single(graph_name)
    #     else:
    #         raise TypeError("The parameter 'graph_name' must be a string or a list of strings.")
    #
    #     if self.normalize:
    #         for key, dataset in datasets.items():
    #             dataset._data.x = self.normalize_features(dataset)
    #
    #     if self.drop_small_classes:
    #         datasets = self.drop_irrelevant(datasets)
    #
    #     # If keeping only the main component is enabled, apply the method here.
    #     if self.keep_main_component_flag:
    #         datasets = self.keep_main_component(datasets)
    #
    #     return datasets
    def load_dataset(self, graph_name):
        """
        Generalized method to load datasets based on the provided graph_name parameter.
        :param graph_name: A string or a list of strings specifying which dataset(s) to load.
        :return: A dictionary of dataset objects.
        """
        planetoid_names = {"cora", "citeseer", "pubmed"}
        webkb_names = {"cornell", "texas", "wisconsin"}
        hetero_names = {"roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"}
        wikipedia_names = {"squirrel", "chameleon"}
        datasets = {}
        def process_single(name):
            name_lower = name.lower()

            if name_lower == "all":
                datasets.update(self._load_planetoid("all"))
                datasets.update(self._load_webkb("all"))
                datasets.update(self._load_heterophilous("all"))
                datasets.update(self._load_wikipedia_network("all"))
                datasets.update(self._load_actor())
            elif name_lower == "planetoid":
                datasets.update(self._load_planetoid("all"))
            elif name_lower == "webkb":
                datasets.update(self._load_webkb("all"))
            elif name_lower == "heterophilous":
                datasets.update(self._load_heterophilous("all"))
            elif name_lower in planetoid_names:
                datasets.update(self._load_planetoid(name))
            elif name_lower in webkb_names:
                datasets.update(self._load_webkb(name))
            elif name_lower in hetero_names:
                datasets.update(self._load_heterophilous(name))
            elif name_lower in wikipedia_names:
                datasets.update(self._load_wikipedia_network(name))
            elif name_lower == "actor":
                datasets.update(self._load_actor())
            else:
                raise ValueError(f"Dataset '{name}' not recognized. "
                                 "Please use 'all', 'planetoid', 'webkb', 'heterophilous', 'wikipedia', 'actor', "
                                 "or one of the available dataset names.")
        if isinstance(graph_name, list):
            for name in graph_name:
                process_single(name)
        elif isinstance(graph_name, str):
            process_single(graph_name)
        else:
            raise TypeError("The parameter 'graph_name' must be a string or a list of strings.")
        if self.normalize:
            for key, dataset in datasets.items():
                dataset._data.x = self.normalize_features(dataset)
        if self.drop_small_classes:
            datasets = self.drop_irrelevant(datasets)
        if self.keep_main_component_flag:
            datasets = self.keep_main_component(datasets)
        return datasets


# -----------------------------------------------------------------------------------------------------------------
#                               ▗▖  ▗▖ ▗▄▖  ▗▄▄▖▗▖ ▗▖ ▗▄▄▖
#                               ▐▛▚▞▜▌▐▌ ▐▌▐▌   ▐▌▗▞▘▐▌
#                               ▐▌  ▐▌▐▛▀▜▌ ▝▀▚▖▐▛▚▖  ▝▀▚▖
#                               ▐▌  ▐▌▐▌ ▐▌▗▄▄▞▘▐▌ ▐▌▗▄▄▞▘
# -----------------------------------------------------------------------------------------------------------------

class MissingMaskGenerator:
    def __init__(self, seed=None, sigma=None, output_probs=False):
        """
        :param seed: Optional random seed to ensure determinism.
        :param sigma: A function mapping a torch.Tensor to a torch.Tensor in [0,1].
                      (Used in MCAR, MAR or when a simple logistic mapping is desired.)
                      Defaults to torch.sigmoid if not provided.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.sigma = sigma if sigma is not None else torch.sigmoid
        self.output_probs = output_probs

    @staticmethod
    def nanmean(v, *args, **kwargs):
        """
        PyTorch version of numpy.nanmean.
        """
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

    @staticmethod
    def quantile(X, q, dim=None):
        """
        Compute the q-th quantile along the specified dimension.
        Uses kthvalue.

        :param X: torch.Tensor
        :param q: quantile level in [0,1]
        :param dim: dimension along which to compute
        :return: Tensor of quantiles along dim
        """
        # Number of elements along the dimension
        k = int(q * len(X))
        # kthvalue returns (values, indices)
        quant = X.kthvalue(k, dim=dim)[0]
        return quant

    @staticmethod
    def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
        """
        Pick coefficients for a logistic model.
        :param X: input data (n x d)
        :param idxs_obs: list/array of column indices used as inputs
        :param idxs_nas: list/array of column indices where missingness is generated.
        :param self_mask: If True, for self-masked logistic.
        :return: coefficients (tensor)
        """
        n, d = X.shape
        if self_mask:
            coeffs = torch.randn(d, device=X.device)
            Wx = X * coeffs  # elementwise multiplication
            coeffs /= torch.std(Wx, 0) + 1e-5
        else:
            # Using a subset of features for the logistic model.
            d_obs = len(idxs_obs)
            d_na = len(idxs_nas)
            # coefficients is a (d_obs x d_na) matrix.
            coeffs = torch.randn(d_obs, d_na, device=X.device)
            Wx = X[:, idxs_obs].mm(coeffs)
            coeffs /= torch.std(Wx, 0, keepdim=True)
        return coeffs

    @staticmethod
    def fit_intercepts(X, coeffs, p, self_mask=False):
        """
        Choose intercepts so that the logistic model produces an average missing rate close to p.
        Uses a bisection method.

        :param X: For non-self-masked: input data corresponding to features used as regressors.
                  For self-masked: full feature matrix.
        :param coeffs: Coefficients from pick_coeffs.
        :param p: Target missing rate (float in [0,1])
        :param self_mask: bool
        :return: intercepts (tensor)
        """
        if self_mask:
            d = coeffs.shape[0]
            intercepts = torch.zeros(d, device=X.device)
            for j in range(d):
                def f(x):
                    return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
                intercepts[j] = optimize.bisect(f, -50, 50)
        else:
            d_obs, d_na = coeffs.shape
            intercepts = torch.zeros(d_na, device=X.device)
            for j in range(d_na):
                # Here X is (n, d_obs) and coeffs[:, j] is (d_obs,)
                def f(x):
                    return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
                intercepts[j] = optimize.bisect(f, -50, 50)
        return intercepts

    def generate_mcar(self, data, missing_rate):
        """
        MCAR: each entry is missing with probability equal to missing_rate.
        :param data: PyG data object with attribute x [n x d].
        :param missing_rate: float in [0,1]
        :return: A Boolean mask of shape (n x d) where True means the value is observed.
        """
        assert 0.0 <= missing_rate <= 1.0, "'missing_rate' must be in [0,1]"
        X = data.x
        n, d = X.shape
        prob = torch.full((n, d), 1 - missing_rate, device=X.device, dtype=torch.float32)

        if self.output_probs:
            return prob

        mask = torch.bernoulli(prob).bool()
        return mask

    def generate_mar(self, data, missing_rate, proportion_observed):
        """
        MAR: A subset (proportion_observed) of features remains fully observed.
        For the remaining features, missingness depends on a logistic model applied to a summary (the mean)
        of the fully observed features.

        :param data: PyG data object with attribute x.
        :param missing_rate: target missing rate for features that can be missing.
        :param proportion_observed: proportion in [0,1] of features that are fully observed.
        :return: Boolean mask (n x d)
        """
        assert 0.0 <= missing_rate <= 1.0, "'missing_rate' must be in [0,1]"
        assert 0.0 <= proportion_observed <= 1.0, "'proportion_observed' must be in [0,1]"
        X = data.x
        n, d = X.shape

        num_obs = int(round(proportion_observed * d))
        all_indices = np.arange(d)
        if self.seed is not None:
            np.random.seed(self.seed)
        np.random.shuffle(all_indices)
        S_obs = all_indices[:num_obs]
        S_mis = all_indices[num_obs:]

        # adjust missing rate for the unobserved features to match the target missing rate
        missing_rate = missing_rate / (1 - proportion_observed)

        mask = torch.ones_like(X, dtype=torch.bool)
        if self.output_probs:
            probs_ = torch.zeros_like(X, dtype=torch.float32)

        # For features allowed to be missing, compute a row-level statistic from fully observed features.
        if len(S_obs) == 0:
            p_row = torch.full((n,), missing_rate, device=X.device, dtype=torch.float32)
        else:
            obs_mean = X[:, S_obs].float().mean(dim=1)
            p_row = self.sigma(obs_mean)
            # Calibrate scaling so that average missing probability approximates missing_rate
            scaling = missing_rate / (p_row.mean() + 1e-8)
            p_row = torch.clamp(p_row * scaling, max=1.0)
        for i in range(n):
            probs = torch.full((len(S_mis),), 1.0 - p_row[i], device=X.device, dtype=torch.float32)
            row_mask = torch.bernoulli(probs).bool()
            mask[i, S_mis] = row_mask
            if self.output_probs:
                probs_[i, S_mis] = probs

        if self.output_probs:
            return probs_

        return mask

    def generate_mnar_quantile(self, data, missing_rate, q, p_params, cut='both', mcar_extra=True):
        """
        MNAR by quantiles that adjusts the raw missing probabilities so that the overall missing rate is as desired.
          1. A subset of variables (p_params fraction) is chosen to be "NA candidates".
          2. For each candidate column, quantile thresholds (cut: 'lower', 'upper' or 'both') are computed,
             and eligible entries are those that fall in the tail(s).
          3. Either only the quantile mechanism is used or, if mcar_extra=True, an extra MCAR mechanism is applied
             over the full matrix.

        The function then scales the probability of the quantile mechanism (when used alone) or the additional MCAR
        component (when combined via union) so that the overall missing proportion is the desired `target_missing_rate`.

        :param data: PyG data object with attribute x.
        :param missing_rate: Desired overall missing proportion (float in [0,1]).
        :param q: quantile level (e.g. 0.25 for lower 25% or 0.5 for median-based cuts).
        :param p_params: Fraction in [0,1] of variables that are candidates for MNAR masking.
        :param cut: One of 'lower', 'upper', or 'both'. Determines which tail(s) are censored.
        :param mcar_extra: If True, an extra MCAR mechanism is applied on top.
        :return: Boolean mask (n x d) with True meaning observed and False meaning missing.
        """
        assert 0.0 <= missing_rate <= 1.0, "'target_missing_rate' must be in [0,1]"
        assert 0.0 <= q <= 1.0, "'q' must be in [0,1]"
        assert 0.0 <= p_params <= 1.0, "'p_params' must be in [0,1]"

        X = data.x.float()
        n, d = X.shape
        mask = torch.zeros(n, d, dtype=torch.bool, device=X.device)

        if self.output_probs:
            probs_ = torch.zeros(n, d, dtype=torch.float32, device=X.device)
            # Default probability of being observed
            probs_.fill_(1.0)

        # Save missing_rate as target_missing_rate
        target_missing_rate = missing_rate

        # Determine the number of columns that will be affected.
        d_na = max(int(p_params * d), 1)  # at least one column
        r = d_na / d  # fraction of columns affected, r should equal p_params

        all_idx = np.arange(d)
        if self.seed is not None:
            np.random.seed(self.seed)
        idxs_na = np.random.choice(all_idx, d_na, replace=False)  # candidate columns

        # Compute quantile-based condition for candidate columns.
        if cut == 'upper':
            quants = self.quantile(X[:, idxs_na], 1 - q, dim=0)
            cond = X[:, idxs_na] >= quants
        elif cut == 'lower':
            quants = self.quantile(X[:, idxs_na], q, dim=0)
            cond = X[:, idxs_na] <= quants
        elif cut == 'both':
            u_quants = self.quantile(X[:, idxs_na], 1 - q, dim=0)
            l_quants = self.quantile(X[:, idxs_na], q, dim=0)
            cond = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)
        else:
            raise ValueError("cut must be one of 'lower', 'upper', or 'both'.")

        # Compute p_cond: the overall probability (over the candidate columns and rows) that an entry
        # meets the quantile ("tail") condition.
        p_cond = cond.float().mean().item()

        if mcar_extra:
            # Use the provided missing_rate p_q for the quantile mechanism.
            # Then overall missing in candidate columns from quantile mechanism is r * (p_q * p_cond).
            # We then seek an extra MCAR missing probability "m" so that overall:
            #     overall_missing = m + r * p_q * p_cond * (1 - m)
            # equals target_missing_rate.
            # Solve: m + r * p_q * p_cond - r * p_q * p_cond * m = target_missing_rate
            #        m*(1 - r * p_q * p_cond) + r * p_q * p_cond = target_missing_rate
            #        m = (target_missing_rate - r * p_q * p_cond) / (1 - r * p_q * p_cond)
            p_q = target_missing_rate  # here, using the given missing_rate as the base probability for quantile masking.
            extra_mcar = (target_missing_rate - r * p_q * p_cond) / (1 - r * p_q * p_cond) if (1 - r * p_q * p_cond) > 0 else 0
            # Always clamp to [0,1]
            extra_mcar = max(0, min(extra_mcar, 1))

            if self.output_probs:
                # Update probabilities for candidate columns
                for i in range(n):
                    for j_idx, j in enumerate(idxs_na):
                        if cond[i, j_idx]:
                            # Probability of being observed (1 - probability of being missing)
                            # Missing if selected for quantile-based mechanism or extra MCAR
                            # P(observed) = P(not missing in quantile) AND P(not missing in MCAR)
                            #             = (1 - p_q) * (1 - extra_mcar)
                            probs_[i, j] = (1 - p_q) * (1 - extra_mcar)
                        else:
                            # Only affected by extra MCAR
                            probs_[i, j] = 1 - extra_mcar

                # Update probabilities for non-candidate columns (only affected by extra MCAR)
                non_candidate_cols = np.setdiff1d(all_idx, idxs_na)
                probs_[:, non_candidate_cols] = 1 - extra_mcar

                return probs_

            # Apply the quantile (MNAR) masking on the candidate columns.
            ber_q = torch.rand(n, d_na, device=X.device)
            mask[:, idxs_na] = (ber_q < p_q) & cond

            # Now apply extra MCAR masking across every entry of X with probability extra_mcar.
            prob = torch.full((n, d), 1 - extra_mcar, device=X.device)
            mcar_mask = torch.bernoulli(prob).bool()
            # The union: an entry is missing if it is masked by the quantile mechanism OR by the extra MCAR.
            mask = mask | (~mcar_mask)
        else:
            # When mcar_extra is False, the only contribution is from the quantile mechanism.
            # Overall missing = r * (adjusted_p_q * p_cond). To hit target_missing_rate, set:
            adjusted_p_q = target_missing_rate / (r * p_cond) if (r * p_cond) > 0 else 0
            adjusted_p_q = max(0, min(adjusted_p_q, 1))

            if self.output_probs:
                # Update probabilities only for candidate columns that meet the condition
                for i in range(n):
                    for j_idx, j in enumerate(idxs_na):
                        if cond[i, j_idx]:
                            # Probability of being observed = 1 - probability of being missing
                            probs_[i, j] = 1 - adjusted_p_q

                return probs_

            ber_q = torch.rand(n, d_na, device=X.device)
            mask[:, idxs_na] = (ber_q < adjusted_p_q) & cond

        # Return the mask so that True means observed and False means missing.
        return ~mask

    def generate_mnar_logistic(self, data, missing_rate, proportion_params, exclude_inputs=True):
        """
        MNAR with a logistic model (two-set mechanism). The idea is:
          - A subset of columns (determined by proportion_params) is used as input for a logistic regression.
          - The remaining columns’ missingness is determined by that model.
          - Additionally, if exclude_inputs is True, the input columns themselves are masked MCAR.

        This method uses pick_coeffs and fit_intercepts so that the average missing rate is near missing_rate.

        :param data: PyG data object with attribute x.
        :param missing_rate: Target proportion of missing values.
        :param proportion_params: Fraction (in [0,1]) of features used as logistic model inputs.
        :param exclude_inputs: If True, the input columns (used for the logistic model) are masked using MCAR (with missing_rate).
        :return: Boolean mask (n x d) where True means observed.
        """
        assert 0.0 <= missing_rate <= 1.0, "'missing_rate' must be in [0,1]"
        assert 0.0 <= proportion_params <= 1.0, "'proportion_params' must be in [0,1]"
        X = data.x.float()
        n, d = X.shape
        mask = torch.ones(n, d, dtype=torch.bool, device=X.device)

        if self.output_probs:
            probs_ = torch.ones(n, d, dtype=torch.float32, device=X.device)

        if exclude_inputs:
            d_params = max(int(proportion_params * d), 1)
            all_idx = np.arange(d)
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(all_idx)
            idxs_params = all_idx[:d_params]
            idxs_nas = np.array([i for i in range(d) if i not in idxs_params])

            # Compute coefficients and intercepts for the logistic model using idxs_params as inputs.
            coeffs = self.pick_coeffs(X, idxs_obs=idxs_params, idxs_nas=idxs_nas, self_mask=False)
            # Fit an intercept per column in the output set.
            intercepts = self.fit_intercepts(X[:, idxs_params], coeffs, missing_rate, self_mask=False)
            # Compute missing probabilities for idxs_nas:
            ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

            if self.output_probs:
                # Update probabilities for non-input columns
                for i, idx in enumerate(idxs_nas):
                    probs_[:, idx] = 1 - ps[:, i]  # Probability of being observed

                # Update probabilities for input columns (MCAR)
                probs_[:, idxs_params] = 1 - missing_rate

                return probs_

            # For each row and each column in idxs_nas, mark as missing if a random draw is below p.
            ber = torch.rand(n, len(idxs_nas), device=X.device)
            mnar_nas = ber >= ps
            mask[:, idxs_nas] = mnar_nas  # these entries become missing when False.

            # Now, for the predictors (input columns), apply a MCAR mechanism.
            mcar_for_params = torch.bernoulli(torch.full((n, d_params), 1-missing_rate, device=X.device)).bool()
            mask[:, idxs_params] = mcar_for_params
        else:
            # Use all features as both inputs and targets (i.e. self-masking but under the logistic framework).
            d_params = d
            d_na = d
            idxs_params = np.arange(d)
            idxs_nas = np.arange(d)
            coeffs = self.pick_coeffs(X, idxs_obs=idxs_params, idxs_nas=idxs_nas, self_mask=False)  # here coeffs shape will be (d, d)
            intercepts = self.fit_intercepts(X[:, idxs_params], coeffs, missing_rate, self_mask=False)
            ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

            if self.output_probs:
                probs_ = 1 - ps  # Probability of being observed
                return probs_

            ber = torch.rand(n, d_na, device=X.device)
            mask[:, idxs_nas] = ber >= ps  # observed if random value is above probability of missing.

        return mask

    def generate_mnar_selfmasked(self, data, missing_rate):
        """
        MNAR with a self-masking logistic model. Here each entry's probability of being missing
        is given by a logistic function applied solely to its own value.
        The intercept is selected (via bisection) to target missing_rate.

        :param data: PyG data object with attribute x.
        :param missing_rate: Target missing rate.
        :return: Boolean mask (n x d) with True indicating an observed value.
        """
        assert 0.0 <= missing_rate <= 1.0, "'missing_rate' must be in [0,1]"
        X = data.x.float()
        n, d = X.shape
        # Pick a coefficient per feature.
        coeffs = self.pick_coeffs(X, self_mask=True)
        intercepts = self.fit_intercepts(X, coeffs, missing_rate, self_mask=True)
        # For each entry: probability of missing is sigmoid(x * coeff + intercept)
        ps = torch.sigmoid(X * coeffs + intercepts)

        if self.output_probs:
            # Return probability of being observed
            return 1 - ps

        ber = torch.rand(n, d, device=X.device)
        mask = ber >= ps   # observed if random draw is high enough
        return mask

    def generate(self, data, mechanism, missing_rate,
                 proportion_observed=None,  # for MAR
                 option=None,              # for MNAR: 'quantile', 'logistic', 'selfmasked'
                 q=None,                   # for MNAR quantile
                 p_params=None,            # for MNAR quantile or logistic (proportion of features to use)
                 cut=None,                 # for MNAR quantile: 'lower', 'upper', or 'both'
                 mcar_extra=False,         # for MNAR quantile
                 exclude_inputs=None):     # for MNAR logistic
        """
        Master method that calls one of the dedicated methods based on the mechanism.

        For MCAR and MAR, missing_rate is used.

        For MNAR, missing_rate is always required along with additional parameters depending on the chosen option:
          - For option 'quantile':
              requires q (quantile level), p_params (fraction of columns to affect), cut (default 'both', choices: 'lower', 'upper', 'both').
          - For option 'logistic':
              requires p_params (fraction used as logistic inputs) and exclude_inputs (bool, default True).
          - For option 'selfmasked': only missing_rate is needed.

        :param data: PyG data object with attribute x.
        :param mechanism: string among 'MCAR', 'MAR', or 'MNAR' (case-insensitive).
        :param missing_rate: float in [0,1]
        :param proportion_observed: for MAR; fraction of features that remain fully observed.
        :param option: for MNAR; one of ['quantile', 'logistic', 'selfmasked'].
        :param q: for MNAR quantile; quantile threshold.
        :param p_params: for MNAR quantile/logistic; fraction of features for the logistic model / quantile selection.
        :param cut: for MNAR quantile; one of 'lower', 'upper', 'both'. Default is 'both'.
        :param exclude_inputs: for MNAR logistic; whether to exclude the logistic inputs from the masking. Default True.
        :return: A Boolean mask (n x d) (True = observed, False = missing)  or probability tensor if output_probs=True
        """
        mech = mechanism.lower()
        if mech == "mcar":
            result = self.generate_mcar(data, missing_rate)
        elif mech == "mar":
            assert proportion_observed is not None, "For MAR, please specify 'proportion_observed'."
            result = self.generate_mar(data, missing_rate, proportion_observed)
        elif mech == "mnar":
            assert option is not None, "For MNAR, please provide an 'option': 'quantile', 'logistic', or 'selfmasked'."
            opt = option.lower()
            if opt == "quantile":
                assert q is not None, "For MNAR quantile, specify a quantile level 'q'."
                assert p_params is not None, "For MNAR quantile, specify 'p_params' (fraction of variables to affect)."
                cut = cut if cut is not None else 'both'
                result = self.generate_mnar_quantile(data, missing_rate, q, p_params, cut, mcar_extra)
            elif opt == "logistic":
                assert p_params is not None, "For MNAR logistic, specify 'p_params' (fraction of logistic inputs)."
                exclude_inputs = exclude_inputs if exclude_inputs is not None else True
                result = self.generate_mnar_logistic(data, missing_rate, p_params, exclude_inputs)
            elif opt == "selfmasked":
                result = self.generate_mnar_selfmasked(data, missing_rate)
            else:
                raise ValueError("Unknown MNAR option. Choose 'quantile', 'logistic', or 'selfmasked'.")
        else:
            raise ValueError("Mechanism not recognized. Choose 'MCAR', 'MAR', or 'MNAR'.")

        if not self.output_probs:
            missing_prop = (result == False).sum().item() / result.numel()
            if missing_rate > 0 and abs(missing_prop - missing_rate)/missing_rate > 0.1:
                print(f"\033[91mWARNING:", end="")
                for name, value in locals().items():
                    if name in ["mechanism", "missing_rate", "proportion_observed", "option", "q", "p_params", "cut", "mcar_extra", "exclude_inputs"]:
                        if value is not None:
                            print(f" {name}={value}", end="")
                print(f"\n\tActual missing proportion ({missing_prop:.2f}) differs significantly from target ({missing_rate:.2f}).\033[0m")

        return result


class GraphMissingMaskGenerator:
    def __init__(self, seed=0, output_probs=False):
        """
        Initialize the graph-aware missing mask generator.

        Parameters:
        :param seed: Optional random seed to ensure determinism.
        :param output_probs: If True, the generate method returns the probability tensor
                             instead of the boolean mask.
        """
        self.seed = seed
        self.output_probs = output_probs
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _calibrate_probs(self, p_missing, target_missing_rate, eps=1e-8):
        """
        Scale the raw missing probability tensor so that its mean approximates the target_missing_rate,
        while preserving columns that are fully observed (p_missing = 0).

        Parameters:
        - p_missing (torch.Tensor): Raw missing probabilities (of shape n x d).
        - target_missing_rate (float): Desired overall missing proportion.

        Returns:
        - p_adj (torch.Tensor): Calibrated missing probabilities.
        """
        # Create a mask for columns that are fully observed (all values are 0)
        n, d = p_missing.shape
        fully_observed_columns = (p_missing.sum(dim=0) == 0)

        # Count number of fully observed features
        num_fully_observed = fully_observed_columns.sum().item()

        # Sanity check: ensure we have enough partially missing features to achieve the target rate
        max_possible_missing_rate = (d - num_fully_observed) / d
        if target_missing_rate > max_possible_missing_rate:
            raise ValueError(
                f"Cannot achieve target missing rate of {target_missing_rate:.2f} with "
                f"{num_fully_observed} fully observed features out of {d} total features. "
                f"Maximum possible missing rate is {max_possible_missing_rate:.2f}."
            )

        # If all columns are fully observed, return the original tensor
        if num_fully_observed == d:
            return p_missing

        # Work with a copy to avoid modifying the original tensor
        p_adj = p_missing.clone()

        # Calculate the adjusted target rate for non-fully-observed columns
        # The fully observed columns contribute 0 to the overall missing rate
        if num_fully_observed < d:
            adjusted_target = target_missing_rate * d / (d - num_fully_observed)
        else:
            adjusted_target = 0  # This case shouldn't happen due to the check above

        # Calculate current mean of non-fully-observed columns
        mask_non_observed = ~fully_observed_columns.unsqueeze(0).expand(n, -1)
        current_mean = p_missing[mask_non_observed].mean().item() if mask_non_observed.any() else 0

        # Scale only the non-fully-observed columns
        if current_mean > eps:  # Avoid division by zero
            scaling = adjusted_target / current_mean
            p_adj[mask_non_observed] = torch.clamp(p_adj[mask_non_observed] * scaling, max=1.0)

        return p_adj


    def generate(self, data, mechanism, scenario, missing_rate, aggregator_prob_funct, **kwargs):
        """
        Generate a missingness mask for the node features of a graph data.

        Parameters:
        - data (torch_geometric.data.Data): The graph data. Must have attribute .x (n x d tensor).
        - mechanism (str): The missingness mechanism to use. One of "MCAR", "MAR", "MNAR" (case-insensitive).
        - scenario (int or None): For MCAR, set scenario to None; For MAR, one of [1,2,3,4,5];
                                  For MNAR, one of [1,2,3,4]. This parameter may be used to inform the aggregator.
        - missing_rate (float): The target proportion of missing entries to generate. In [0,1].
        - aggregator_prob_funct (callable): A user-provided function that, given the graph data, returns a
                                             torch.Tensor of probabilities (shape matching data.x) in [0,1].
                                             This corresponds to the function g in our LaTeX formulation.

        Returns:
        - If output_probs=False: A boolean mask (n x d) where True indicates the entry is observed
                                and False indicates a missing entry.
        - If output_probs=True: A float tensor (n x d) containing the probability of each entry being observed.
        """
        # Basic checks
        assert 0.0 <= missing_rate <= 1.0, "'missing_rate' must be in [0,1]"
        mech = mechanism.lower()
        n, d = data.x.shape

        if mech == "mcar":
            # For MCAR, we expect that missingness is independent from both features and graph structure.
            # In this case, the aggregator function is not used and scenario should be None.
            assert scenario is None, "For MCAR, scenario should be set to None."

            # Create observation probabilities (1 - missing_rate for all entries)
            obs_probs = torch.full((n, d), 1 - missing_rate, device=data.x.device, dtype=torch.float32)

            if self.output_probs:
                return obs_probs

            # Otherwise, generate a mask by drawing Bernoulli trials
            mask = torch.bernoulli(obs_probs).bool()

        elif mech in ["mar", "mnar"]:
            # For graph-aware MAR and MNAR, we require that the scenario is given.
            # Check for allowed scenarios.
            if mech == "mar":
                assert scenario in [1, 2, 3, 4, 5], "For MAR, scenario must be one of the integers: 1,2,3,4,5."
                if scenario in [1, 3, 4, 5]:
                    assert "prop_obs" in kwargs.keys(), "For MAR, we need to define a proportion of observed features."
            else:  # mech == "mnar"
                assert scenario in [1, 2, 3, 4], "For MNAR, scenario must be one of the integers: 1,2,3,4."

            # The aggregator function extracts relevant information and returns missing probabilities
            assert callable(aggregator_prob_funct), "aggregator_prob_funct must be callable."
            p_missing = aggregator_prob_funct(data, **kwargs)

            # Check that the output has the same shape as the feature matrix
            assert p_missing.shape == data.x.shape, "aggregator_prob_funct must return a tensor of shape matching data.x."

            # Ensure probabilities are in [0, 1]
            if not (torch.all(p_missing >= 0) and torch.all(p_missing <= 1)):
                raise ValueError("The output of aggregator_prob_funct should be in [0,1].")

            # Calibrate the probabilities so that the overall missing rate is as desired
            p_missing_calib = self._calibrate_probs(p_missing, missing_rate)


            # Generate mask by comparing random draws to calibrated missing probabilities
            rand_tensor = torch.rand(n, d, device=data.x.device)
            mask = (rand_tensor >= p_missing_calib)
        else:
            raise ValueError("Mechanism not recognized. Please choose one of 'MCAR', 'MAR', or 'MNAR'.")

        actual_missing_prop = (~mask).float().mean().item()
        if missing_rate > 0 and abs(actual_missing_prop - missing_rate) / missing_rate > 0.1:
            print(f"\033[91mWARNING: Actual missing proportion ({actual_missing_prop:.2f}) differs from target ({missing_rate:.2f}).\033[0m")

        if self.output_probs :
            # Calculate observation probabilities (1 - missing probabilities)
            obs_probs = 1 - p_missing_calib
            return obs_probs

        return mask



# -----------------------------------------------------------------------------------------------------------------
#                                ▗▄▖  ▗▄▄▖ ▗▄▄▖▗▄▄▖ ▗▄▄▄▖ ▗▄▄▖ ▗▄▖▗▄▄▄▖▗▄▖ ▗▄▄▖  ▗▄▄▖
#                               ▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌▐▌   ▐▌   ▐▌ ▐▌ █ ▐▌ ▐▌▐▌ ▐▌▐▌
#                               ▐▛▀▜▌▐▌▝▜▌▐▌▝▜▌▐▛▀▚▖▐▛▀▀▘▐▌▝▜▌▐▛▀▜▌ █ ▐▌ ▐▌▐▛▀▚▖ ▝▀▚▖
#                               ▐▌ ▐▌▝▚▄▞▘▝▚▄▞▘▐▌ ▐▌▐▙▄▄▖▝▚▄▞▘▐▌ ▐▌ █ ▝▚▄▞▘▐▌ ▐▌▗▄▄▞▘
# -----------------------------------------------------------------------------------------------------------------

def aggregator_mar_scenario2(data, spread=5, center=0.5, **kwargs):
    """
    MAR Scenario 2: Missingness based solely on the structural property of degree.

    For each node v_i, we compute:
      d_i = degree(v_i)
      \tilde{d}_i = d_i / max_k d_k,
    and then for every feature j:
      p_{ij} = sigma(spread * (\tilde{d}_i - 0.5)),
    with spread set to 5.

    This function returns a tensor of shape (n x d) with the same probability per row.
    """
    # Ensure data contains edge_index.
    if not hasattr(data, 'edge_index'):
        raise ValueError("Data must include 'edge_index' for structural aggregation.")

    device = data.x.device
    n, d = data.x.shape

    # Calculate degree for each node.
    degrees = torch.zeros(n, device=device)
    # edge_index is assumed to be a (2 x num_edges) tensor: [source, target].
    src, _ = data.edge_index
    degrees.index_add_(0, src, torch.ones(src.shape[0], device=device))
    max_deg = torch.max(degrees)
    d_normalized = degrees / (max_deg + 1e-8)  # Avoid division by zero.

    # Apply sigmoid on the shifted degree.
    p = torch.sigmoid(spread * (d_normalized - center))  # shape (n,)
    p = p.unsqueeze(1).expand(n, d)  # same value across all features for each node.
    return p

def aggregator_mar_scenario30(data, prop_obs=0.5, spread1=3.0, spread2=1.0, center=0.5, **kwargs):
    """
    MAR Scenario 3: Missingness based on both the node's own observed features and its neighbors'
    observed features—giving higher weight to neighbors.

    Let:
      - F^(OBS) be a subset of F (column-wise) such that all features in F^(OBS) are fully observed.
      - F^(MIS) = F \ F^(OBS) are the potentially unobserved features.
      - F^(self)_i is the average of node v_i's observed features (from F^(OBS)).
      - F^(nbr)_i is the average of the observed features of the neighbors of v_i (computed from F^(OBS)).

    Then, for each node v_i and each feature j the missingness probability is defined as:
      p_ij = sigma( spread1 * (F^(nbr)_i - center) + spread2 * (F^(self)_i - center) )
    with, by default, spread1 = 3.0, spread2 = 1.0, and center = 0.5. The same probability is
    applied across all features for a given node.

    Parameters:
      data (torch_geometric.data.Data): Graph data object with attributes `x` (node features) and `edge_index`.
      prop_obs (float): Proportion of features (columns) to consider as fully observed.
      spread1 (float): Weight multiplying the neighbors' average.
      spread2 (float): Weight multiplying the node's own average.
      center (float): Centering constant subtracted from the averages.
      **kwargs: Other keyword arguments (if needed).

    Returns:
      torch.Tensor: A tensor of missingness probabilities of shape (n x d) (same probability for all features of a node).
    """
    # Check that we have edge_index.
    if not hasattr(data, 'edge_index'):
        raise ValueError("Data must include 'edge_index' for aggregator_mar_scenario3.")

    device = data.x.device
    n, d = data.x.shape

    # Select a subset of columns to act as fully observed features.
    num_obs = int(round(prop_obs * d))
    if num_obs < 1 or num_obs > d:
        raise ValueError("The computed number of observed features must be between 1 and the total number of features.")

    # For reproducibility, one might fix a permutation (here we use torch.randperm).
    obs_indices = torch.randperm(d)[:num_obs]

    # Compute the node's own average over the observed features.
    F_self = data.x[:, obs_indices].float().mean(dim=1)  # shape: (n,)

    # Compute the neighbors' average (using F_self) for each node.
    F_neighbors = torch.zeros(n, device=device)
    counts = torch.zeros(n, device=device) + 1e-8

    src, dst = data.edge_index  # expected shape (2, num_edges)
    for s, d_idx in zip(src, dst):
        F_neighbors[d_idx] += F_self[s]
        counts[d_idx] += 1
    # In case a node has no neighbors, use its own value.
    F_neighbors = torch.where(counts > 0, F_neighbors / (counts), F_self)
    # raise warning if a node is found to have no neighbors
    if any(counts == 0):
        # get the index of the nodes with no neighbors
        no_neighbors = torch.where(counts == 0)[0]
        # print the number of nodes with no neighbors
        print(f"\033[91mWARNING: {len(no_neighbors)} nodes found to have no neighbors.\033[0m")

    # Compute the missingness probability for each node.
    # The formula:
    # p_ij = sigma( spread1*(F^(nbr)_i - center) + spread2*(F^(self)_i - center) )
    logits = spread1 * (F_neighbors - center) + spread2 * (F_self - center)
    p_node = torch.sigmoid(logits)  # shape: (n,)

    # Expand to shape (n x d) so that the same probability is applied to all features for a given node.
    p = p_node.unsqueeze(1).expand(n, d).clone()

    # Set observed feature with prob = 0 (=observed)
    p[:, obs_indices] = 0

    return p


def aggregator_mar_scenario310(data, prop_obs=0.5, spread1=1.0, spread2=0.0, center=0.5, **kwargs):
    """
    MAR Scenario 3: Missingness based on both the node's own observed features and its neighbors'
    observed features—giving higher weight to neighbors.

    Let:
      - F^(OBS) be a subset of F (column-wise) such that all features in F^(OBS) are fully observed.
      - F^(MIS) = F \ F^(OBS) are the potentially unobserved features.
      - F^(self)_i is the average of node v_i's observed features (from F^(OBS)).
      - F^(nbr)_i is the average of the observed features of the neighbors of v_i (computed from F^(OBS)).

    Then, for each node v_i and each feature j the missingness probability is defined as:
      p_ij = sigma( spread1 * (F^(nbr)_i - center) + spread2 * (F^(self)_i - center) )
    with, by default, spread1 = 3.0, spread2 = 1.0, and center = 0.5. The same probability is
    applied across all features for a given node.

    Parameters:
      data (torch_geometric.data.Data): Graph data object with attributes `x` (node features) and `edge_index`.
      prop_obs (float): Proportion of features (columns) to consider as fully observed.
      spread1 (float): Weight multiplying the neighbors' average.
      spread2 (float): Weight multiplying the node's own average.
      center (float): Centering constant subtracted from the averages.
      **kwargs: Other keyword arguments (if needed).

    Returns:
      torch.Tensor: A tensor of missingness probabilities of shape (n x d) (same probability for all features of a node).
    """
    # Check that we have edge_index.
    if not hasattr(data, 'edge_index'):
        raise ValueError("Data must include 'edge_index' for aggregator_mar_scenario3.")

    device = data.x.device
    n, d = data.x.shape

    # Select a subset of columns to act as fully observed features.
    num_obs = int(round(prop_obs * d))
    if num_obs < 1 or num_obs > d:
        raise ValueError("The computed number of observed features must be between 1 and the total number of features.")

    # For reproducibility, one might fix a permutation (here we use torch.randperm).
    obs_indices = torch.randperm(d)[:num_obs]

    # Compute the node's own average over the observed features.
    F_self = data.x[:, obs_indices].float().mean(dim=1)  # shape: (n,)

    # Compute the neighbors' average (using F_self) for each node.
    F_neighbors = torch.zeros(n, device=device)
    counts = torch.zeros(n, device=device) + 1e-8

    src, dst = data.edge_index  # expected shape (2, num_edges)
    for s, d_idx in zip(src, dst):
        F_neighbors[d_idx] += F_self[s]
        counts[d_idx] += 1
    # In case a node has no neighbors, use its own value.
    F_neighbors = torch.where(counts > 0, F_neighbors / (counts), F_self)
    # raise warning if a node is found to have no neighbors
    if any(counts == 0):
        # get the index of the nodes with no neighbors
        no_neighbors = torch.where(counts == 0)[0]
        # print the number of nodes with no neighbors
        print(f"\033[91mWARNING: {len(no_neighbors)} nodes found to have no neighbors.\033[0m")

    # Compute the missingness probability for each node.
    # The formula:
    # p_ij = sigma( spread1*(F^(nbr)_i - center) + spread2*(F^(self)_i - center) )
    logits = spread1 * (F_neighbors - center) + spread2 * (F_self - center)
    p_node = torch.sigmoid(logits)  # shape: (n,)

    # Expand to shape (n x d) so that the same probability is applied to all features for a given node.
    p = p_node.unsqueeze(1).expand(n, d).clone()

    # Set observed feature with prob = 0 (=observed)
    p[:, obs_indices] = 0

    return p

def aggregator_mnar_scenario3(data, spread1=1.0, spread2=3.0, center=0.5, **kwargs):
    """
    MNAR Scenario 3: Missingness based on unobserved features from the node itself and its neighbors,
    with higher weight on the neighbors.

    For each node v_i and each feature j:
      p_{ij} = sigma( spread1*(F_{ij} - center) + spread2*(mean_{v_k in N(v_i)} F_{kj} - center) ),
    with spread1 = 1 and spread2 = 3.

    The output is a tensor of probabilities for each (i,j) entry.
    """
    if not hasattr(data, 'edge_index'):
        raise ValueError("Data must contain 'edge_index' for aggregator_mnar_scenario3.")

    device = data.x.device
    n, d = data.x.shape
    F_neighbors = torch.zeros(n, d, device=device)

    src, dst = data.edge_index  # shape: (2, num_edges)
    counts = torch.zeros(n, d, device=device) + 1e-8  # add epsilon to avoid division by zero
    # We'll iterate over the edges, accumulating per-feature sums.
    for i in range(src.shape[0]):
        F_neighbors[dst[i]] += data.x[src[i]].float()
        counts[dst[i]] += 1
    # In case a node has no neighbors, use 0.
    F_neighbors = torch.where(counts > 0, F_neighbors / (counts), center)
    # raise warning if a node is found to have no neighbors
    if any(counts[:,0] == 0):
        # get the index of the nodes with no neighbors
        no_neighbors = torch.where(counts == 0)[0]
        # print the number of nodes with no neighbors
        print(f"\033[91mWARNING: {len(no_neighbors)} nodes found to have no neighbors.\033[0m")

    p = torch.sigmoid(spread1 * (data.x - center) + spread2 * (F_neighbors - center))
    # p has shape (n, d) with a distinct value for each feature.
    return p


def aggregator_mnar_scenario31(data, spread1=0.0, spread2=1.0, center=0.5, **kwargs):
    """
    MNAR Scenario 3: Missingness based on unobserved features from the node itself and its neighbors,
    with higher weight on the neighbors.

    For each node v_i and each feature j:
      p_{ij} = sigma( spread1*(F_{ij} - center) + spread2*(mean_{v_k in N(v_i)} F_{kj} - center) ),
    with spread1 = 1 and spread2 = 3.

    The output is a tensor of probabilities for each (i,j) entry.
    """
    if not hasattr(data, 'edge_index'):
        raise ValueError("Data must contain 'edge_index' for aggregator_mnar_scenario3.")

    device = data.x.device
    n, d = data.x.shape
    F_neighbors = torch.zeros(n, d, device=device)

    src, dst = data.edge_index  # shape: (2, num_edges)
    counts = torch.zeros(n, d, device=device) + 1e-8  # add epsilon to avoid division by zero
    # We'll iterate over the edges, accumulating per-feature sums.
    for i in range(src.shape[0]):
        F_neighbors[dst[i]] += data.x[src[i]].float()
        counts[dst[i]] += 1
    # In case a node has no neighbors, use 0.
    F_neighbors = torch.where(counts > 0, F_neighbors / (counts), center)
    # raise warning if a node is found to have no neighbors
    if any(counts[:,0] == 0):
        # get the index of the nodes with no neighbors
        no_neighbors = torch.where(counts == 0)[0]
        # print the number of nodes with no neighbors
        print(f"\033[91mWARNING: {len(no_neighbors)} nodes found to have no neighbors.\033[0m")

    p = torch.sigmoid(spread1 * (data.x - center) + spread2 * (F_neighbors - center))
    # p has shape (n, d) with a distinct value for each feature.
    return p

def aggregator_mnar_scenario4(data, spread1=1.0, spread2=1.0, spread3=1.0, center=0.5, **kwargs):
    """
    MNAR Scenario 4: Missingness based on a balanced combination of the node's own feature, its
    structural degree, and the average of its neighbors' features.

    For each node v_i and each feature j:
      Let d_i be the node's degree and let
         \tilde{d}_i = d_i / max_k d_k.
      Let \overline{F}^{(nbr)}_{ij} be the average value of feature j over neighbors of v_i.

      Then, set:
         p_{ij} = sigma( spread1*(F_{ij} - center) + spread2*(\tilde{d}_i - center) + spread3*(\overline{F}^{(nbr)}_{ij} - center) ),
      with spread1 = spread2 = spread3 = 1.
    """
    if not hasattr(data, 'edge_index'):
        raise ValueError("Data must contain 'edge_index' for aggregator_mnar_scenario4.")

    device = data.x.device
    n, d = data.x.shape

    # Compute the normalized degree for each node.
    degrees = torch.zeros(n, device=device)
    src, _ = data.edge_index
    degrees.index_add_(0, src, torch.ones(src.shape[0], device=device))
    max_deg = torch.max(degrees)
    d_norm = degrees / (max_deg + 1e-8)  # shape (n,)
    d_term = d_norm.unsqueeze(1).expand(n, d)  # same for all features of a node

    # Compute the neighbors' average for each node and feature.
    F_neighbors = torch.zeros(n, d, device=device)
    src, dst = data.edge_index
    counts = torch.zeros(n, d, device=device) + 1e-8
    # Accumulate features from neighbors for each (node, feature) pair.
    for i in range(src.shape[0]):
        F_neighbors[dst[i]] += data.x[src[i]].float()
        counts[dst[i]] += 1
    # In case a node has no neighbors, use 0.5
    F_neighbors = torch.where(counts > 0, F_neighbors / (counts), center)
    # raise warning if a node is found to have no neighbors
    if any(counts[:,0] == 0):
        # get the index of the nodes with no neighbors
        no_neighbors = torch.where(counts == 0)[0]
        # print the number of nodes with no neighbors
        print(f"\033[91mWARNING: {len(no_neighbors)} nodes found to have no neighbors.\033[0m")

    p = torch.sigmoid(spread1 * (data.x - center) + spread2 * (d_term - center) + spread3 * (F_neighbors - center))
    return p


# -----------------------------------------------------------------------------------------------------------------
#                               ▗▄▄▖  ▗▄▖  ▗▄▄▖▗▄▄▄▖▗▖   ▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖ ▗▄▄▖
#                               ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌   ▐▌     █  ▐▛▚▖▐▌▐▌   ▐▌
#                               ▐▛▀▚▖▐▛▀▜▌ ▝▀▚▖▐▛▀▀▘▐▌     █  ▐▌ ▝▜▌▐▛▀▀▘ ▝▀▚▖
#                               ▐▙▄▞▘▐▌ ▐▌▗▄▄▞▘▐▙▄▄▖▐▙▄▄▖▗▄█▄▖▐▌  ▐▌▐▙▄▄▖▗▄▄▞▘
# -----------------------------------------------------------------------------------------------------------------

# ===== TABULAR BASELINES

def impute_column_average(data, mask, seed=0, **kwargs):
    """
    Tabular imputation using column (feature) average.
    data.x is a tensor of shape (n, d)
    mask: Boolean tensor of shape (n, d) where True indicates observed entries.
    For missing entries (mask==False), impute the average of observed entries (per column).
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_default_dtype(torch.double)

    X = data.x.clone().double()  # original features
    observed = mask  # True means observed, False means missing
    n, d = X.shape
    # Compute column-wise average of observed values
    col_avg = torch.zeros(d, device=X.device)
    for j in range(d):
        observed_j = X[observed[:, j], j]
        if observed_j.numel() > 0:
            col_avg[j] = observed_j.mean()
        else:
            col_avg[j] = 0.0
    # Impute missing entries
    imputed = X.clone()
    for j in range(d):
        missing_idx = (~observed[:, j]).nonzero(as_tuple=True)[0]
        if missing_idx.numel() > 0:
            imputed[missing_idx, j] = col_avg[j]
    return imputed

def impute_random(data, mask, seed=0, **kwargs):
    """
    Tabular imputation with random values sampled from the observed distribution of each column.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X = data.x.clone().float()  # original features
    observed = mask  # True means observed
    n, d = X.shape
    imputed = X.clone()
    for j in range(d):
        observed_vals = X[observed[:, j], j]
        if observed_vals.numel() > 0:
            # sample randomly from the observed values
            rand_vals = observed_vals[torch.randint(0, observed_vals.numel(), (n,), device=X.device)]
            # Only replace missing entries
            missing_idx = (~observed[:, j]).nonzero(as_tuple=True)[0]
            imputed[missing_idx, j] = rand_vals[missing_idx % rand_vals.shape[0]]
        else:
            imputed[:, j] = 0.0  # fallback if no observed values
    return imputed

# ----- MICE - VAN BUUREN 2018
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute_mice(data, mask, verbose=False, seed=0, disable_tqdm=True, **kwargs):
    """
    Imputes missing values using MICE (Multiple Imputation by Chained Equations).

    Args:
        data: A torch_geometric data object. Only the 'x' attribute (features) is used.
        mask: A boolean tensor of the same shape as data.x, where True indicates an observed value.

    Returns:
        imputed: A torch.Tensor of the same shape as data.x with imputed values.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X_original = data.x.clone().numpy()  # Convert to NumPy array
    X_masked = np.where(mask.numpy(), X_original, np.nan)  # Apply the mask

    if verbose:
        t_start = time.time()
        print(f"Strat Iterative Imputer at {time.strftime('%H:%M:%S', time.localtime())}...")
    imputer = IterativeImputer(random_state=seed)  # we can adjust parameters if needed
    X_imputed = imputer.fit_transform(X_masked)
    if verbose:
        print(f"Imputation completed in {time.time()-t_start:.2f} seconds at {time.strftime('%H:%M:%S', time.localtime())}.")

    imputed = torch.tensor(X_imputed, dtype=data.x.dtype, device=data.x.device)
    return imputed


# ===== GRAPH IMPUTER

def impute_neighbor_average(data, mask, seed=0, **kwargs):
    """
    Graph imputation using a 1-hop neighbors average.
    For each missing entry on node v, estimate it by the average for the same feature among its neighbors.
    If a node has no neighbors, fallback to column average.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    X = data.x.clone().float()
    observed = mask  # shape (n, d)
    n, d = X.shape
    # Build the adjacency list from the edge_index
    A = {i: [] for i in range(n)}
    src, dst = data.edge_index
    for s, t in zip(src.tolist(), dst.tolist()):
        A[t].append(s)

    imputed = X.clone()
    # First, compute column averages (for fallback)
    col_avg = torch.zeros(d, device=X.device)
    for j in range(d):
        obs_j = X[observed[:, j], j]
        if obs_j.numel() > 0:
            col_avg[j] = obs_j.mean()
        else:
            col_avg[j] = 0.0

    for i in range(n):
        for j in range(d):
            if not observed[i, j]:
                # get neighbors for node i
                neigh_idx = A[i]
                if len(neigh_idx) > 0:
                    # average over neighbor values (only considering observed entries for feature j)
                    neigh_vals = []
                    for ni in neigh_idx:
                        # only use neighbor value if it is observed there
                        if observed[ni, j]:
                            neigh_vals.append(X[ni, j])
                    if len(neigh_vals) > 0:
                        imputed[i, j] = torch.stack(neigh_vals).mean()
                    else:
                        imputed[i, j] = col_avg[j]
                else:
                    imputed[i, j] = col_avg[j]
    return imputed



# ----- FEATURE PROPAGATION - ROSSI et AL. 2022

"""
Feature Propagation Imputer - Based of the article by Rossi et al. - 2022

The below functions provide a single-function interface for imputation through
feature propagation on attributed graphs. It is entirely based on the code shared
at https://github.com/twitter-research/feature-propagation
This one file functions runs code originally spread over multiple files 
(e.g. feature_propagation.py, filling_strategies.py, and utils.py).

The impute_feature_propagation function takes as input a torch_geometric.data object
(with attributes:
    - x: feature matrix (n_nodes x n_features)
    - edge_index: edge connectivity (2 x num_edges)
)
and a boolean mask (of same shape as x, where True indicates that the value is observed)
and returns an imputed feature matrix computed via diffusion on the graph.
"""

def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.

    Args:
        edge_index (torch.Tensor): Tensor of shape (2, num_edges)
        n_nodes (int): number of nodes

    Returns:
        edge_index (torch.Tensor): same as input.
        DAD (torch.Tensor): edge weights computed as above.
    """
    # All edges have weight 1
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    # Compute degree per node
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    # Compute D^{-1/2} (set inf values to 0)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    # Compute normalized weights: deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, DAD

class FeaturePropagation(torch.nn.Module):
    """
    Feature Propagation Module.

    The propagation model diffuses the features over the graph for a fixed
    number of iterations while preserving the observed (non-missing) entries.
    """
    def __init__(self, num_iterations: int):
        """
        Args:
            num_iterations (int): Number of propagation (diffusion) iterations.
        """
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Propagate features over the graph.

        Args:
            x (torch.Tensor): Input feature matrix of shape (n_nodes, n_features).
            edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges).
            mask (torch.Tensor): Boolean tensor (n_nodes, n_features) where True indicates an observed entry.

        Returns:
            out (torch.Tensor): The imputed feature matrix.
        """
        # Initialize the output:
        # Missing entries are set to zero, while observed entries are kept fixed.
        out = torch.zeros_like(x, dtype=x.dtype)
        out[mask] = x[mask]

        n_nodes = x.size(0)
        # Build the propagation (normalized adjacency) matrix.
        adj_edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        # Construct a sparse tensor for the adjacency matrix.
        adj = torch.sparse_coo_tensor(adj_edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

        # Iterate propagation diffusion steps.
        for _ in range(self.num_iterations):
            # Diffuse the features.
            out = torch.sparse.mm(adj, out)
            # Reset known (observed) features.
            out[mask] = x[mask]

        return out

def impute_feature_propagation(data, mask, train_idx=None, val_idx=None, num_iterations=40, seed=0, **kwargs):
    """
    Impute missing features using feature propagation.

    Args:
        data: A torch_geometric.data object containing:
              - x: a feature matrix of shape (n_nodes, n_features)
              - edge_index: a tensor of shape (2, num_edges) representing the graph's connectivity.
        mask: A boolean tensor of shape (n_nodes, n_features) where True indicates the feature is observed.
        num_iterations (int): Number of diffusion iterations (default is 40).

    Returns:
        imputed (torch.Tensor): An imputed feature matrix of the same shape as data.x.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_default_dtype(torch.double)

    # Clone the original features (convert to float if needed).
    X = data.x.clone().double()
    edge_index = data.edge_index
    # Create and run the feature propagation model.
    propagation_model = FeaturePropagation(num_iterations=num_iterations)
    imputed = propagation_model.propagate(x=X, edge_index=edge_index, mask=mask)
    return imputed.detach()

# ----- MissingDataOT - Muzellec et al. 2020

"""
The below functions implement two imputation methods for missing data
using Optimal Transport based methods as in Muzellec et al. (2020).

The code is entirely based of the code repository at:
https://github.com/BorisMuzellec/MissingDataOT

Functions:
    impute_ot_tab(data, mask)
        Implements Algorithm 1 (“one parameter equals one imputed value”).
    
    impute_ot_tab_RR(data, mask)
        Implements Algorithm 3 (the round-robin imputer).

Both functions expect:
    - data: a torch.Tensor of shape (n, d) containing the dataset.
             Missing values should be represented as NaN.
    - mask: a boolean torch.Tensor of the same shape as data (True where value is missing).

Usage example:
    imputed = impute_ot_tab(data, mask)
    imputed_RR = impute_ot_tab_RR(data, mask)
    
Dependencies: torch, numpy, geomloss
"""

class OTimputer():
    """
    One parameter per imputed value imputer using a batched Sinkhorn loss
    corresponding to Algorithm 1 in Muzellec et al. 2020.

    Parameters (with defaults):
      eps: Sinkhorn regularization parameter (default 0.01).
      lr: learning rate (default 1e-2).
      opt: optimizer class (default torch.optim.RMSprop).
      niter: number of gradient updates per imputation (default 15).
      batchsize: batch size for Sinkhorn divergence evaluation (default 128).
      n_pairs: number of pairs batches per gradient update (default 1)
      noise: noise level added for initialization (default 0.1).
      scaling: scaling parameter in Sinkhorn iterations (default 0.9).
    """
    def __init__(self,
                 train_idx=None,
                 val_idx=None,
                 eps=0.01,
                 lr=1e-2,
                 opt=torch.optim.RMSprop,
                 niter=15,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1,
                 scaling=0.9):
        self.train_idx=train_idx
        self.val_idx=val_idx
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")


    @staticmethod
    def nanmean(X, dim=0):
        """Compute the mean over dim ignoring NaNs."""
        # Replace NaNs by 0 and divide by count of non-NaNs
        X_nan = X.clone()
        nan_mask = torch.isnan(X_nan)
        X_nan[nan_mask] = 0.0
        count = (~nan_mask).double().sum(dim=dim)
        # Avoid division by zero
        count[count == 0] = 1.0
        return X_nan.sum(dim=dim) / count

    @staticmethod
    def MAE(X_filled, X_true, mask):
        """Compute the Mean Absolute Error only on missing entries (mask == 1)."""
        # Here mask is assumed to be a double/float tensor with 1 where X is missing.
        mse = torch.abs(X_filled - X_true)
        return mse[mask.bool()].mean()

    @staticmethod
    def RMSE(X_filled, X_true, mask):
        """Compute the Root Mean Squared Error only on missing entries."""
        se = (X_filled - X_true) ** 2
        return torch.sqrt(se[mask.bool()].mean())

    def fit_transform(self, X, verbose=False, report_interval=500, X_true=None):
        """
        Impute missing values using a batched OT loss.

        Parameters:
          X: a torch tensor (n x d) in which missing values are represented as NaN.
          verbose: if True, print progress.
          X_true: ground truth (optional) for printing validation error.

        Returns:
          X_filled: imputed X.
        """
        X = X.clone()
        n, d = X.shape

        # Adjust batch size if too large
        if self.batchsize > n // 2:
            new_bs = 2**int(np.log2(n // 2))
            if verbose:
                print(f"Batchsize ({self.batchsize}) larger than half the dataset size; setting batchsize to {new_bs}.")
            self.batchsize = new_bs

        mask = torch.isnan(X).double()
        # Initialize imputed values: use noise + mean of non-nan for each column.
        init = self.nanmean(X, dim=0)
        imps = (self.noise * torch.randn(mask.shape, dtype=X.dtype, device=X.device) + init.unsqueeze(0))[mask.bool()]
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        if verbose:
            print(f"Starting OTimputer.fit_transform with batchsize = {self.batchsize}, epsilon = {self.eps:.4f}")

        # Set available indices for computing the loss based on train_idx.
        if self.train_idx is not None:
            available_idx = self.train_idx.cpu().numpy()
        else:
            available_idx = np.arange(n)

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)

        for i in range(self.niter):
            # Replace missing entries by current imputed values.
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss = 0.0

            # Evaluate Sinkhorn loss over n_pairs batches.
            for _ in range(self.n_pairs):
                # idx1 = np.random.choice(n, self.batchsize, replace=False)
                # idx2 = np.random.choice(n, self.batchsize, replace=False)
                # Sample indices from available_idx (training set) if provided.
                idx1 = np.random.choice(available_idx, self.batchsize, replace=False)
                idx2 = np.random.choice(available_idx, self.batchsize, replace=False)
                X1 = X_filled[idx1]
                X2 = X_filled[idx2]
                loss = loss + self.sk(X1, X2)

            if torch.isnan(loss) or torch.isinf(loss) and verbose:
                print("Encountered NaN or Inf loss; stopping early.")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if X_true is not None:
                maes[i] = self.MAE(X_filled, X_true, mask).item()
                rmses[i] = self.RMSE(X_filled, X_true, mask).item()

            if verbose and (i % report_interval == 0):
                if X_true is not None:
                    print(f"Iteration {i}:\tLoss: {loss.item() / self.n_pairs:.4f}\tValidation MAE: {maes[i]:.4f}\tRMSE: {rmses[i]:.4f}")
                else:
                    print(f"Iteration {i}:\tLoss: {loss.item() / self.n_pairs:.4f}")

        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps.detach()

        if X_true is not None:
            return X_filled, maes, rmses
        else:
            return X_filled

class RRimputer():
    """
    Round-Robin imputer using a batched Sinkhorn loss (Algorithm 3 in Muzellec et al. 2020).

    The idea is to impute each variable in turn, using models to predict each missing entry
    from the remaining variables.

    Parameters:
      models: iterable (or dictionary) of torch.nn.Module, one per variable.
      eps: Sinkhorn regularization parameter (default 0.01).
      lr: learning rate (default 1e-2).
      opt: optimizer class (default torch.optim.Adam).
      max_iter: maximum number of round-robin cycles (default 10).
      niter: gradient updates per variable per cycle (default 15).
      batchsize: batch size for Sinkhorn evaluation (default 128).
      n_pairs: number of pairs per gradient update (default 10).
      tol: tolerance for convergence (default 1e-3).
      noise: noise level for initialization (default 0.1).
      weight_decay: L2 regularization magnitude (default 1e-5).
      order: order for imputation ("random" or "increasing", default "random").
      unsymmetrize: if True, sample one batch with no missing data in every pair (default True).
      scaling: scaling parameter for Sinkhorn iterations (default 0.9).
    """
    def __init__(self,
                 models,
                 train_vertex_index=None,
                 eps=0.01,
                 lr=1e-2,
                 opt=torch.optim.Adam,
                 max_iter=10,
                 niter=15,
                 batchsize=128,
                 n_pairs=10,
                 tol=1e-3,
                 noise=0.1,
                 weight_decay=1e-5,
                 order='random',
                 unsymmetrize=False,
                 scaling=0.9,
                 disable_tqdm=True):

        self.models = models
        self.train_vertex_index = train_vertex_index
        self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="auto")
        self.lr = lr
        self.opt = opt
        self.max_iter = max_iter
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.tol = tol
        self.noise = noise
        self.weight_decay = weight_decay
        self.order = order
        self.unsymmetrize = unsymmetrize
        self.is_fitted = False
        self.disable_tqdm=disable_tqdm

    @staticmethod
    def nanmean(X, dim=0):
        """Compute the mean over dim ignoring NaNs."""
        # Replace NaNs by 0 and divide by count of non-NaNs
        X_nan = X.clone()
        nan_mask = torch.isnan(X_nan)
        X_nan[nan_mask] = 0.0
        count = (~nan_mask).double().sum(dim=dim)
        # Avoid division by zero
        count[count == 0] = 1.0
        return X_nan.sum(dim=dim) / count

    @staticmethod
    def MAE(X_filled, X_true, mask):
        """Compute the Mean Absolute Error only on missing entries (mask == 1)."""
        # Here mask is assumed to be a double/float tensor with 1 where X is missing.
        mse = torch.abs(X_filled - X_true)
        return mse[mask.bool()].mean()

    @staticmethod
    def RMSE(X_filled, X_true, mask):
        """Compute the Root Mean Squared Error only on missing entries."""
        se = (X_filled - X_true) ** 2
        return torch.sqrt(se[mask.bool()].mean())

    @staticmethod
    def mae_str(val):
        return f"{val:.4f}"

    def fit_transform(self, X, verbose=False, report_interval=1, X_true=None):
        """
        Fit the imputer on incomplete data X and return the imputed X.
        Missing values are expected to be NaNs.
        """
        X = X.clone()
        n, d = X.shape
        mask = torch.isnan(X).double()
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))

        if self.batchsize > n // 2:
            new_bs = 2**int(np.log2(n // 2))
            if verbose:
                print(f"Batchsize larger than half dataset size; setting batchsize to {new_bs}.")
            self.batchsize = new_bs

        # Set initial ordering: we can sort variables by number of missings
        order_ = torch.argsort(mask.sum(0))

        # Create one optimizer per model.
        optimizers = [self.opt(self.models[i].parameters(), lr=self.lr, weight_decay=self.weight_decay)
                      for i in range(d)]

        # Initialize missing entries with noise + column mean.
        init = self.nanmean(X, dim=0)
        imps = (self.noise * torch.randn(mask.shape, dtype=X.dtype, device=X.device) + init.unsqueeze(0))[mask.bool()]
        X[mask.bool()] = imps
        X_filled = X.clone()

        # Pre-compute the indices for loss computation: use train indices if provided.
        if self.train_vertex_index is not None:
            # Ensure train_vertex_index is a numpy array for sampling.
            available_idx = self.train_vertex_index.cpu().numpy()
        else:
            available_idx = np.arange(n)

        if X_true is not None:
            maes = np.zeros(self.max_iter)
            rmses = np.zeros(self.max_iter)

        for i in tqdm(range(self.max_iter), desc="RR - max_iter", leave=False, disable=self.disable_tqdm):
            if self.order == 'random':
                order_ = np.random.choice(d, d, replace=False)
            X_old = X_filled.clone().detach()

            # Round-robin update over each variable.
            for l in range(d):
                j = order_[l].item()
                n_not_miss = (~mask[:, j].bool()).sum().item()
                if n - n_not_miss == 0:
                    continue  # No missing value for this variable.
                # For each variable, do niter gradient updates.
                for k in range(self.niter):
                    loss = 0.0
                    # Detach so gradients do not flow backwards incorrectly.
                    X_filled = X_filled.detach()
                    # Update the j-th variable using the corresponding model.
                    # Use all other columns as input.
                    X_input = X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]
                    # X_input = X_filled[mask[:, j].bool(), :][:, torch.arange(d) != j]
                    pred = self.models[j](X_input).squeeze()
                    X_filled[mask[:, j].bool(), j] = pred

                    for _ in range(self.n_pairs):
                        idx1 = np.random.choice(available_idx, self.batchsize, replace=False)
                        X1 = X_filled[idx1]
                        if self.unsymmetrize:
                            valid_idx = (~mask[:, j].bool()).nonzero(as_tuple=False).view(-1).cpu().numpy()
                            # Optionally intersect valid_idx with available_idx if desired.
                            if len(valid_idx) < self.batchsize:
                                replace_flag = True
                            else:
                                replace_flag = False
                            idx2 = np.random.choice(valid_idx, self.batchsize, replace=replace_flag)
                            X2 = X_filled[idx2]
                        else:
                            idx2 = np.random.choice(available_idx, self.batchsize, replace=False)
                            X2 = X_filled[idx2]
                        loss = loss + self.sk(X1, X2)
                    optimizers[j].zero_grad()
                    loss.backward()
                    optimizers[j].step()

                # Impute with the final parameters for variable j.
                with torch.no_grad():
                    X_input = X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j+1: d]]
                    # X_input = X_filled[mask[:, j].bool(), :][:, torch.arange(d) != j]
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_input).squeeze()

            if X_true is not None:
                maes[i] = self.MAE(X_filled, X_true, mask).item()
                rmses[i] = self.RMSE(X_filled, X_true, mask).item()
            if verbose and (i % report_interval == 0):
                if X_true is not None:
                    print(f"Iteration {i}:\tLoss: {loss.item() / self.n_pairs:.4f}\tValidation MAE: {self.mae_str(maes[i])}\tRMSE: {rmses[i]:.4f}")
                else:
                    print(f"Iteration {i}:\tLoss: {loss.item() / self.n_pairs:.4f}")

            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break

        if i == (self.max_iter - 1) and verbose:
            print("Early stopping criterion not reached.")

        self.is_fitted = True
        if X_true is not None:
            return X_filled, maes, rmses
        else:
            return X_filled

    def transform(self, X, mask, verbose=False, report_interval=1, X_true=None):
        """
        Impute missing values on new data using the fitted models.
        """
        assert self.is_fitted, "The model has not been fitted yet."
        n, d = X.shape
        normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))
        # Fill missing entries initially with global column mean.
        X[mask.bool()] = self.nanmean(X, dim=0)
        X_filled = X.clone()

        for i in range(self.max_iter):
            X_old = X_filled.clone().detach()
            for l in range(d):
                j = l  # here we simply cycle through indices.
                with torch.no_grad():
                    X_input = X_filled[mask[:, j].bool(), :][:, torch.arange(d) != j]
                    X_filled[mask[:, j].bool(), j] = self.models[j](X_input).squeeze()
            if verbose and (i % report_interval == 0) and X_true is not None:
                print(f"Iteration {i}:\tValidation MAE: {self.MAE(X_filled, X_true, mask).item():.4f}\tRMSE: {self.RMSE(X_filled, X_true, mask).item():.4f}")
            if torch.norm(X_filled - X_old, p=np.inf) < normalized_tol:
                break
        if i == (self.max_iter - 1) and verbose:
            print("Early stopping criterion not reached in transform.")
        return X_filled


def impute_ot_tab(data, mask, train_idx=None, val_idx=None, niter=15, batchsize=128, verbose=False, seed=0, **kwargs):
    """
    Impute node features from a torch_geometric data object using the
    Sinkhorn-based OTimputer (Algorithm 1 in Muzellec et al. 2020).

    Parameters:
      data:        A torch_geometric.data.Data object containing node features in data.x.
      mask:        A boolean torch.Tensor with the same shape as data.x where
                   (mask == False) indicates missing entries.
      train_idx:   Indices of the training set.
      val_idx:     Indices of the validation set.
      niter:       Number of gradient updates for the OTimputer (default: 15).
      batchsize:   Batch size for evaluating the Sinkhorn divergence (default: 128).

    Returns:
      imputed:     A torch.Tensor (with the same shape as data.x)
                   containing the imputed features.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_default_dtype(torch.double)
    # Extract the features and convert to float.
    x = data.x.clone().double()

    # In the experimental procedure: (mask == False) indicates missing data.
    # So we convert the missing entries to NaN.
    missing_mask = ~mask  # True where missing.
    x[missing_mask] = float('nan')

    # Instantiate and run the OTimputer
    ot_imputer = OTimputer(train_idx=train_idx, val_idx=val_idx, eps=0.01, lr=1e-2, niter=niter, batchsize=batchsize,
                           n_pairs=1, noise=0.1, scaling=0.9)
    if verbose:
        print("Starting OT imputation (Algorithm 1)...")
    imputed = ot_imputer.fit_transform(x, verbose=verbose)
    if verbose:
        print("OT imputation completed.")

    return imputed.detach()

def impute_ot_tab_RR(data, mask, train_idx=None, val_idx=None, niter_rr=15, max_iter=10, batchsize=128, verbose=False, disable_tqdm=False, seed=0, **kwargs):
    """
    Impute node features from a torch_geometric data object using the
    Round-Robin imputer (Algorithm 3 in Muzellec et al. 2020).

    Parameters:
      data:         A torch_geometric.data.Data object containing node features in data.x.
      mask:         A boolean torch.Tensor with the same shape as data.x where
                    (mask == False) indicates missing entries.
      train_idx:    Indices of the training set.
      val_idx:      Indices of the validation set.
      niter_rr:     Number of gradient updates for each variable in each cycle
                    (default: 15).
      max_iter:     Maximum number of round-robin cycles (default: 10).
      batchsize:    Batch size for evaluating the Sinkhorn divergence (default: 128).

    Returns:
      imputed:      A torch.Tensor (with the same shape as data.x) containing the imputed features.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_default_dtype(torch.double)
    # Extract the features and convert to float.
    x = data.x.clone().double()

    # As above, mark missing entries with NaN:
    missing_mask = ~mask
    x[missing_mask] = float('nan')

    # Get the shape and create one simple model per feature.
    n, d = x.shape
    models = {}
    for i in range(d):
        # Each model predicts one feature using all other features.
        models[i] = torch.nn.Linear(d - 1, 1)

    # Instantiate the RRimputer.
    rr_imputer = RRimputer(models,
                           train_vertex_index=train_idx,
                           eps=0.01,
                           lr=1e-2,
                           opt=torch.optim.Adam,
                           max_iter=max_iter,
                           niter=niter_rr,
                           batchsize=batchsize,
                           n_pairs=10,
                           tol=1e-3,
                           noise=0.1,
                           weight_decay=1e-5,
                           order='random',
                           unsymmetrize=False,
                           scaling=0.9,
                           disable_tqdm=disable_tqdm)
    if verbose:
        print("Starting Round-Robin OT imputation (Algorithm 3)...")
    imputed = rr_imputer.fit_transform(x, verbose=verbose)
    if verbose:
        print("Round-Robin OT imputation completed.")

    return imputed


# ----- PCFI - Um et al. - 2023

"""
PCFI Imputer, based on the article by Um et al. 2023.

The below functions are a consolidation of the original PCFI code for
"Confidence-Based Feature Imputation for Graphs with Partially Known Features"
(Um et al., 2023). 
The entire code below is based on the code at the repository:
https://github.com/daehoum1/pcfi/tree/main
It reuses functions and classes from that repository,
with modifications to focus solely on feature imputation.
 
Usage:
    Given a torch_geometric.data object (with attributes `x` and `edge_index`)
    and a boolean mask (same shape as x, with True for observed values),
    call the function:
    
        imputed = impute_pcfi(data, mask)
        
The returned tensor (`imputed`) is an imputed feature matrix.
"""

class PCFI(torch.nn.Module):
    def __init__(self, num_iterations: int, alpha: float, beta: float, verbose: bool = False, seed: int = 0):
        """
        Args:
            num_iterations (int): Number of diffusion iterations.
            alpha (float): Confidence parameter alpha.
            beta (float): Confidence parameter beta.
            verbose (bool): If True, print debug messages.
        """
        super(PCFI, self).__init__()
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.seed = seed

    def propagate(self, x: Tensor, edge_index: Tensor, mask: Tensor, mask_type: str, edge_weight: Optional[Tensor] = None) -> Tensor:
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        n_nodes = x.shape[0]
        feat_dim = x.shape[1]
        out = x.clone()  # start with initial features

        if mask_type == 'structural':
            # Compute node-to-distance mapping using the first channel of the mask.
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type)
            # Compute the confidence-based normalized adjacency matrix.
            adj_c = self.compute_edge_weight_c(edge_index, f_n2d, n_nodes)
            if mask is not None:
                out = torch.zeros_like(x)
                out[mask] = x[mask]
            if self.verbose:
                print("Starting propagation with structural mask.")
            for i in range(self.num_iterations):
                out = torch.sparse.mm(adj_c, out)
                out[mask] = x[mask]
                if self.verbose:
                    print(f"Iteration {i+1}/{self.num_iterations} complete.")
            # Repeat the structural f_n2d so that it matches the feature dimension.
            f_n2d = f_n2d.repeat(feat_dim, 1)
        else:
            # Per-channel propagation when mask_type is not structural.
            if self.verbose:
                print(f"Starting propagation on {feat_dim} channels.")
            out = torch.zeros_like(x)
            if mask is not None:
                out[mask] = x[mask]
            f_n2d = self.compute_f_n2d(edge_index, mask, mask_type, feat_dim)
            for i in range(feat_dim):
                adj_c = self.compute_edge_weight_c(edge_index, f_n2d[i], n_nodes)
                for j in range(self.num_iterations):
                    # Diffuse features on channel i.
                    out_channel = torch.sparse.mm(adj_c, out[:, i].reshape(-1, 1)).reshape(-1)
                    out[:, i] = out_channel
                    # Reset observed entries.
                    out[mask[:, i], i] = x[mask[:, i], i]
                    if self.verbose and j % 10 == 0:
                        print(f"Channel {i+1}/{feat_dim}, iteration {j+1}/{self.num_iterations}")
        # Adjust the imputed features using inter-channel correlations.
        cor = torch.corrcoef(out.T)
        cor = cor.nan_to_num()
        cor.fill_diagonal_(0)
        f_n2d = f_n2d.to(out.device)
        a_1 = (self.alpha ** f_n2d.T) * (out - torch.mean(out, dim=0))
        a_2 = torch.matmul(a_1, cor)
        out_1 = self.beta * (1 - (self.alpha ** f_n2d.T)) * a_2
        out = out + out_1
        return out

    def compute_f_n2d(self, edge_index: Tensor, feature_mask: Tensor, mask_type: str, feat_dim: Optional[int] = None) -> Tensor:
        n_nodes = feature_mask.shape[0]
        if mask_type == 'structural':
            len_v_0tod_list = []
            f_n2d = torch.zeros(n_nodes, dtype=torch.int, device=edge_index.device)
            v_0 = torch.nonzero(feature_mask[:, 0]).view(-1)
            len_v_0tod_list.append(len(v_0))
            v_0_to_now = v_0.clone()
            f_n2d[v_0] = 0
            d = 1
            while True:
                v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=n_nodes)[0]
                v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                if len(v_d) == 0:
                    break
                f_n2d[v_d] = d
                v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                len_v_0tod_list.append(len(v_d))
                d += 1
            return f_n2d
        else:
            # For non-structural masks, we compute f_n2d per channel.
            if feat_dim is None:
                raise ValueError("feat_dim must be provided for non-structural mask type.")
            f_n2d = torch.zeros((feat_dim, n_nodes), dtype=torch.int, device=edge_index.device)
            if self.verbose:
                print(f"Computing f_n2d for {feat_dim} channels.")
            for i in range(feat_dim):
                v_0 = torch.nonzero(feature_mask[:, i]).view(-1)
                v_0_to_now = v_0.clone()
                f_n2d[i, v_0] = 0
                d = 1
                while True:
                    v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=n_nodes)[0]
                    v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
                    if len(v_d) == 0:
                        break
                    f_n2d[i, v_d] = d
                    v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
                    d += 1
            if self.verbose:
                print("f_n2d computed.")
            return f_n2d

    def compute_edge_weight_c(self, edge_index: Tensor, f_n2d: Tensor, n_nodes: int) -> torch.sparse_coo_tensor:
        # Compute edge weights using the difference in f_n2d values.
        row, col = edge_index[0], edge_index[1]
        d_row = f_n2d[row]
        d_col = f_n2d[col]
        edge_weight_c = (self.alpha ** (d_col - d_row + 1)).to(edge_index.device)
        deg_W = scatter_add(edge_weight_c, row, dim=0, dim_size=f_n2d.shape[0])
        deg_W_inv = deg_W.pow(-1.0)
        deg_W_inv.masked_fill_(deg_W_inv == float("inf"), 0)
        A_Dinv = edge_weight_c * deg_W_inv[row]
        adj = torch.sparse_coo_tensor(edge_index, A_Dinv, size=(n_nodes, n_nodes)).to(edge_index.device)
        return adj

def pcfi(edge_index: Tensor, X: Tensor, feature_mask: Tensor, num_iterations: int, mask_type: str, alpha: float, beta: float, verbose: bool = False, seed=0) -> Tensor:
    """
    Convenience function to instantiate PCFI and perform propagation.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    propagation_model = PCFI(num_iterations=num_iterations, alpha=alpha, beta=beta, verbose=verbose, seed=seed)
    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask, mask_type=mask_type)

# --------------------------------------------------------------------
# Final imputation function
# --------------------------------------------------------------------

def impute_pcfi(data, mask, train_idx=None, val_idx=None, num_iterations: int = 100, mask_type: str = "structural",
                alpha: float = 0.9, beta: float = 1.0, verbose: bool = False, seed=0):
    """
    Impute missing node features using the PCFI algorithm.

    Args:
        data: A torch_geometric.data object with attributes:
              - x: feature matrix (n_nodes x n_features)
              - edge_index: edge connectivity (2 x num_edges) OR
                train_pos_edge_index (e.g., for link prediction settings).
        mask: A boolean tensor of shape (n_nodes, n_features) where True indicates observed features.
        train_idx, val_idx: Indices for training and validation nodes. Unused as no training is performed.
        num_iterations (int): Number of diffusion iterations.
        mask_type (str): Type of missing feature mask ("structural" or other).
        alpha (float): Parameter alpha for PCFI.
        beta (float): Parameter beta for PCFI.
        verbose (bool): If True, prints additional information during imputation.

    Returns:
        imputed: The imputed feature matrix (torch.Tensor) of the same shape as data.x.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_default_dtype(torch.double)
    # Clone original features and ensure floating type.
    X = data.x.clone().double()
    # Mark missing entries as NaN.
    X[~mask] = float("nan")
    # Use train_pos_edge_index if available, otherwise data.edge_index.
    if hasattr(data, "train_pos_edge_index"):
        edge_index = data.train_pos_edge_index
    else:
        edge_index = data.edge_index
    if verbose:
        print("Starting PCFI imputation...")
    imputed = pcfi(edge_index, X, mask, num_iterations, mask_type, alpha, beta, verbose=verbose, seed=seed)
    if verbose:
        print("PCFI imputation complete.")
    return imputed.detach()

# ----- GRIOT - Serrano et al. - 2024

"""
GRIOT Imputer Module

This module implements data imputation using the GRIOT framework (Serrano et al., 2024)
for attributed graphs. The code below reuses, as much as possible, functions and structures
from the original GRIOT implementation at https://github.com/RichardSrn/GRIOT

The file defines all necessary helper functions (e.g. for building the graph from an adjacency matrix,
computing OT distances, etc.) as well as a single imputation function:

    def impute_griot(data, mask):
        ...
        return imputed
"""


class MultiW:
    def __init__(self,
                 alpha=0.5,
                 epsilon=0.1,
                 p=2,
                 scaling=0.9,
                 numItermax=1000,
                 stopThr=1e-9,
                 method="sinkhorn",
                 path=".",
                 plot=False,
                 unique=None,
                 p_unif=True,
                 normalize_F=True,
                 normalize_MF_MC=False,
                 use_geomloss=True,
                 CrossEtpy=False,
                 **kwargs):
        self.alpha = alpha
        self.epsilon = epsilon
        self.p = p
        self.scaling = scaling
        self.numItermax = numItermax
        self.name = "MultiW"
        self.stopThr = stopThr
        self.method = method
        self.path = path
        self.plot = plot
        self.unique = unique
        self.p_unif = p_unif
        self.normalize_F = normalize_F
        self.normalize_MF_MC = normalize_MF_MC
        self.use_geomloss = use_geomloss
        self.CrossEtpy = CrossEtpy
        self.kwargs = kwargs

    @staticmethod
    def cross_entropy_distance(A, B):
        M = torch.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                M[i, j] = F.cross_entropy(A[i], B[j])
        return M

    def __call__(self,
                 C1: torch.Tensor = None,
                 C2: torch.Tensor = None,
                 F1: torch.Tensor = None,
                 F2: torch.Tensor = None,
                 p1: torch.Tensor = None,
                 p2: torch.Tensor = None,
                 i=None,
                 total_iterations=None,
                 **kwargs):

        C1 = C1.clone()
        C2 = C2.clone()
        F1 = F1.clone()
        F2 = F2.clone()

        # get histograms (nodes' weight)
        if p1 is None or self.p_unif:
            p1 = torch.ones(C1.shape[0], dtype=torch.float32)
        if p2 is None or self.p_unif:
            p2 = torch.ones(C2.shape[0], dtype=torch.float32)

        p1 = p1.clone()
        p2 = p2.clone()

        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()

        p1 = p1.to(F1.device)
        p2 = p2.to(F2.device)

        ### GEOMLOSS ###
        if self.use_geomloss:
            # compute pairwise distance between rows of F1 and F2 according to the L2 norm: compute M_F and M_C
            if not self.CrossEtpy:
                M_F_12 = torch.cdist(F1, F2, p=self.p) ** 2 / 2
                M_F_21 = M_F_12.T
                M_F_11 = torch.cdist(F1, F1, p=self.p) ** 2 / 2
                M_F_22 = torch.cdist(F2, F2, p=self.p) ** 2 / 2
            else:  # compute the crossentropy between the vectors
                M_F_12 = self.cross_entropy_distance(F1, F2).to(F1.device)
                M_F_21 = M_F_12.T
                M_F_11 = self.cross_entropy_distance(F1, F1).to(F1.device)
                M_F_22 = self.cross_entropy_distance(F2, F2).to(F1.device)

            M_C_12 = torch.cdist(C1, C2, p=self.p) ** 2 / 2
            M_C_21 = M_C_12.T
            M_C_11 = torch.cdist(C1, C1, p=self.p) ** 2 / 2
            M_C_22 = torch.cdist(C2, C2, p=self.p) ** 2 / 2

            # normalize M_F and M_C if required
            if self.normalize_MF_MC:
                M_F_11 = (M_F_11 - M_F_11.min()) / (M_F_11.max() - M_F_11.min())
                M_C_11 = (M_C_11 - M_C_11.min()) / (M_C_11.max() - M_C_11.min())

                M_F_12 = (M_F_12 - M_F_12.min()) / (M_F_12.max() - M_F_12.min())
                M_C_12 = (M_C_12 - M_C_12.min()) / (M_C_12.max() - M_C_12.min())

                M_F_22 = (M_F_22 - M_F_22.min()) / (M_F_22.max() - M_F_22.min())
                M_C_22 = (M_C_22 - M_C_22.min()) / (M_C_22.max() - M_C_22.min())

            # compute M
            M_12 = (1 - self.alpha) * M_F_12 + (self.alpha) * M_C_12
            M_21 = (1 - self.alpha) * M_F_21 + (self.alpha) * M_C_21
            M_11 = (1 - self.alpha) * M_F_11 + (self.alpha) * M_C_11
            M_22 = (1 - self.alpha) * M_F_22 + (self.alpha) * M_C_22

            # normalize M
            M_max = max(M_11.max(), M_12.max(), M_21.max(), M_22.max())
            M_11 = M_11 / M_max
            M_12 = M_12 / M_max
            M_21 = M_21 / M_max
            M_22 = M_22 / M_max

            diameter, eps, eps_list, _ = scaling_parameters(x=F1.unsqueeze(0),
                                                            y=F2.unsqueeze(0),
                                                            p=2,
                                                            blur=self.epsilon,
                                                            reach=None,
                                                            diameter=None,
                                                            scaling=self.scaling)

            f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
                softmin_tensorized,
                log_weights(p1),
                log_weights(p2),
                M_11.unsqueeze(0),
                M_22.unsqueeze(0),
                M_12.unsqueeze(0),
                M_21.unsqueeze(0),
                eps_list,
                rho=None,
                debias=True,
            )

            w = sinkhorn_cost(
                eps=eps,
                rho=None,
                a=p1.unsqueeze(0),
                b=p2.unsqueeze(0),
                f_aa=f_aa,
                g_bb=g_bb,
                g_ab=g_ab,
                f_ba=f_ba,
                batch=True,
                debias=True,
                potentials=False,
            )
        ### GEOMLOSS END ###

        # Compute the Wasserstein distance between F1 and F2 if not using geomloss or if transport is requested
        else:
            if self.normalize_F:
                F1 = (F1 - F1.mean(dim=0)) / (F1.max(dim=0).values - F1.min(dim=0).values + 1e-5)
                F2 = (F2 - F2.mean(dim=0)) / (F2.max(dim=0).values - F2.min(dim=0).values + 1e-5)
            # normalize C_k
            C1 = (C1 - C1.min()) / (C1.max() - C1.min())
            C2 = (C2 - C2.min()) / (C2.max() - C2.min())

            if not self.CrossEtpy:
                M_F = torch.cdist(F1, F2, p=self.p) ** 2 / 2
            else:
                M_F = self.cross_entropy_distance(F1, F2).to(F1.device)

            M_C = torch.cdist(C1, C2, p=self.p) ** 2 / 2

            if self.normalize_MF_MC:
                M_F = (M_F - M_F.min()) / (M_F.max() - M_F.min())
                M_C = (M_C - M_C.min()) / (M_C.max() - M_C.min())

            M = (1 - self.alpha) * M_F + (self.alpha) * M_C

            w = ot.sinkhorn2(p1, p2, M, reg=self.epsilon)

        del C1, C2
        del F1, F2
        del p1, p2
        if self.use_geomloss:
            del M_F_11, M_F_12, M_F_22, M_C_11, M_C_12, M_C_22, M_C_21, M_F_21, M_11, M_12, M_22, M_21
            del f_aa, g_bb, g_ab, f_ba
            del diameter, eps, eps_list
        else:
            del M_F, M_C, M
        del kwargs
        del i
        del total_iterations

        return w


def get_edge_index(adjacency_matrix):
    if type(adjacency_matrix) == np.ndarray:
        G = nx.from_numpy_array(adjacency_matrix)
    else:
        G = nx.from_numpy_array(adjacency_matrix.detach().numpy())
    # Create edge index from G using scipy
    adj = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def griot(model=None,
          p=None,
          lr=1e-2,
          opt=torch.optim.Adam,
          max_iters=5,
          n_iters=1,
          batch_size=2,
          n_pairs=8,
          noise=0.1,
          weight_decay=1e-5,
          tildeC=1,
          device=torch.device("cpu"),
          verbose=False,
          report_interval=1,
          lossfn=MultiW(alpha=0.5, epsilon=0.1, p=2, p_unif=True, normalize_F=True, normalize_MF_MC=False),
          F=None,
          C=None,
          train_vertex_index=None,
          mask=None,
          valid_idx=None,                      # New: evaluation indices tensor/list.
          evaluation_check_interval:int=10,   # New: check interval (if zero, skip evaluation)
          plots: bool = False):               # New: whether to plot loss and MAE curves.
    """
    This function implements the GRIOT imputation procedure.

    New behavior:
      - Every evaluation_check_interval epochs (if >0) the current imputed features are evaluated
        (using MAE on valid_idx compared to the ground-truth F) and stored.
      - The best imputed feature matrix (lowest MAE on valid_idx) is returned.
      - If evaluation_check_interval is 0, no evaluation is done and the final imputed matrix is returned.
      - If plots is True, a plot of loss and evaluation MAE versus iterations is displayed.
    """
    def nanmean(v, *args, **kwargs):
        v = v.clone()
        is_nan = torch.isnan(v)
        v[is_nan] = 0
        return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

    # Prepare dense adjacency matrix from C (assumed to be a dense matrix)
    adjacency_matrix = C.clone()
    adjacency_matrix[adjacency_matrix != 1] = 0
    edge_index = get_edge_index(adjacency_matrix)
    n, d = F.shape
    rand = torch.randn(mask.shape).double()

    if batch_size > n // 2:
        e = int(np.log2(n // 2))
        batch_size = 2 ** e
        if verbose:
            print(f"Batchsize larger than half size ({len(F) // 2}). "
                  f"Setting batch_size to {batch_size}.")

    optimizer = opt(model.parameters(), lr=lr, weight_decay=weight_decay)

    # For evaluation, we need the ground-truth features for the missing entries.
    F_true = F.clone().cpu()  # ground truth is assumed to be in F (for evaluation only)

    imps = (noise * rand + nanmean(F, 0))
    F[~mask.bool()] = imps[~mask.bool()]
    F_filled = F.clone().cpu()

    # For plotting and early stopping based on evaluation.
    loss_steps = dict()
    valid_mae_list = []
    epoch_list = []
    best_mae = None
    best_F = None

    del imps, rand

    for i in range(max_iters):
        if verbose:
            print("=" * 20, f"Iteration {i}")
        for k in range(n_iters):
            # if verbose:
            #     print(k, "-" * 15)
            optimizer.zero_grad()
            loss = torch.tensor(0.0, requires_grad=True)

            F_filled = F_filled.detach()
            imps = model(F_filled.clone(), edge_index).squeeze()
            if len(imps.shape) == 1:
                imps = imps.unsqueeze(1)
            F_filled[~mask.bool()] = imps[~mask.bool()].cpu()
            for _ in range(n_pairs):
                idx1 = torch.randint(high=train_vertex_index.size(0), size=(batch_size,))
                idx1 = train_vertex_index[idx1]
                idx2 = torch.randint(high=train_vertex_index.size(0), size=(batch_size,))
                idx2 = train_vertex_index[idx2]

                # if verbose:
                #     print(f"idx1: {idx1.tolist()}, idx2: {idx2.tolist()}")

                F1 = F_filled[idx1]
                F2 = F_filled[idx2]

                p1 = p[idx1]
                p2 = p[idx2]

                if tildeC == 0:
                    C1 = C[idx1, :][:, list(set(idx1.tolist()) | set(idx2.tolist()))]
                    C2 = C[idx2, :][:, list(set(idx1.tolist()) | set(idx2.tolist()))]
                elif tildeC > 0:
                    C1 = C[idx1]
                    C2 = C[idx2]
                else:
                    C1 = C[idx1, :][:, idx1]
                    C2 = C[idx2, :][:, idx2]

                l = lossfn(C1=C1.to(device), C2=C2.to(device),
                           F1=F1.to(device), F2=F2.to(device),
                           p1=p1.to(device), p2=p2.to(device))
                loss = loss + l / n_pairs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute the imputed features with the current parameters.
        with torch.no_grad():
            imps = model(F_filled.clone(), edge_index).squeeze().cpu()
            if len(imps.shape) == 1:
                imps = imps.unsqueeze(1)
            F_filled[~mask.bool()] = imps[~mask.bool()]

        if verbose and (i % report_interval == 0):
            print(f'Iteration {i}:\t Loss: {loss.item() / n_pairs:.4f}')

        loss_steps[i] = loss.item()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("Nan or inf loss encountered.")
            loss_steps[i] = -1
            break

        # Evaluation check: if evaluation_check_interval > 0 and valid_idx provided.
        if evaluation_check_interval > 0 and valid_idx is not None and (i % evaluation_check_interval == 0):
            # Compute Mean Absolute Error on evaluation indices.
            with torch.no_grad():
                # Re-compute imputed features.
                imputed_current = model(F_filled.clone(), edge_index).squeeze().cpu()
                if len(imputed_current.shape) == 1:
                    imputed_current = imputed_current.unsqueeze(1)
                # Create a combined evaluation mask (True where both mask and valid_idx are True)
                valid_mask = torch.zeros_like(mask, dtype=torch.bool)  # Initialize as all False
                valid_mask[valid_idx] = mask[valid_idx].bool()        # Set True only for valid_idx
                # Select ground truth and predicted values over the evaluation mask.
                F_gt = F_true[valid_mask]
                F_pred = imputed_current[valid_mask]
                mae_eval = torch.mean(torch.abs(F_gt - F_pred)).item()
                valid_mae_list.append(mae_eval)
                epoch_list.append(i)
                if verbose:
                    print(f"Evaluation at iter {i}: MAE = {mae_eval:.4f}")
                # Update best imputed features
                if best_mae is None or mae_eval < best_mae:
                    F_filled[~mask.bool()] = imputed_current[~mask.bool()]
                    best_mae = mae_eval
                    best_F = F_filled.clone()

    # Plotting if requested.
    if plots and evaluation_check_interval > 0 and len(epoch_list) > 0:
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(list(loss_steps.keys()), list(loss_steps.values()), marker='o')
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Loss per Iteration")

            plt.subplot(1, 2, 2)
            plt.plot(epoch_list, valid_mae_list, marker='o', color='orange')
            plt.xlabel("Iteration")
            plt.ylabel("MAE on valid_idx")
            plt.title("Evaluation MAE per Check Interval")
            plt.tight_layout()
            plt.show()
        except ImportError:
            if verbose:
                print("Matplotlib not available; skipping plot.")

    # If evaluation was not done, set best_F to final imputation.
    if evaluation_check_interval == 0 or best_F is None:
        best_F = F_filled

    if verbose and evaluation_check_interval > 0:
        print(f"Best evaluation MAE {best_mae:.4f} observed at iteration {epoch_list[valid_mae_list.index(best_mae)]}.")

    # Return the imputed feature matrix corresponding to the smallest evaluation MAE.
    return best_F

class GCN_IMPUTER(torch.nn.Module):
    def __init__(self, d, dropout=0.5, min_max=None, device=None):
        super(GCN_IMPUTER, self).__init__()
        self.dropout = dropout
        self.device = device
        self.conv1 = GCNConv(in_channels=d,
                             out_channels=int(d ** 0.5)).to(self.device)
        self.conv2 = GCNConv(in_channels=int(d ** 0.5),
                             out_channels=int(d ** 0.5)).to(self.device)
        self.out = torch.nn.Linear(int(d ** 0.5), d).to(self.device)
        self.min_max = min_max

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sigmoid(self.out(x))
        if self.min_max is not None:
            x = x * (self.min_max[1] - self.min_max[0]) + self.min_max[0]
        return x


def get_ot(C1, C2, F1, F2, p1=None, p2=None,
           p_unif=True,
           normalize_F=True,
           normalize_MF_MC=False,
           alpha=0.5,
           epsilon=0.1):
    C1 = C1.clone()
    C2 = C2.clone()
    F1 = F1.clone()
    F2 = F2.clone()

    if p1 is None or p_unif:
        p1 = torch.ones(C1.shape[0], dtype=torch.float32)
    if p2 is None or p_unif:
        p2 = torch.ones(C2.shape[0], dtype=torch.float32)

    p1 = p1.clone()
    p2 = p2.clone()

    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()

    p1 = p1.to(F1.device)
    p2 = p2.to(F2.device)

    if normalize_F:
        F1 = (F1 - F1.mean(dim=0)) / (F1.max(dim=0).values - F1.min(dim=0).values + 1e-5)
        F2 = (F2 - F2.mean(dim=0)) / (F2.max(dim=0).values - F2.min(dim=0).values + 1e-5)
    C1 = (C1 - C1.min()) / (C1.max() - C1.min())
    C2 = (C2 - C2.min()) / (C2.max() - C2.min())

    M_F = torch.cdist(F1, F2, p=2) ** 2 / 2
    M_C = torch.cdist(C1, C2, p=2) ** 2 / 2

    if normalize_MF_MC:
        M_F = (M_F - M_F.min()) / (M_F.max() - M_F.min())
        M_C = (M_C - M_C.min()) / (M_C.max() - M_C.min())

    M = (1 - alpha) * M_F + (alpha) * M_C
    ot_plan = ot.sinkhorn(p1, p2, M, reg=epsilon)
    return ot_plan

def compute_distance_matrix(edge_index, n):
    """
    Compute the distance matrix (proximity matrix) for a graph with n nodes.
    Uses networkx to compute all-pairs shortest path lengths.

    Parameters:
      edge_index: torch.Tensor of shape [2, E] containing edge indices.
      n:         Number of nodes.

    Returns:
      C:         torch.Tensor of shape (n, n) containing distances between nodes.
                 Nodes that are not connected will have value np.inf.
    """
    # Create an undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # edge_index is assumed to be a 2xE tensor; convert it to a numpy array of edges
    edges = edge_index.cpu().numpy().T  # shape (E, 2)
    # Add edges with unit weight
    G.add_edges_from(edges)

    # Initialize the distance matrix with infinity and zeros on the diagonal.
    C = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(C, 0.0)

    # Compute shortest path length for each node.
    for i in range(n):
        lengths = nx.shortest_path_length(G, source=i).items()
        for j, l in lengths:
            C[i, j] = l
    return torch.tensor(C, dtype=torch.float64)

def impute_griot(data, mask, train_idx=None, val_idx=None, verbose=False, seed=0, **kwargs):
    """
    Impute missing values on the graph data using the GRIOT framework.

    Parameters:
        data: a torch_geometric.data.Data object containing at least:
              - data.x as a tensor of node features
              - data.edge_index (if available, although we reconstruct the adjacency from C)
        mask: Boolean tensor of shape (n,d) where True indicates observed values and False missing.
        train_idx: tensor of indices for training nodes.
        val_idx: tensor of indices for validation nodes.
        verbose: Boolean flag for printing detailed information.

    Returns:
        imputed: torch.Tensor of shape (n,d) with imputed features.
    """
    # Set seeds for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Set device from data.x
    device = data.x.device
    # Set default dtype to double
    torch.set_default_dtype(torch.double)

    # Prepare ground truth features F_true (for internal monitoring) and F (working copy)
    F = data.x.clone().double().to(device)

    n, d = F.shape

    # Prepare dense adjacency matrix C from data.edge_index.
    # If data already has an adjacency matrix, you can use it instead.
    if hasattr(data, 'edge_index'):
        # C = to_dense_adj(data.edge_index, max_num_nodes=n).squeeze(0).to(device)
        C = compute_distance_matrix(data.edge_index, n).to(device)
    else:
        raise ValueError("data must have attribute edge_index.")

    # Create a p vector (uniform weights) for nodes.
    p = torch.ones(n, dtype=torch.float64, device=device)
    p = p / p.sum()

    # For training indices, simply use all nodes.
    train_vertex_index = train_idx

    # Instantiate the graph imputer model.
    model = GCN_IMPUTER(d=d, dropout=0.5, device=device)
    model = model.to(device)

    # Set hyperparameters.
    lr = 1e-2
    opt = torch.optim.Adam
    max_iters = 250        # For demonstration; increase for better performance.
    n_iters = 16
    batch_size = min(32, n // 2)
    n_pairs = 8
    noise = 0.1
    weight_decay = 1e-5
    tildeC = 1
    scaling = 0.9
    report_interval = 1

    # Instantiate the loss function (MultiW).
    lossfn = MultiW(alpha=0.5,
                    epsilon=0.1,
                    p=2,
                    scaling=scaling,
                    p_unif=True,
                    normalize_F=True,
                    normalize_MF_MC=False,
                    use_geomloss=True)

    if verbose:
        print("Starting GRIOT imputation...")

    # Call the griot procedure. It returns F_filled with imputed values.
    imputed = griot(model=model,
                    p=p,
                    lr=lr,
                    opt=opt,
                    max_iters=max_iters,
                    n_iters=n_iters,
                    batch_size=batch_size,
                    n_pairs=n_pairs,
                    noise=noise,
                    weight_decay=weight_decay,
                    tildeC=tildeC,
                    device=device,
                    verbose=verbose,
                    report_interval=report_interval,
                    lossfn=lossfn,
                    F=F,
                    C=C,
                    train_vertex_index=train_vertex_index,
                    mask=mask,
                    valid_idx=None,#val_idx,
                    evaluation_check_interval=1,
                    plots=False)

    if verbose:
        print("GRIOT imputation completed.")

    return imputed.detach()


# -----------------------------------------------------------------------------------------------------------------
#                                  ▗▄▄▖▗▖    ▗▄▖  ▗▄▄▖ ▗▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▖  ▗▄▄▖
#                                 ▐▌   ▐▌   ▐▌ ▐▌▐▌   ▐▌     █  ▐▌     █  ▐▌   ▐▌ ▐▌▐▌
#                                 ▐▌   ▐▌   ▐▛▀▜▌ ▝▀▚▖ ▝▀▚▖  █  ▐▛▀▀▘  █  ▐▛▀▀▘▐▛▀▚▖ ▝▀▚▖
#                                 ▝▚▄▄▖▐▙▄▄▖▐▌ ▐▌▗▄▄▞▘▗▄▄▞▘▗▄█▄▖▐▌   ▗▄█▄▖▐▙▄▄▖▐▌ ▐▌▗▄▄▞▘
# -----------------------------------------------------------------------------------------------------------------


def get_conv(conv_type, in_channels, out_channels):
    # For simplicity, we only define a GCN; you can extend it for other convolution types.
    if conv_type == "GCN":
        return GCNConv(in_channels, out_channels)
    else:
        raise ValueError(f"Unrecognized conv type: {conv_type}")

class GNNClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=2, dropout=0.0,
                 conv_type="GCN", jumping_knowledge=False, seed=0):
        super(GNNClassifier, self).__init__()
        self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim))
        output_dim = hidden_dim if jumping_knowledge else num_classes
        self.convs.append(get_conv(conv_type, hidden_dim, output_dim))

        self.jumping_knowledge = jumping_knowledge
        if jumping_knowledge:
            self.jump = JumpingKnowledge(mode="max", channels=hidden_dim, num_layers=num_layers)
            self.lin = Linear(hidden_dim, num_classes)
        self.dropout = dropout
        self.num_layers = num_layers

        # Set seeds for reproducibility.
        self.seed=seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs)-1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)
        return F.log_softmax(x, dim=1)


