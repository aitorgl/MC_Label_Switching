#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:51:23 2025

@author: fran
"""
import numpy as np
import yaml
import logging
from itertools import product
import importlib

def get_class_from_string(class_path):
    """Converts a string class path to a class object."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def load_config(filepath="config.yaml"):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(config):
    """Sets up the logger based on configuration, avoiding duplicate handlers."""
    logger = logging.getLogger(__name__)
    logger.setLevel(config["logging"]["level"])
    formatter = logging.Formatter(config["logging"]["format"], datefmt=config["logging"]["datefmt"])

    # Check for existing handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Matplotlib Logger
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)

    return logger

def generate_model_configurations(model_list):
    """Generates full model configurations from the model list, including LSEnsemble optimization logic."""
    CV_config = {}

    for model_item in model_list:
        model_name = model_item["name"]
        param_grid = model_item["params"]
        CV_config[model_name] = []

        # LSEnsemble optimization logic
        if model_name == 'LSEnsemble':
            SW_optimization = model_item['LSE_optimization']['SW']
            QC_optimization = model_item['LSE_optimization']['QC']
            RI_C_optimization = model_item['LSE_optimization']['RI_C']
            RI_P_optimization = model_item['LSE_optimization']['RI_P']
            
            if not SW_optimization:
                model_item["dynamic_params"]["LS_alpha"] = [0]
                model_item["dynamic_params"]["LS_beta"] = [0]
            if not QC_optimization:
                model_item["dynamic_params"]["LS_Q_C"] = [1]
            if not RI_C_optimization:
                model_item["dynamic_params"]["LS_Q_RB_C"] = [1]
            if not RI_P_optimization:
                model_item["dynamic_params"]["LS_Q_RB_S"] = [1]

        dynamic_params = model_item["dynamic_params"]
        keys = list(dynamic_params.keys())
        values = list(dynamic_params.values())

        for combination in product(*values):
            updated_config = param_grid.copy()
            param_dict = dict(zip(keys, combination))

            if model_name == "MLPBayesBinW":
                updated_config.update({
                    "layers_size": (param_dict['s_NnBase'],),
                    "drop_out": [param_dict['s_pDOent'], param_dict['s_pDOocu']],
                    "activations": [param_dict['s_tActBase'], param_dict['s_tActSalida']],
                    "n_epoch": param_dict['s_nEpoch'],
                    "n_batch": param_dict['s_nBatch'],
                })
            elif model_name == "LGBMClassifier":
                updated_config.update({
                    "num_leaves": param_dict['LGBM_num_leaves'],
                    "learning_rate": param_dict['LGBM_learning_rate'],
                    "n_estimators": param_dict['LGBM_n_estimators'],
                })
            elif model_name == "LSEnsemble":
                updated_config.update({
                    "alpha": param_dict['LS_alpha'],
                    "beta": param_dict['LS_beta'],
                    "QC": param_dict['LS_Q_C'],
                    "Q_RB_S": param_dict['LS_Q_RB_S'],
                    "Q_RB_C": param_dict['LS_Q_RB_C'],
                    "num_experts": param_dict['LS_num_experts'],
                    "hidden_size": param_dict['LS_hidden_size'],
                    "drop_out": param_dict['LS_drop_out'],
                    "n_batch": param_dict['LS_n_batch'],
                    "n_epoch": param_dict['LS_n_epoch'],
                    "mode": param_dict['LS_mode'],
                })
            elif model_name == "LogisticRegression":
                updated_config.update({
                    "C": param_dict['LR_C'],
                    "penalty": param_dict['LR_penalty'],
                })
            elif model_name == "RandomForestClassifier":
                updated_config.update({
                    "n_estimators": param_dict['RF_n_estimators'],
                    "max_depth": param_dict['RF_max_depth'],
                    "min_samples_split": param_dict['RF_min_samples_split'],
                    "min_samples_leaf": param_dict['RF_min_samples_leaf'],
                })
            elif model_name == "MLPClassifier":
                updated_config.update({
                    "hidden_layer_sizes": param_dict['MLP_hidden_layer_sizes'],
                    "activation": param_dict['MLP_activation'],
                    "solver": param_dict['MLP_solver'],
                    "alpha": param_dict['MLP_alpha'],
                })
            CV_config[model_name].append(updated_config)

    return CV_config

def initialize_model_data(model_list, dynamic_combinations, num_dichotomies, num_folds, n_simus, logger):
    """Initializes data structures for models."""
    CM_accumulated = {}
    metric_conf = {}
    best_metric_conf = {}
    best_model_conf = {}
    best_metric_overall = {}
    best_model_overall = {}

    for model_item in model_list:
        model_name = model_item["name"]
        dynamic_combinations = model_item["dynamic_params"]
        num_configurations = len(dynamic_combinations)
        CM_accumulated[model_name] = [np.zeros((num_configurations, 2, 2)) for _ in range(num_dichotomies)]
        metric_conf[model_name] = np.zeros((num_configurations, num_dichotomies, num_folds, n_simus))
        best_metric_conf[model_name] = np.zeros(num_dichotomies)
        best_model_conf[model_name] = [dict() for _ in range(num_dichotomies)]
        best_metric_overall[model_name] = np.zeros(num_dichotomies)
        best_model_overall[model_name] = [dict() for _ in range(num_dichotomies)]

    return CM_accumulated, metric_conf, best_metric_conf, best_model_conf, best_metric_overall, best_model_overall


def apply_ecoc_binarization(M, y_train, y_test, apply_flag_swap=True, flag_swap=None, eps=1e-5, verbose=False):
    """
    Converts multiclass labels into multiple binary labels using the ECOC matrix (-1,1 format).

    Parameters:
    - M: ECOC matrix (shape: [n_classes, n_dichotomies]), defining class partitions.
    - y_train: Multiclass training labels (NumPy array) with values in [1, n_classes].
    - y_test: Multiclass test labels (NumPy array) with values in [1, n_classes].
    - apply_flag_swap: If True, applies label swapping for imbalanced dichotomies.
    - flag_swap: Optional NumPy array indicating swap status per dichotomy (computed if None).
    - eps: Threshold to identify highly imbalanced classes.
    - verbose: If True, prints additional processing details.

    Returns:
    - Y_train_ecoc: List of NumPy arrays, each containing binary training labels per dichotomy.
    - Y_test_ecoc: List of NumPy arrays, each containing binary test labels per dichotomy.
    - flag_swap: NumPy array indicating whether swapping was applied per dichotomy.
    - idx_train_ecoc: List of NumPy arrays containing training indices per dichotomy.
    - idx_test_ecoc: List of NumPy arrays containing test indices per dichotomy.
    """
    num_dichotomies = M.shape[1]
    Y_train_ecoc = [[] for _ in range(num_dichotomies)]
    Y_test_ecoc = [[] for _ in range(num_dichotomies)]
    idx_train_ecoc = [[] for _ in range(num_dichotomies)]
    idx_test_ecoc = [[] for _ in range(num_dichotomies)]
    QP_tr = np.zeros(num_dichotomies)

    if flag_swap is None:
        flag_swap = np.zeros(num_dichotomies)

    for j_dic in range(num_dichotomies):
        dicotomia = M[:, j_dic]

        # Prepare train labels and indices for the dichotomy
        y_train_ecoc = []
        idx_train_ecoc[j_dic] = []
        for i, clase in enumerate(y_train):
            if dicotomia[int(clase) - 1] != 0:  # Check for valid dichotomy label
                y_train_ecoc.append(dicotomia[int(clase) - 1])
                idx_train_ecoc[j_dic].append(i)
            else:
                print(1)

        y_train_ecoc = np.array(y_train_ecoc)
        N0_tr = np.sum(y_train_ecoc == -1)
        N1_tr = np.sum(y_train_ecoc == 1)

        # Apply label swapping if necessary
        if apply_flag_swap and flag_swap[j_dic] == 0 and N0_tr < 0.95 * N1_tr:
            y_train_ecoc *= -1
            flag_swap[j_dic] = 1
            N0_tr, N1_tr = N1_tr, N0_tr
        elif apply_flag_swap and flag_swap[j_dic] == 1:
            y_train_ecoc *= -1
            N0_tr, N1_tr = N1_tr, N0_tr

        P0_tr = N0_tr / (N0_tr + N1_tr)
        P1_tr = N1_tr / (N0_tr + N1_tr)
        QP_tr[j_dic] = 1000 if P1_tr < eps else P0_tr / P1_tr

        # Prepare test labels and indices for the dichotomy
        y_test_ecoc = []
        idx_test_ecoc[j_dic] = []
        for i, clase in enumerate(y_test):
            if dicotomia[int(clase) - 1] != 0:  # Check for valid dichotomy label
                y_test_ecoc.append(dicotomia[int(clase) - 1])
                idx_test_ecoc[j_dic].append(i)
            else:
                print(2)

        y_test_ecoc = np.array(y_test_ecoc)
        if flag_swap[j_dic]:
            y_test_ecoc *= -1

        # Store the results for the current dichotomy
        Y_train_ecoc[j_dic] = y_train_ecoc
        Y_test_ecoc[j_dic] = y_test_ecoc

        if verbose:
            print(f'Dichotomy {j_dic + 1}: {dicotomia}, N0_tr: {N0_tr}, N1_tr: {N1_tr}, IR = {QP_tr[j_dic]:.2f}')

    return Y_train_ecoc, Y_test_ecoc, flag_swap, idx_train_ecoc, idx_test_ecoc