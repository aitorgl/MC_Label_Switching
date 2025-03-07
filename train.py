import os
import pickle
import logging

import warnings
from itertools import product

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Create a mapping dictionary
METRIC_FUNCTIONS = {
    "balanced_accuracy_score": balanced_accuracy_score,
    "f1_score": f1_score,
    "matthews_corrcoef": matthews_corrcoef,
}

# Imbalanced Learning Metrics
from imblearn.metrics import geometric_mean_score, sensitivity_score

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier

# Custom Models
from libraries.mlpbayesian import MLPBayesBinW, MLPBayesSwitch
from libraries.labelswitching import LSEnsemble

# Custom Libraries
from libraries.ecoc import ECOC
from libraries.data_loading import load_datasets
from libraries.functions import load_config, setup_logger, get_class_from_string
from libraries.functions import generate_model_configurations, apply_ecoc_binarization

from libraries.imbalance_degree import imbalance_degree

# Suppress ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load Configuration
config = load_config('config.yaml')

# -----------------------------------------------------------------------------
# INITIALIZE LOGGER
# -----------------------------------------------------------------------------
# Setup Logger
logger = setup_logger(config)
"""
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)
FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(
    format=FORMAT, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
"""

# Common Parameters
verbose = config["simulation"]["verbose"]
data_path = config["paths"]["data_folder"]
output_path = config["paths"]["output_folder"]
if not os.path.exists(output_path):
    print(" New folder created: {}".format(output_path))
    os.makedirs(output_path)
    
n_simus = config["simulation"]["n_simus"]
Test_size = config["simulation"]["test_size"]
N_max = config["simulation"]["N_MAX"]
num_folds = config["simulation"]["num_folds"]
ECOC_enc = config["simulation"]["ecoc_enc"]
flg_swp = config["simulation"]["flg_swp"]
maj_min = config["simulation"]["maj_min"]
min_maj = config["simulation"]["min_maj"]
metric_function_name = config["simulation"]["f_sel"]

# Retrieve the actual function from the mapping
if metric_function_name not in METRIC_FUNCTIONS:
    raise ValueError(f"Invalid metric function '{metric_function_name}'. Choose from {list(METRIC_FUNCTIONS.keys())}.")
f_sel = METRIC_FUNCTIONS[metric_function_name]
model_selection = config["simulation"]["model_selection"]

# Model Configuration
model_list = config["models"]
model_list = [model_list[2]]  # Just LSEnsemble

# Generate Model Configurations
CV_config = generate_model_configurations(model_list)

# Load Datasets
dataset_special_cases = {}
datasets = load_datasets(data_path, N_max, dataset_special_cases)

# Process Datasets
for dataset_name, (X, y, C0) in datasets.items():
    n_samples, n_attributes = X.shape
    nclass = np.unique(y).shape[0]

    if maj_min:
        sorted_classes = pd.Series(y).value_counts().index.to_list()
        class_dict = {sorted_classes[k]: nclass - k for k in range(nclass)}
    elif min_maj:
        sorted_classes = pd.Series(y).value_counts().index.to_list()
        class_dict = {sorted_classes[k]: k + 1 for k in range(nclass)}
    else:
        class_dict = {k + 1: k + 1 for k in range(nclass)}
    y = np.vectorize(class_dict.get)(y)

    class_labels = np.array(sorted(class_dict.keys()))

    M_ecoc = ECOC(encoding=ECOC_enc, labels=class_labels)
    M = (M_ecoc._code_matrix > 0).astype(int)

    if True:
        M[M == 1] = -C0
        M[M == 0] = C0

    M_ecoc._code_matrix = M

    num_dichotomies = M_ecoc._code_matrix.shape[1]
    logger.info('')
    logger.info('-----------------------------------------------------')
    logger.info(f'Dataset: {dataset_name}. Imbalance Degree = {imbalance_degree(y, "EU"):.2f}')
    logger.info(f'ECOC encoding: {M_ecoc.encoding}. Flag swap: {flg_swp}')
    logger.info(f'Number of classes: {nclass}. Number of Dichotomies: {num_dichotomies}')
    logger.info('f_sel: '+f_sel.__name__)
    
    best_model_avg = {}
    metric_conf = {}
    CM_accumulated = {}
    best_metric_avg = dict()
    best_metric_peak = dict()
    best_model_peak = dict()
    for model_item in model_list:
        model_name = model_item["name"]
        logger.info(f'Model: {model_name}')

        if model_name == 'LSEnsemble':
            SW_optimization = model_item['LSE_optimization']['SW']
            QC_optimization = model_item['LSE_optimization']['QC']
            RI_C_optimization = model_item['LSE_optimization']['RI_C']
            RI_P_optimization = model_item['LSE_optimization']['RI_P']
            logger.info(f'    Switching: {int(SW_optimization)}, QC: {int(QC_optimization)}, Cost: {int(RI_C_optimization)}, Population: {int(RI_P_optimization)}')
        
        dynamic_combinations = model_item["dynamic_params"]
        n_conf_test = len(list(product(*model_item["dynamic_params"].values())))

        CM_accumulated[model_name] = [np.zeros((n_conf_test, 2, 2)) for j_dic in range(num_dichotomies)]  # Accumulated confusion matrix
        metric_conf[model_name] = np.zeros((n_conf_test, num_dichotomies, num_folds, n_simus))
        best_metric_avg[model_name] = np.zeros(num_dichotomies)
        best_model_avg[model_name] = [dict() for j_dic in range(num_dichotomies)]
        best_metric_peak[model_name] = np.zeros(num_dichotomies)
        best_model_peak[model_name] = [dict() for j_dic in range(num_dichotomies)]
 
    logger.info('-----------------------------------------------------')
            
    for k_simu in range(n_simus):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01*Test_size, random_state=42+k_simu)
        M_tr = X_train.shape[0]
        M_tst = X_test.shape[0]
        input_size = X_train.shape[1]
    
        scaler = StandardScaler()  # Normalización de características
        scaler.fit(X_train)
        X_train_n = scaler.transform(X_train)
        X_test_n = scaler.transform(X_test)
        
        # Inner 5-Fold for validation
        inner_cv = KFold(n_splits=num_folds, shuffle=True, random_state=42+k_simu)
        for nFold, (train_index, val_index) in enumerate(inner_cv.split(X_train, y_train), 1):
            if verbose:
                logger.info(f"Simulation {k_simu+1}. Inner Fold {nFold}:")

            X_train_cv, X_test_cv = X_train_n[train_index], X_train_n[val_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[val_index]
            
            # Apply ECOC binarization
            Ye_train, Ye_test, flag_swap, idx_train_ecoc, idx_test_ecoc = apply_ecoc_binarization(M, y_train_cv, y_test_cv, apply_flag_swap = flg_swp)
                
            for j_dic in range(num_dichotomies):
                ye_train, ye_test = Ye_train[j_dic], Ye_test[j_dic]
                
                # Check if the dichotomy contains both classes
                unique_labels = np.unique(np.concatenate([ye_train, ye_test]))
                unique_labels_train = unique_labels[np.isin(unique_labels, ye_train)]
                unique_labels_test = unique_labels[np.isin(unique_labels, ye_test)]
                
                if len(unique_labels_train) < 2:
                    continue

                x_train = X_train_cv[idx_train_ecoc[j_dic], :]
                x_test = X_test_cv[idx_test_ecoc[j_dic], :]
    
                cw_train = np.ones(len(ye_train))
                
                for model_item in model_list:
                    model_name = model_item["name"]
                    model_class = get_class_from_string(model_item["class"])
                    
                    k_conf = 0
                    for cv_config in CV_config[model_name]:
                        if model_name == "LSEnsemble":
                            cv_config.update({"input_size": input_size})
                        model = model_class(**cv_config)
                                
                        # Train the model
                        if model_name == "MLPClassifier":
                            model.fit(x_train, ye_train)
                        else:
                            model.fit(x_train, ye_train, sample_weight=cw_train)
    
                        # Evaluate the model
                        ye_pred = model.predict(x_test)
                            
                        CM = confusion_matrix(ye_test, ye_pred, labels=unique_labels)
                        CM_accumulated[model_name][j_dic][k_conf] += CM
                                                      
                        if len(unique_labels_test) == 1 and len(np.unique(ye_pred)) == 1:
                            metric = 1.0 if np.array_equal(ye_test, ye_pred) else 0.0
                        else:
                            metric = f_sel(ye_test, ye_pred)

                        metric_conf[model_name][k_conf, j_dic, nFold-1, k_simu] = metric
    
                        if metric > best_metric_peak[model_name][j_dic]:
                            best_metric_peak[model_name][j_dic] = metric
                            best_model_peak[model_name][j_dic] = [cv_config, metric, CM]
                            if model_selection == "peak" and verbose:
                                logger.info(f'  Model: {model_name}. Dichotomy: {j_dic+1}')
                                logger.info(f'      Best configuration (peak): {cv_config}')
                                logger.info(f'      Best metric (peak): {best_metric_peak[model_name][j_dic]:.5f}')
                                logger.info('')
                        k_conf += 1
        
    
    # 1. Compute best_metric_avg (average performance)
    for model_item in model_list:
        model_name = model_item["name"]
        for j_dic in range(num_dichotomies):
            k_conf = 0
            for cv_config in CV_config[model_name]:
                # Compute the averaged metric for this configuration after all simulations and folds
                avg_metric_conf = np.mean(metric_conf[model_name][k_conf, j_dic, :, :])
                # Normalize by the total number of test samples to get the mean confusion matrix
                CM_avg = CM_accumulated[model_name][j_dic][k_conf] / (num_folds * n_simus)
    
                # Update best metric and model for the current configuration
                if avg_metric_conf > best_metric_avg[model_name][j_dic]:
                    best_metric_avg[model_name][j_dic] = avg_metric_conf
                    best_model_avg[model_name][j_dic] = [cv_config, avg_metric_conf, CM_avg]
    
                k_conf += 1
    
    best_model = dict()
    best_metric = dict()
    # 2. Best model selection logic
    for model_item in model_list:
        model_name = model_item["name"]
        best_model[model_name] = [dict() for j_dic in range(num_dichotomies)] #initialize
        best_metric[model_name] = [0 for j_dic in range(num_dichotomies)] #initialize
    
        for j_dic in range(num_dichotomies):
            if model_selection == "peak":
                # Select based on peak performance (best_metric_peak)
                best_model[model_name][j_dic] = best_model_peak[model_name][j_dic]  # cv_config, metric, CM
                best_metric[model_name][j_dic] = best_metric_peak[model_name][j_dic]
            elif model_selection == "avg":
                # Select based on average performance (best_metric_avg)
                best_model[model_name][j_dic] = best_model_avg[model_name][j_dic] # cv_config, metric, CM
                best_metric[model_name][j_dic] = best_metric_peak[model_name][j_dic]
            if verbose:
                logger.info(f'Model: {model_name}. Dichotomy: {j_dic+1}')
                logger.info(f'Best configuration ({model_selection}): {best_model[model_name][j_dic][0]}')  # cv_config
                logger.info(f'Best metric ({model_selection}): {best_model[model_name][j_dic][1]:.5f}')  # metric
                logger.info(f'Best CM ({model_selection}): {best_model[model_name][j_dic][2]}')  # CM
                logger.info('')
    
        # Prepare data to save
        result_data = {
            "model_name": model_name,
            "best_config": best_model[model_name],
            "metrics": {
                "best_metric_peak": best_metric_peak[model_name],
                "best_metric_avg": best_metric_avg[model_name],
                "best_metric": best_metric[model_name],
            },
        }
    
        # Generate filename based on dataset and model
        if model_name == 'LSEnsemble':
            SW_optimization = model_item['LSE_optimization']['SW']
            QC_optimization = model_item['LSE_optimization']['QC']
            RI_C_optimization = model_item['LSE_optimization']['RI_C']
            RI_P_optimization = model_item['LSE_optimization']['RI_P']
    
            filename_o = f"{dataset_name}_{ECOC_enc}_{model_name}_SW_{int(SW_optimization)}_QC_{int(QC_optimization)}_RIC_{int(RI_C_optimization)}_RIP_{int(RI_P_optimization)}_train.pkl"
        else:
            filename_o = f"{dataset_name}_{ECOC_enc}_{model_name}_train.pkl"
    
        file_path = os.path.join(output_path, filename_o)
    
        # Save the combined configuration and metrics using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(result_data, f)
    
        logger.info(f"Metrics and configuration saved to {filename_o}")
        logger.info('')