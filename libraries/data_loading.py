#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:36:18 2025

@author: FJGS
"""

import numpy as np
import pandas as pd
import os
from scipy.io import arff
from sklearn.model_selection import train_test_split

def random_sample(X, y, n_max=None):
    """Randomly selects up to n_max samples if specified; otherwise, returns the full dataset."""
    if n_max is not None and X.shape[0] > n_max:
        X, _, y, _ = train_test_split(X, y, train_size=n_max, random_state=42, stratify=y if len(set(y)) > 1 else None)
    return X, y

def load_npy_dataset(filepath_x, filepath_y, n_max=None, special_case=False):
    """Loads a dataset from .npy files."""
    X = np.load(filepath_x)
    y = np.load(filepath_y)

    X, y = random_sample(X, y, n_max)
    C0 = 1 if special_case else -1
    return X, y, C0

def load_arff_dataset(filepath, adjust_labels=True, n_max=None):
    """Loads a dataset from an .arff file, handling categorical attributes."""
    data, meta = arff.loadarff(filepath)
    attribute_types = meta.types()[:-1]  # Exclude the class label

    X = []
    for row in data:
        decoded_row = []
        list_row = list(row)[:-1]  # Convert to list, and slice
        for i, val in enumerate(list_row):  # Exclude the class label
            if attribute_types[i] == 'numeric':
                decoded_row.append(float(val.decode('utf-8')) if isinstance(val, bytes) else float(val))
            else:  # Categorical attribute
                decoded_row.append(val.decode('utf-8') if isinstance(val, bytes) else val)
        X.append(decoded_row)

    # Convert categorical values to integers
    for i, attr_type in enumerate(attribute_types):
        if attr_type != 'numeric':
            unique_vals = np.unique(np.array(X)[:, i])
            val_map = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            for row in X:
                row[i] = val_map[row[i]]

    X = np.array(X, dtype=float)
    y = np.array([row[-1] for row in data])

    if isinstance(y[0], bytes):
        y = np.array([val.decode('utf-8') for val in y])

    if adjust_labels:
        unique_labels = np.unique(y)
        label_map = {label: idx + 1 for idx, label in enumerate(unique_labels)}
        y = np.array([label_map[label] for label in y]).astype(int)

    X, y = random_sample(X, y, n_max)
    return X, y, -1

def _load_data_with_converter(filepath, delimiter=","):
    """Loads data from a file, automatically inferring the number of features
    and handling potentially non-numeric target column and '?' for unknown.
    Handles categorical features by converting them to numerical indices.
    '?' values are replaced with NaN.
    """
    try:
        with open(filepath, 'r') as f:
            first_data_line = None
            for line in f:
                line = line.strip()
                if line and not line.startswith('@') and line:  # Ensure line is not empty
                    first_data_line = line
                    break
            if first_data_line:
                parts = [p.strip() for p in first_data_line.split(delimiter) if p.strip()]
                num_features = len(parts) - 1
                d_x = num_features
            else:
                raise ValueError(f"Could not infer the number of features from the data lines in {filepath}.")

        # First pass to identify categorical columns and unique values
        categorical_cols = {}
        data_lines = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('@'):
                    data_lines.append([item.strip() for item in line.split(delimiter) if item.strip()])

        for i in range(d_x):
            is_categorical = False
            unique_values = set()
            for row in data_lines:
                if len(row) > i and row[i] != '?':
                    try:
                        float(row[i])
                    except ValueError:
                        is_categorical = True
                        unique_values.add(row[i])
            if is_categorical:
                categorical_cols[i] = {val: idx for idx, val in enumerate(sorted(unique_values))}

        classes = {}
        def target_converter(x):
            value = x.strip()
            if value == '?':
                return 0  # Or np.nan

            if value == '0' or value.lower() == 'negative':
                return -1
            elif value == '1' or value.lower() == 'positive':
                return 1
            else:
                try:
                    int_value = int(value)
                    if int_value == 0:
                        return -1
                    elif int_value == 1:
                        return 1
                    else:
                        # For other numeric targets, you might need a specific mapping.
                        # Returning as is for now.
                        return int_value
                except ValueError:
                    # Handle other non-numeric target values by mapping them to +/- 1
                    if value not in classes:
                        if not classes:
                            classes[value] = 1
                        elif len(classes) % 2 == 0:
                            classes[value] = -1
                        else:
                            classes[value] = 1
                    return classes.get(value)

        converters = {}
        for i in range(d_x):
            if i in categorical_cols:
                mapping = categorical_cols[i]
                converters[i] = lambda x: mapping.get(x.strip()) if x.strip() != '?' else 0 # np.nan
            else:
                converters[i] = lambda x: float(x) if x.strip() != '?' else 0 # np.nan
        # Assuming target is the last column
        converters[d_x] = target_converter

        # Load data as strings using the pre-processed data_lines
        raw_data = np.array(data_lines)
        num_rows = raw_data.shape[0]
        processed_data = np.empty((num_rows, d_x + 1))
        processed_data[:] = np.nan # Initialize with NaN

        for col_idx in range(d_x):
            converter = converters.get(col_idx)
            if converter is not None:
                for row_idx in range(num_rows):
                    processed_data[row_idx, col_idx] = converter(raw_data[row_idx, col_idx])
            else:
                for row_idx in range(num_rows):
                    value = raw_data[row_idx, col_idx]
                    if value != '?':
                        try:
                            processed_data[row_idx, col_idx] = float(value)
                        except ValueError:
                            processed_data[row_idx, col_idx] = 0 # np.nan
                    else:
                        processed_data[row_idx, col_idx] = 0

        # Convert the target column
        target_converter_func = converters.get(d_x)
        if target_converter_func:
            for row_idx in range(num_rows):
                processed_data[row_idx, d_x] = target_converter_func(raw_data[row_idx, d_x])

        return processed_data

    except Exception as e:
        raise RuntimeError(f"Error loading {filepath} with converter method: {e}")


def load_special_dat_dataset(filepath_tra, filepath_tst, n_max=None):
    """Loads a dataset from .dat files and merges train and test sets."""
    try:
        data_train = np.loadtxt(filepath_tra, comments='@')
    except ValueError as e:
        if "could not convert string" in str(e):
            # print(f"Error loading {filepath_tra} with default np.loadtxt: {e}")
            # print(f"Attempting to load with alternative method (comma-separated)...")
            try:
                data_train = _load_data_with_converter(filepath_tra)
                # print(f"Successfully loaded {filepath_tra} with alternative method.")
            except Exception as e_alt:
                raise RuntimeError(f"Failed to load {filepath_tra} with comma-separated method: {e_alt}")
        else:
            raise  # Re-raise the original ValueError if it's not the string conversion issue
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading {filepath_tra}: {e}")

    try:
        data_test = np.loadtxt(filepath_tst, comments='@')
    except ValueError as e:
        if "could not convert string" in str(e):
            # print(f"Error loading {filepath_tst} with default np.loadtxt: {e}")
            # print(f"Attempting to load with alternative method (comma-separated, assuming same format as train)...")
            try:
                data_test = _load_data_with_converter(filepath_tst)
                print(f"Successfully loaded {filepath_tst} with alternative method.")
            except RuntimeError as e_alt:
                raise RuntimeError(f"Failed to load {filepath_tst} with comma-separated method: {e_alt}")
        else:
            raise
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading {filepath_tst}: {e}")

    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    
    if n_max is not None and len(X) > n_max:
        indices = np.random.choice(len(X), n_max, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y, -1

def load_dat_dataset(filepath, n_max=None):
    path_prefix = os.path.dirname(filepath)
    filename_with_ext = os.path.basename(filepath)
    filename_i, ext = os.path.splitext(filename_with_ext)
    fnames = filepath  # Optimized: Use filepath directly

    if ext == '.npz': 
        filepath_npz = os.path.join(path_prefix, f'{filename_i}.npz')
        try:
            data = np.load(filepath_npz)
            Xy = np.hstack((data['data'], data['label'].reshape(-1, 1)))
        except FileNotFoundError:
            print(f"Warning: NPZ file not found for {filename_i}. Attempting to load as .dat.")
    elif filename_i == 'page-blocks':
        Xy = _load_data_with_converter(fnames, delimiter=' ')
    elif filename_i == 'balance':
        d_x = 4
        classes = {b'B': +1, b'R': -1, b'L': -1}
        Xy = np.loadtxt(fnames, skiprows=0, delimiter=",", comments="@", converters={d_x: lambda x: classes[x.strip()]})
    else:
        Xy = _load_data_with_converter(fnames)

    if 'Xy' in locals():
        d_x = Xy.shape[1] - 1
        X = Xy[:, range(d_x)]
        y = Xy[:, d_x].astype(int)
    else:
        return None, None, -1 # Handle cases where loading fails
    
    if n_max is not None and len(X) > n_max:
        indices = np.random.choice(len(X), n_max, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y, -1

def load_csv_dataset(filepath, n_max=None):
    """
    Loads a dataset from a CSV file.

    Args:
        filepath (str): The full path to the CSV file.
        n_max (int, optional): The maximum number of samples to load.
                                 If None, all samples are loaded. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): The feature matrix.
            - y (numpy.ndarray): The target vector (as integers).
            - C0 (int): The number of majority class samples (set to -1 as it's not directly
                      determined from a generic CSV).
    """
    filename_with_ext = os.path.basename(filepath)
    filename_i, ext = os.path.splitext(filename_with_ext)

    try:
        data = pd.read_csv(filepath, sep=';')  # Specify the semicolon as the separator
        if 'data' in data.columns and 'label' in data.columns:
            X = data['data'].values
            y = data['label'].values.astype(int)
        elif 'target' in data.columns:
            # Assume all other columns are features if 'target' is present
            y = data['target'].values.astype(int)
            X = data.drop(columns=['target']).values
        elif data.shape[1] > 1:
            # Assume the last column is the target if 'data' and 'label'/'target' are not found
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values.astype(int)
        elif data.shape[1] == 1:
            raise ValueError(f"CSV file '{filepath}' contains only one column. "
                             "Please ensure it contains features and a target variable.")
        else:
            raise ValueError(f"CSV file '{filepath}' is empty.")

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{filepath}'.")
        return None, None, -1
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file at '{filepath}' is empty.")
        return None, None, -1
    except ValueError as e:
        print(f"Error loading '{filepath}': {e}")
        return None, None, -1
    except Exception as e:
        print(f"An unexpected error occurred while loading '{filepath}': {e}")
        return None, None, -1

    if filename_i == 'winequality-red':
        y -= 2
    elif filename_i == 'winequality-white':
        y -= 2
    if n_max is not None and len(X) > n_max:
        indices = np.random.choice(len(X), n_max, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y, -1
        
def load_datasets(data_path, n_max=None, dataset_special_cases=None):
    """Loads datasets from a directory with an optional limit on the number of samples."""
    if dataset_special_cases is None:
        dataset_special_cases = set()
        
    datasets = {}
    for input_path_prefix, _, files in os.walk(data_path):
        for file in files:
            filename_i, ext = os.path.splitext(file)

            if file.startswith('X_') and ext == '.npy':
                dataset_name = filename_i.removeprefix('X_')
                X, y, C0 = load_npy_dataset(
                    os.path.join(input_path_prefix, f"X_{dataset_name}.npy"),
                    os.path.join(input_path_prefix, f"Y_{dataset_name}.npy"),
                    n_max,
                    special_case=(dataset_name == "Sintetica300v035"),
                )
                datasets[dataset_name] = (X, y, C0)

            elif ext == '.arff':
                X, y, C0 = load_arff_dataset(
                    os.path.join(input_path_prefix, file),
                    adjust_labels=(filename_i not in dataset_special_cases),
                    n_max=n_max
                )
                datasets[filename_i] = (X, y, C0)

            elif file.endswith('tra.dat'):
                dataset_name = filename_i.rsplit('-', 2)[0] if len(filename_i.rsplit('-', 2)) > 2 else filename_i
                X, y, C0 = load_special_dat_dataset(
                    os.path.join(input_path_prefix, file),
                    os.path.join(input_path_prefix, filename_i.replace("tra", "tst") + ".dat"),
                    n_max=n_max
                )
                datasets[dataset_name] = (X, y, C0)
            elif ext == '.dat' or ext == '.npz':
                dataset_name = filename_i
                X, y, C0 = load_dat_dataset(
                    os.path.join(input_path_prefix, file),
                    n_max=n_max
                )
                datasets[dataset_name] = (X, y, C0)

            elif ext == '.csv':
                dataset_name = filename_i
                X, y, C0 = load_csv_dataset(
                    os.path.join(input_path_prefix, file),
                    n_max=n_max
                )
                datasets[dataset_name] = (X, y, C0)
    return datasets
