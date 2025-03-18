#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:36:18 2025

@author: FJGS
"""

import numpy as np
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

def load_dat_dataset(filepath_tra, filepath_tst, n_max=None):
    """Loads a dataset from .dat files and merges train and test sets."""
    def read_dat(filepath):
        data = np.loadtxt(filepath)
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y

    X_train, y_train = read_dat(filepath_tra)
    X_test, y_test = read_dat(filepath_tst)

    y_train += 1
    y_test += 1

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    X, y = random_sample(X, y, n_max)
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
                X, y, C0 = load_dat_dataset(
                    os.path.join(input_path_prefix, file),
                    os.path.join(input_path_prefix, filename_i.replace("tra", "tst") + ".dat"),
                    n_max=n_max
                )
                datasets[dataset_name] = (X, y, C0)

    return datasets
