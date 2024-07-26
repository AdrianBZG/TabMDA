from collections import defaultdict
import logging
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.datasets import MNIST

from utils import GLOBAL_SEED

VALID_DATASETS = ['qsar-biodeg', 'vehicle', 'texture', 'steel-plates-fault', 'MiceProtein', 'mfeat-fourier']

class ManualStopError(Exception):
    def __init__(self, message="Manually stop the process."):
        self.message = message
        super().__init__(self.message)

dataset2supported_num_real_samples = {
    'mfeat-fourier': [50, 100, 200, 500, -1],
    'MiceProtein': [50, 100, 200, 500, -1],
    'qsar-biodeg': [20, 50, 100, 200, 500, -1],
    'steel-plates-fault': [20, 50, 100, 200, 500, -1],
    'texture': [50, 100, 200, 500, -1],
    'vehicle': [20, 50, 100, 200, -1],
}

logging.basicConfig(level=logging.INFO,
                    format='[utils:%(levelname)s] %(message)s')


# ========================================================
#                      LOAD DATASETS
# ========================================================

def load_tabular_dataset(dataset: str) -> dict:
    """Load the full dataset with its name

    Args:
        dataset (str): The name of the dataset to load

    Returns:
        dict: loaded dataset (X, y)
    """

    # ===== Dataset info =====
    dataset2openml_id = {
        'qsar-biodeg': 1494,
        'vehicle': 54,
        'texture': 40499,
        'steel-plates-fault': 1504,
        'MiceProtein': 40966,
        'mfeat-fourier': 14,
    }

    # ===== Load the dataset =====
    if dataset in dataset2openml_id.keys():
        # Load the qsar-biodeg dataset
        data_loaded = fetch_openml(data_id=dataset2openml_id[dataset], as_frame=True, parser='auto')
        # Get the features and target variables
        X = data_loaded.data
        y = data_loaded.target
        # map the labels to start from 0
        X = np.asarray(data_loaded.data, dtype=np.float32)
        y = y.astype('category').cat.codes
        y = np.asarray(y, dtype=int)
    else:
        raise ValueError(f"Dataset {dataset} not recognised.")

    return {'X': X, 'y': y}


# =======   CORE FUNCTION FOR DATA LOADING   =======    
def get_tabular_dataset(
    dataset: str,
    num_real_samples: int | None,
    repeat_id: int,
    ratio_train: float = 0.8,
) -> dict:
    """Load the dataset and split it into training, validation, oracle, and test sets

    Args:
        dataset (str): name of the dataset
        num_real_samples (int | None): number of real samples in the training set. None for Oracle
        repeat_id (int): repeat id for the dataset
        ratio_train (float): ratio of training samples

    Returns:
        dict: dictionary of the loaded dataset
    """
    assert dataset in VALID_DATASETS, f"Dataset {dataset} not recognised from supported datasets {VALID_DATASETS}."

    if dataset in dataset2supported_num_real_samples.keys() and num_real_samples not in dataset2supported_num_real_samples[dataset]:
        raise ManualStopError(f"Number of real samples is not supported for the dataset {dataset}."
                              f"Supported numbers are {dataset2supported_num_real_samples[dataset]}.")

    # ===== Load the full dataset =====
    data_dict = load_tabular_dataset(dataset)
    X = data_dict['X']
    y = data_dict['y']

    # ===== Load the indices for Oracle and Test =====
    indices_dict = load_oracle_test_indices(dataset, repeat_id)
    indices_oracle_per_class = indices_dict['indices_oracle_per_class']
    indices_oracle = indices_dict['indices_oracle']
    indices_test = indices_dict['indices_test']

    # ===== Sample stratified Train and Validation =====
    indices_train_val = get_train_val_indices(
        indices_oracle_per_class=indices_oracle_per_class,
        num_real_samples=num_real_samples,
        ratio_train=ratio_train,
    )
    indices_train = indices_train_val['indices_train']
    indices_val = indices_train_val['indices_val']

    # ===== Split the dataset into training, oracle, and test sets =====
    X_train = X[indices_train]
    y_train = y[indices_train]
    X_val = X[indices_val]
    y_val = y[indices_val]
    X_test = X[indices_test]
    y_test = y[indices_test]

    # ===== Impute missing values =====
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    if X_val.shape[0] > 0:
        X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # ===== Standardize the data =====
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_val.shape[0] > 0:
        X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ===== Print the dataset information =====
    print(f"[Data INFO] Dataset: {dataset}")
    print(f"[Data INFO] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[Data INFO] X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"[Data INFO] X_test: {X_test.shape}, y_test: {y_test.shape}")

    print('--------- Class distribution for Train / Val / Test ---------')
    # print the percentage of each class in the training set
    for class_id in np.unique(y_train):
        print(f"[Data INFO] Class {class_id}: Train {np.sum(y_train == class_id) / len(y_train) * 100:.2f}%, "\
                f"Val {np.sum(y_val == class_id) / len(y_val) * 100:.2f}%, "\
                f"Test {np.sum(y_test == class_id) / len(y_test) * 100:.2f}%")


    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }


def load_oracle_test_indices(
    dataset: str,
    repeat_id: int,
) -> dict:
    """Load the indices for training, oracle, and test sets

    Args:
        project_dir (str): project directory
        dataset (str): name of the dataset
        repeat_id (int): repeat id for the dataset

    Returns:
        dict: dictionary of indices for training, oracle, and test sets
    """
    # ===== Parse the project directory =====
    current_dir = os.getcwd()
    if 'evaluation_TabEBM' in current_dir:
        project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir))
    elif 'notebooks_TabEBM' in current_dir:
        project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    else:
        project_dir = os.getcwd()

    # ===== Load the indices for Oracle and Test =====
    # indices_oracle is a pickled dictionary, and `item()` is used to convert it to a dictionary
    indices_oracle_per_class = np.load(os.path.join(
        project_dir,
        'data',
        'index',
        dataset,
        f'split_{repeat_id}',
        'indices_oracle.npy',
    ),
                                       allow_pickle=True).item()
    indices_oracle = np.concatenate(
        [indices_oracle_per_class[class_id] for class_id in indices_oracle_per_class.keys()])
    indices_test = np.load(
        os.path.join(
            project_dir,
            'data',
            'index',
            dataset,
            f'split_{repeat_id}',
            'indices_test.npy',
        ))

    return {
        'indices_oracle_per_class': indices_oracle_per_class,
        'indices_oracle': indices_oracle,
        'indices_test': indices_test,
    }


def get_train_val_indices(
    indices_oracle_per_class: dict,
    num_real_samples: int | None,
    ratio_train: float,
) -> dict:
    """Sample training and validation indices from the Oracle set

    Args:
        X_oracle (np.ndarray): Oracle data features
        y_oracle (np.ndarray): Oracle data labels
        num_real_samples (int | float | None): number/ratio of real samples in the training set. None for Oracle
        ratio_train (float): ratio of training samples in Train+Val
        random_state (int): random state for data splitting

    Returns:
        dict: dictionary of indices for training and validation sets
    """
    # ===== Use Oracle as Train =====
    oracle_size = sum(len(indices_per_class) for indices_per_class in indices_oracle_per_class.values())
    if num_real_samples is None or (isinstance(num_real_samples, int) and num_real_samples > oracle_size):
        num_real_samples = oracle_size

    # ===== Subsample training data sets from Oracle =====
    num_real_samples_per_class = {
        class_id: int((len(indices_oracle_per_class[class_id]) / oracle_size) * num_real_samples)
        for class_id in indices_oracle_per_class.keys()
    }

    # ===== Split the training indices into Train and Validation =====
    indices_train = []
    indices_val = []
    for class_id in indices_oracle_per_class.keys():
        # === Compute the number of training samples per class ===
        num_train_samples_per_class = int(num_real_samples_per_class[class_id] * ratio_train)
        # === at least one sample per class ===
        if num_train_samples_per_class == 0:
            raise ValueError(
                f"Number of training samples for class {class_id} is 0. Increase the number of training samples.")
        # === Record the indices for Train and Validation ===
        indices_train.extend(indices_oracle_per_class[class_id][:num_train_samples_per_class])
        indices_val.extend(
            indices_oracle_per_class[class_id][num_train_samples_per_class:num_real_samples_per_class[class_id]])

    return {
        'indices_train': indices_train,
        'indices_val': indices_val,
    }


def get_oracle_test_indices(
    X: pd.DataFrame,
    y: pd.Series,
    num_splits: int,
    num_test_samples: int,
    random_state: int,
) -> dict:
    """Split the dataset into **stratified** oracle sets and a fixed test set

    Args:
        X (pd.DataFrame): data features
        y (pd.Series): data labels
        num_splits (int): number of data splits
        num_test_samples (int): number of test samples
        random_state (int): random state for data splitting

    Returns:
        dict: dictionary of indices (per split) for oracle and test sets
    """
    indices_oracle_list = []
    indices_test_list = []

    # ===== Split the dataset =====
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=num_test_samples, random_state=random_state)
    for _, (indices_oracle, indices_test) in enumerate(splitter.split(X, y)):
        # === Transform Oracle indices into index dictionary per class ===
        indices_oracle_dict = {}
        for class_id in np.unique(y):
            indices_oracle_dict[class_id] = indices_oracle[y[indices_oracle] == class_id]

        # === Shuffle the Oracle index dictionary ===
        for _ in range(num_splits):
            indices_oracle_dict_temp = copy.deepcopy(indices_oracle_dict)
            for class_id in np.unique(y):
                np.random.shuffle(indices_oracle_dict_temp[class_id])
            # Save the Oracle (per class) and Test indices
            indices_oracle_list.append(indices_oracle_dict_temp)
            indices_test_list.append(indices_test)

    return {
        'indices_oracle_list': indices_oracle_list,
        'indices_test_list': indices_test_list,
    }




# ========================================================
#                      SAMPLING DATAPOINTS
# ========================================================
def get_samples_per_class(
        X,
        y,
        samples_per_class=5,
        classes_to_sample=None,  # If None, then sample from all classes. Otherwise, it should be a list of class labels from which to extract additional samples.
        random=False,  # If False, the first samples are selected. If True, the samples are randomly selected.
        seed=42):
    """
    Return `samples_per_class` samples of each class, together with their labels.
    """
    assert classes_to_sample == None or type(classes_to_sample) == list

    if classes_to_sample == None:
        classes_to_sample = np.arange(len(set(y)))

    X_sampled, y_sampled = [], []
    for target_class in classes_to_sample:
        X_sampled.append(get_samples_of_class(X, y, target_class, samples_per_class, random, seed))
        y_sampled.append(np.full(samples_per_class, target_class))

    if len(X_sampled) == 0:
        return np.array([]), np.array([])
    else:
        return np.concatenate(X_sampled, axis=0), np.concatenate(y_sampled, axis=0)


def get_samples_of_class(
        X,
        y,  # The dataset
        target_class,  # The class to sample from
        samples=-1,  # The number of samples to return (first samples of the class). If -1, return all samples of the class.
        random=False,  # If True, the samples are randomly selected. If False, the first samples are selected.
        seed=42):
    """
    Deterministically return the first `samples` samples of the class `target_class`.
    """
    return get_samples(X[y == target_class], samples, random, seed)


def get_samples(
        X,
        samples=-1,  # The number of samples to return. If -1, return all samples.
        random=False,  # If True, the samples are randomly selected. If False, the first samples are selected.
        seed=42):
    """
    Deterministically return samples from the dataset (either the first samples, or random samples).

    Calling the function with increasing `samples` will return increasing subsets. E.g., the samples selected
        with 'samples=5' will be a subset of the samples selected with 'samples=10'.

    
    Intended use-case: select from the pool of augmented samples.
    """
    if samples == -1:
        return X
    elif random:
        rng = np.random.default_rng(seed)
        indices = np.arange(X.shape[0])
        rng.shuffle(indices)
        return X[indices[:samples]]
    else:
        return X[:samples]



# ========================================================
#                   PCA AND SCALING (FOR VISUALIZATION)
# ========================================================
class PCAandScale(TransformerMixin):
    """
    Transform a dataset by applying PCA transform, and then standardizing the data.
    """

    def __init__(self, pca_components) -> None:
        # === zero-mean scaler for PCA
        self.mean_scaler = StandardScaler(with_mean=True, with_std=False)
        self.pca = PCA(n_components=pca_components, random_state=0)
        self.post_scaler = StandardScaler()

    def fit(self, X):
        X = self.mean_scaler.fit_transform(X)
        X = self.pca.fit_transform(X)
        self.post_scaler.fit(X)

    def transform(self, X):
        X = self.mean_scaler.transform(X)
        X = self.pca.transform(X)
        X = self.post_scaler.transform(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)