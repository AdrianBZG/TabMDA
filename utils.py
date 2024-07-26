import json
import os
from collections import defaultdict
import builtins
from typing import Union

import pandas as pd
import torch
import torch.nn.functional as F
import random
import logging
import numpy as np
from tqdm import tqdm

GLOBAL_SEED = 56739723

logging.basicConfig(level=logging.INFO,
                    format='[utils:%(levelname)s] %(message)s')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_model(model_to_freeze):
    for name, param in model_to_freeze.named_parameters():
        param.requires_grad = False


def unfreeze_model(model_to_freeze):
    for name, param in model_to_freeze.named_parameters():
        param.requires_grad = True


def get_model_size(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    return num_parameters


def get_available_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"  # Many issues with MPS so will force CPU here for now
    else:
        device = "cpu"

    logging.info(f"get_available_device() returned {device}")

    return device


def calculate_similarities(X, y):
    # Group embeddings by classes
    embeddings_per_class = defaultdict(list)
    for sample, label in zip(X, y):
        embeddings_per_class[label].append(sample)

    # Mean similarity between samples in the same class
    in_class_similarities = {}
    for label, embeddings in tqdm(embeddings_per_class.items(), desc="Calculating in-class similarities"):
        in_class_similarity = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = F.cosine_similarity(embeddings[i], embeddings[j], dim=0)
                in_class_similarity.append(sim)

        in_class_similarities[label] = torch.stack(in_class_similarity).mean().item()

    # Similarities between classes
    all_labels = list(set(embeddings_per_class.keys()))
    between_class_similarities = {}
    for label, embeddings in tqdm(embeddings_per_class.items(), desc="Calculating between-class similarities"):
        other_labels = [l for l in all_labels if l != label]
        for other_label in other_labels:
            if f'{other_label}<->{label}' in between_class_similarities:
                continue

            similarity_key = f'{label}<->{other_label}'
            other_embeddings = embeddings_per_class[other_label]
            between_class_similarity = []
            for i in range(len(embeddings)):
                for j in range(len(other_embeddings)):
                    sim = torch.abs(F.cosine_similarity(embeddings[i], other_embeddings[j], dim=0))
                    between_class_similarity.append(sim)

            between_class_similarities[similarity_key] = torch.stack(between_class_similarity).mean().item()

    return {"in_class": in_class_similarities,
            "between_class": between_class_similarities}


def to_numpy(x):
    if isinstance(x, builtins.list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.clone().detach().cpu().numpy()
    elif isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, pd.Series):
        return x.values
    elif x is None:
        return None
    else:
        raise ValueError("X must be either a np.ndarray or a torch.Tensor")


def stack_embeddings_to_dataset(
    X_emb: Union[np.ndarray, torch.Tensor],  # The embeddings vector to convert to a dataset (N_samples, N_contexts, embedding_size)
    y: Union[np.ndarray, torch.Tensor],      # Size (N_samples) with labels
):
    """
    Convert the embeddings outputted by the context subsetting function to a dataset.

    Returns:
    - X_emb_stack: The embeddings stacked into a single tensor (N_samples * N_contexts, embedding_size)
    - y_emb: The labels stacked into a single tensor (N_samples * N_contexts)
    """
    assert type(X_emb) == type(y), f"X_emb and y should be of the same type, but are {type(X_emb)} and {type(y)}."

    if type(X_emb) == torch.Tensor:
        X_emb_stack = X_emb.reshape(-1, X_emb.shape[-1])
        y_emb = torch.repeat_interleave(y, X_emb.shape[1], dim=0).to(X_emb_stack.device)
    elif type(X_emb) == np.ndarray:
        X_emb_stack = X_emb.reshape(-1, X_emb.shape[-1])
        y_emb = np.repeat(y, X_emb.shape[1], axis=0)
    else:
        raise ValueError(f"X_emb and y should be either a np.ndarray or a torch.Tensor, but are {type(X_emb)} and {type(y)}.")

    return X_emb_stack, y_emb


def load_model_from_checkpoint(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f'Model checkpoint does not exist at {model_path}')

    model = torch.load(model_path)
    return model


def smote_augmentation_classwise(batch, labels, k=3, rounds=1):
    # batch shape: [batch_size, embedding_size]
    # labels shape: [batch_size]

    new_batch = batch.clone()
    new_labels = labels.clone()

    for round_idx in range(rounds):
        # Initialize tensor to hold augmented samples
        augmented = torch.zeros_like(batch)

        # Process each class separately
        for class_value in torch.unique(labels):
            # Mask to select only samples of the current class
            class_mask = labels == class_value
            class_samples = batch[class_mask]

            if class_samples.size(0) < k:
                raise ValueError(f"Class {class_value} has less than {k} samples. Cannot perform SMOTE augmentation.")

            # Calculate pairwise distances within this class
            distances = torch.cdist(class_samples, class_samples)

            # Set diagonals to a very high value to ignore self-distance
            distances.fill_diagonal_(float('inf'))

            # Get indices of k nearest neighbors, ignoring self (diagonal)
            knn_indices = distances.topk(k, largest=False).indices

            # Randomly select one neighbor from the k nearest
            chosen_indices = knn_indices[torch.arange(class_samples.size(0)), torch.randint(0, k, (class_samples.size(0),))]

            # Gather the selected samples B
            B = class_samples[chosen_indices]

            # Generate random alpha for each sample in the class
            alpha = torch.rand(class_samples.size(0), 1, device=batch.device)

            # Perform interpolation
            class_augmented = alpha * class_samples + (1 - alpha) * B

            # Store augmented samples in the appropriate places
            augmented[class_mask] = class_augmented

        new_batch = torch.cat([new_batch, augmented], dim=0)
        new_labels = torch.cat([new_labels, labels], dim=0)

    return new_batch, new_labels
