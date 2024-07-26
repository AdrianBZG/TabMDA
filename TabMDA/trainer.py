"""
Module defining a trainer to train a TabMDA model (encoder + classifier) on a given dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from utils import to_numpy


def get_torch_items(tensors):
    """
    Given an object with tensors, detach and return the item/numpy of all tensors.

    The original objective of this function is to move the tensors to the CPU to log them.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.detach().item()
    if isinstance(tensors, list):
        return [tensor.detach().item() if isinstance(tensor, torch.Tensor) else tensor for tensor in tensors]
    elif isinstance(tensors, dict):
        result = {}
        for key, value in tensors.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    result[key] = value.detach().item()
                else:
                    result[key] = value.detach().cpu().numpy()
            else:
                result[key] = value
        return result
    else:
        raise Exception("tensors must be a list or a dict")
    

def aggregate_metrics_across_batches(metrics_per_batch):
    """
    Args:
    - metrics_per_batch: List of dictionaries, where each dictionary contains the metrics for a batch
    """
    epoch_metrics = dict()

    # ---- Compute metrics (e.g., balanced accuracy) ----
    y_true = np.concatenate([batch['y_true'] for batch in metrics_per_batch], axis=0)
    y_pred = np.concatenate([batch['y_pred'] for batch in metrics_per_batch], axis=0)
    epoch_metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    # ---- Compute average losses ----
    for metric_name in metrics_per_batch[0].keys():
        if metric_name.endswith('_summed'):
            metric_name_stripped = metric_name[:-7]
            epoch_metrics[metric_name_stripped] = sum(batch[metric_name] for batch in metrics_per_batch) / len(y_true)

    return epoch_metrics


class TrainerTemplate:
    """
    General trainer class for training a model on a given dataset.
    """
    def __init__(self, args, model, loss_fn, device='cpu'):
        """
        Args:
        - args: An object with the configuration parameters (e.g., learning rate, optimizer, etc.)
        """
        super().__init__()

        self.args = args
        self.model = model

        if not loss_fn or not isinstance(loss_fn, nn.Module):
            raise ValueError(f'Loss function has to be defined as a torch.nn.Module subclass.')

        self.classification_loss_fn = loss_fn
        self.device = device

    def calculate_loss(self, y_hat, y_true, x=None):
        classification_loss = self.classification_loss_fn(y_hat, y_true)

        # === Compute the total loss ===
        total_loss = classification_loss

        return {
            # === These are averaged over the batch size ===
            "classification_loss": classification_loss,
            "total_loss": total_loss,

            # === These are *not* averaged over the batch size ===
            "classification_loss_summed": classification_loss * y_true.size(0),
            "total_loss_summed": total_loss * y_true.size(0)
        }

    def calculate_metrics(self, y_true, y_pred):        
        balanced_accuracy = balanced_accuracy_score(y_true=to_numpy(y_true), y_pred=to_numpy(y_pred))
        return {
            "balanced_accuracy": balanced_accuracy
        }

    def prepare_optimizers(self):
        # === Get the parameters that require gradients ===
        params = []
        print(f"=========== TRAINABLE PARAMETERS ===========")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params.append(param)
        trainable_params_count = sum(p.numel() for p in params)
        print(f"=========== TOTAL TRAINABLE PARAMETERS: {trainable_params_count:,} ===========")

        optimizer_type = self.args.optimizer
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(list(params),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(list(params),
                                         lr=self.args.learning_rate,
                                         weight_decay=self.args.weight_decay)
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(list(params),
                                        lr=self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return optimizer
