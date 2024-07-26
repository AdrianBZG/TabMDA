"""
Module to wrap a generic dataset into a PyTorch Dataset
"""

import logging

import numpy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import get_available_device, to_numpy

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(f"{__name__}")


class PytorchDataset(Dataset):
    def __init__(self, x, y, x_context, y_context, device='cpu'):
        assert len(x) == len(y)
        assert type(x) == type(y)
        assert len(x_context) == len(y_context)
        assert type(x_context) == type(y_context)
        
        x = to_numpy(x)
        y = to_numpy(y)
        x_context = to_numpy(x_context)
        y_context = to_numpy(y_context)

        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.long, device=device)
        self.x_context = torch.tensor(x_context, dtype=torch.float32, device=device)
        self.y_context = torch.tensor(y_context, dtype=torch.long, device=device)

    def __getitem__(self, idx):
        return {"x": self.x[idx],
                "y": self.y[idx],
                "x_context": self.x_context,
                "y_context": self.y_context}

    def __len__(self):
        return len(self.x)
    
    def move_to_device(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.x_context = self.x_context.to(device)
        self.y_context = self.y_context.to(device)
        return self
