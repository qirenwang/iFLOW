# subset_dataloader.py
from torch.utils.data import Subset, DataLoader
import numpy as np

def create_subset_dataloader(dataset, subset_ratio=0.1, batch_size=32, shuffle=True):
    subset_size = int(len(dataset) * subset_ratio)  # Calculate subset size
    indices = np.random.choice(len(dataset), subset_size, replace=False)  # Get random subset indices
    subset = Subset(dataset, indices)  # Create subset
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
