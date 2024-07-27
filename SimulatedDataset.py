from torch.utils.data import Dataset, DataLoader
import torch

class SimulatedDataset(Dataset):
    """
    A simulated data set that generates random data for models A and B.
    The input data dimensions of each model are 10 and the labels are 2 classification problems.
    """
    def __init__(self, num_samples=2500, input_dim=10, num_classes=2): # 25000
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



