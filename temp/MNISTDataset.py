# MNISTDataset.py
import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None):
        self.images = self._read_images(image_path)
        self.labels = self._read_labels(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def _read_images(self, file_path):
        with open(file_path, 'rb') as file:
            magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
            images = np.fromfile(file, dtype=np.uint8).reshape(num, 1, 28, 28)
            images = torch.from_numpy(images).float()
            return images

    def _read_labels(self, file_path):
        with open(file_path, 'rb') as file:
            magic, num = struct.unpack(">II", file.read(8))
            labels = np.fromfile(file, dtype=np.uint8)
            labels = torch.from_numpy(labels).long()
            return labels

def custom_transform(image):
    return image / 255.0  # Normalize images from [0, 255] to [0, 1]

def create_mnist_dataloader(image_path, label_path, batch_size=32, transform=None):
    dataset = MNISTDataset(image_path, label_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example of using this dataset
if __name__ == "__main__":
    dataset = MNISTDataset(
        image_path="/home/orin/mnist_data/MNIST/raw/train-images-idx3-ubyte",
        label_path="/home/orin/mnist_data/MNIST/raw/train-labels-idx1-ubyte",
        transform=custom_transform
    )
    print("Dataset size:", len(dataset))
    image, label = dataset[0]
    print("Sample image shape:", image.shape)
    print("Sample label:", label)
