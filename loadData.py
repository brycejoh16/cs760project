# --- Imports ------------------------------------------------------------------
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

# --- Load MNIST Data ----------------------------------------------------------
def loadMNIST():
    batch_size = 12665

    # tensor and normalization transforms
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,)),
    ])

    # load MNIST data from dataset
    train_data = datasets.MNIST(
        root='input/data',
        train=True,
        download=True,
        transform=transform
    )

    # resample to binary dataset
    idx = (train_data.targets == 0) | (train_data.targets == 1)
    train_data.targets = train_data.targets[idx]
    train_data.data = train_data.data[idx]

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # put targets and data in numpy arrays
    data = []
    for value in example_data:
        data.append(np.array(value[0]))

    targets = []
    for value in example_targets:
        targets.append(int(value))

    return data, targets
