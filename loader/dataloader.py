import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset/onepiecedata.pt')


class OnePieceDataset(Dataset):

    def __init__(self):
        xy = torch.load(DATASET_PATH)
        self.x = torch.stack([tuple[0] for tuple in xy])
        self.y = [tuple[1] for tuple in xy]

        self.num_samples = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num_samples


dataset = OnePieceDataset()
