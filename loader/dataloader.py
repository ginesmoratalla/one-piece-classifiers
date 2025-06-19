import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np


class OnePieceDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt()
        self.batch_size = 16
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0])
        print(self.y)
        self.num_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num_samples
