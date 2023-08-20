import os
from random import sample

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, Subset



class SonarDataset(Dataset):
    def __init__(self, X, Y):
        self.data_len=len(X)
        self.x = [item for item in X]
        self.y = [item for item in Y]


    def __len__(self):
        # this should return the size of the dataset
        return self.data_len

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()


        return self.x[idx], self.y[idx]








def create_loader(data_set,batch_size, shuffle, drop_last):
    return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)





