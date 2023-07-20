import os
from random import sample

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, Subset

from constants import Constants
from utils import *


def load_data_names(train_or_test):
    dirs = ["/y/", "/ev_y/", "/f_x/", "/ev_x/", "/f_circle/", "/f_polygon/", "/output/"]
    out_path = Constants.path + train_or_test
    data = []
    for dir in dirs:
        data.append(extract_path_from_dir(out_path + dir))
    return [data[0], data[1], data[2], data[3], data[4], data[5]], [data[6]]


class SonarDataset(Dataset):
    def __init__(self, X, y):

        self.load_type = False

        #  X is list of length num-inputs. each item in the list is a list of file names.
        if self.load_type:
            self.x = [[torch.load(name) for name in X[k]] for k in range(len(X))]
            self.y = [[torch.load(name) for name in y[k]] for k in range(len(y))]
        else:
            self.x = [[name for name in X[k]] for k in range(len(X))]
            self.y = [[name for name in y[k]] for k in range(len(y))]

    def __len__(self):
        # this should return the size of the dataset
        assert len(self.x[0]) == len(self.y[0])
        return len(self.x[0])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.load_type:
            return [self.x[k][idx] for k in range(len(self.x))], [
                self.y[k][idx] for k in range(len(self.y))
            ]
        else:
            return [torch.load(self.x[k][idx]) for k in range(len(self.x))], [
                torch.load(self.y[k][idx]) for k in range(len(self.y))
            ]


input, output = load_data_names("train")
my_dataset = SonarDataset(input, output)

train_size = int(0.7 * len(my_dataset))
val_size = len(my_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    my_dataset, [train_size, val_size]
)

val_dataloader = DataLoader(val_dataset, batch_size=Constants.batch_size, shuffle=True)
train_dataloader = DataLoader(
    train_dataset, batch_size=Constants.batch_size, shuffle=True
)


input, output = load_data_names("test")
test_dataset = SonarDataset(input, output)
test_dataloader = DataLoader(
    test_dataset, batch_size=Constants.batch_size, shuffle=False
)

# input,output=next(iter(test_dataloader))
# print('\n single input data-point dimensions:')
# print([inp[0].shape for inp in input])
# print('\n single output data-point dimensions:')
# print([out[0].shape for out in output])

# for input, output in train_dataloader:
#     print(output[0][7])
# model_constants.dim=[input[k].shape[1] for k in range(len(input))]
# print(model_constants.dim)


# y, ev_y, f_x, ev_x, output  =load_data_names('hints')

# hints_dataset = SonarDataset([y, ev_y, f_x, ev_x], output)
# hints_dataloader = DataLoader (hints_dataset, batch_size=len(y), shuffle=False)

# print(len(hints_dataloader))
