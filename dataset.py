import torch
from torchvision import transforms, datasets
import os
import pickle
from random import sample

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
import dmsh

from utils import *
from data_generator import Polygon
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


from torch.utils.data import Dataset, Subset


#plot polygon:
# polygons_dir=Constants.path+'polygons/'
# polygon_pathes=[polygons_dir+p for p in polygons_files_names]
# p=torch.load(polygon_pathes[3])  
# pol=Polygon(p['X'],p['cells'], p['generators'])
# pol.plot_polygon()
            
# dmsh.show(p['X'],p['cells'], dmsh.Polygon(p['generators']))


def load_data(train_or_test):
    dir=['y/', 'ev_y/', 'f_x/', 'ev_x/', 'output/']
    if train_or_test=='train':
      filenames = torch.load(Constants.path+'train_data_names/train_data_names.pt')
    else:  
        filenames = torch.load(Constants.path+'test_data_names/test_data_names.pt')

    y=[torch.load(Constants.path+'y/'+name) for name in filenames]
    ev_y=[torch.load(Constants.path+'ev_y/'+name) for name in filenames]
    f_x=[torch.load(Constants.path+'f_x/'+name) for name in filenames]
    ev_x=[torch.load(Constants.path+'ev_x/'+name) for name in filenames]
    output=[torch.load(Constants.path+'output/'+name) for name in filenames]
    return torch.stack(y), torch.stack(ev_y), torch.stack(f_x), torch.stack(ev_x), torch.stack(output)





class SonarDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.x1 = X[0]
        self.x2 =X[1]
        self.x3 = X[2]
        self.x4 = X[3]
        self.y = y
        
 
    def __len__(self):
        # this should return the size of the dataset
        return self.y.shape[0]
 
    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx][-Constants.ev_per_polygon:], self.x3[idx], self.x4[idx][-Constants.ev_per_polygon:], self.y[idx]
    

y, ev_y, f_x, ev_x, output  =load_data('train')
my_dataset = SonarDataset([y, ev_y, f_x, ev_x], output)

train_size = int(0.8 * len(my_dataset))
val_size = len(my_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, [train_size, val_size])
val_dataloader = DataLoader (val_dataset, batch_size=Constants.batch_size, shuffle=True)
train_dataloader = DataLoader (train_dataset, batch_size=Constants.batch_size, shuffle=True)


y, ev_y, f_x, ev_x, output  =load_data('test')
test_dataset = SonarDataset([y, ev_y, f_x, ev_x], output)
test_dataloader = DataLoader (test_dataset, batch_size=Constants.batch_size, shuffle=False)



# print(len(my_dataset))
# print(output.shape)
# torch.save(my_dataset, Constants.path+'data_sets/my_dataset.pt')
# train_dataset = torch.load(Constants.path+'data_sets/my_dataset.pt')


# for i, data in enumerate(train_loader):
#     x1,x2,x3,x4,y=data
#     print(x1)
#     print(x2)
#     print(y)


