import torch
from torchvision import transforms, datasets
import os
import pickle
from random import sample

import torch
import numpy as np
from utils import *
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


# def load_data(train_or_test):
#     dir=['y/', 'ev_y/', 'f_x/', 'ev_x/', 'output/']
#     if train_or_test=='train':
#       filenames = torch.load(Constants.path+'train_data_names/train_data_names.pt')
#     else:  
#         filenames = torch.load(Constants.path+'test_data_names/test_data_names.pt')
       

#     y=[torch.load(Constants.path+'y/'+name) for name in filenames]
#     ev_y=[torch.load(Constants.path+'ev_y/'+name) for name in filenames]
#     f_x=[torch.load(Constants.path+'f_x/'+name) for name in filenames]
#     ev_x=[torch.load(Constants.path+'ev_x/'+name) for name in filenames]
#     output=[torch.load(Constants.path+'output/'+name) for name in filenames]
#     return torch.stack(y), torch.stack(ev_y), torch.stack(f_x), torch.stack(ev_x), torch.stack(output)

# add_new_polygon()
def load_data_names(train_or_test):
    dirs=['/y/', '/ev_y/', '/f_x/', '/ev_x/', '/output/']
    out_path=Constants.path+train_or_test
    data=[]
    for dir in dirs:
        data.append(extract_path_from_dir(out_path+dir  ))
    return data[0], data[1], data[2], data[3], data[4]    



class SonarDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.x1 = X[0]
        self.x2 =X[1]
        self.x3 = X[2]
        self.x4 = X[3]
        self.y = y

        self.x1 = [torch.load(name) for name in X[0]]
        self.x2 = [torch.load(name) for name in X[1]]
        self.x3 = [torch.load(name) for name in X[2]]
        self.x4 = [torch.load(name) for name in X[3]]
        self.y = [torch.load(name) for name in y]
        
 
    def __len__(self):
        # this should return the size of the dataset
        return len(y)

 
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
             idx = idx.tolist()
        
        return self.x1[idx], self.x2[idx][-Constants.ev_per_polygon:], self.x3[idx], self.x4[idx][-Constants.ev_per_polygon:], self.y[idx]
        # return torch.load(self.x1[idx]), torch.load(self.x2[idx])[-Constants.ev_per_polygon:], torch.load(self.x3[idx]), torch.load(self.x4[idx])[-Constants.ev_per_polygon:], torch.load(self.y[idx])
   
    

y, ev_y, f_x, ev_x, output  =load_data_names('train')
my_dataset = SonarDataset([y, ev_y, f_x, ev_x], output)


train_size = int(0.7 * len(my_dataset))
val_size = len(my_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, [train_size, val_size])
val_dataloader = DataLoader (val_dataset, batch_size=Constants.batch_size, shuffle=True)
train_dataloader = DataLoader (train_dataset, batch_size=Constants.batch_size, shuffle=True)


y, ev_y, f_x, ev_x, output  =load_data_names('test')
test_dataset = SonarDataset([y, ev_y, f_x, ev_x], output)
test_dataloader = DataLoader (test_dataset, batch_size=Constants.batch_size, shuffle=False)

# y, ev_y, f_x, ev_x, output  =load_data_names('hints')

# hints_dataset = SonarDataset([y, ev_y, f_x, ev_x], output)
# hints_dataloader = DataLoader (hints_dataset, batch_size=len(y), shuffle=False)

# print(len(hints_dataloader))





