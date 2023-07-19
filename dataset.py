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
from constants import Constants, model_constants



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
    dirs=['/y/', '/ev_y/', '/f_x/', '/ev_x/','/f_circle/', '/f_polygon/', '/output/']
    out_path=Constants.path+train_or_test
    data=[]
    for dir in dirs:
        data.append(extract_path_from_dir(out_path+dir  ))
    return [data[0], data[1], data[2], data[3], data[4], data[5]], [data[6] ]   



class SonarDataset(Dataset):
    def __init__(self, X, y):
        #  X is list of length num-inputs. each item in the list is a list of file names.

        # self.x = [[torch.load(name) for name in X[k]] for k in range(len(X))]
        # self.y = [[torch.load(name) for name in y[k]] for k in range(len(y))]

        self.x = [[name for name in X[k]] for k in range(len(X))]
        self.y = [[name for name in y[k]] for k in range(len(y))]
         
    def __len__(self):
        # this should return the size of the dataset
        assert len(self.x[0])==len(self.y[0])
        return len(self.x[0])

 
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
             idx = idx.tolist()

        # print(self.x[5][0].shape)
        # return [self.x[k][idx] for k in range(len(self.x))], [self.y[k][idx] for k in range(len(self.y))]
        return [torch.load(self.x[k][idx]) for k in range(len(self.x))], [torch.load(self.y[k][idx]) for k in range(len(self.y))]

    
        
        # return self.x1[idx], self.x2[idx][-Constants.ev_per_polygon:], self.x3[idx], self.x4[idx][-Constants.ev_per_polygon:], self.x5[idx], self.x6[idx], self.y[idx]
        # return torch.load(self.x1[idx]), torch.load(self.x2[idx])[-Constants.ev_per_polygon:], torch.load(self.x3[idx]), torch.load(self.x4[idx])[-Constants.ev_per_polygon:], torch.load(self.y[idx])
   
    

input, output   =load_data_names('train')
my_dataset = SonarDataset(input, output)


train_size = int(0.7 * len(my_dataset))
val_size = len(my_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, [train_size, val_size])
val_dataloader = DataLoader (val_dataset, batch_size=Constants.batch_size, shuffle=True)
train_dataloader = DataLoader (train_dataset, batch_size=Constants.batch_size, shuffle=True)


input, output  =load_data_names('test')
test_dataset = SonarDataset(input, output)
test_dataloader = DataLoader (test_dataset, batch_size=Constants.batch_size, shuffle=False)

# input,output=next(iter(train_dataloader))
# model_constants.dim=[input[k].shape[1] for k in range(len(input))]
# print(model_constants.dim)






# y, ev_y, f_x, ev_x, output  =load_data_names('hints')

# hints_dataset = SonarDataset([y, ev_y, f_x, ev_x], output)
# hints_dataloader = DataLoader (hints_dataset, batch_size=len(y), shuffle=False)

# print(len(hints_dataloader))





