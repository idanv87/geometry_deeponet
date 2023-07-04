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

polygons_dir=Constants.path+'polygons/'
polygons_raw_names=next(os.walk(polygons_dir), (None, None, []))[2]
polygons_files_names=[n for n in polygons_raw_names if n.endswith('.pt')]
polygon_pathes=[polygons_dir+p for p in polygons_files_names]
indices=list(np.random.permutation(len(polygons_files_names)))
control_polygons=list(map(polygons_files_names.__getitem__, indices[:Constants.num_control_polygons]))
train_polygons=list(map(polygons_files_names.__getitem__, indices[Constants.num_control_polygons:]))


#plot polygon:
p=torch.load(polygon_pathes[0])  
pol=Polygon(p['X'],p['cells'], p['generators'])
pol.plot_polygon()
            
# dmsh.show(p['X'],p['cells'], dmsh.Polygon(p['generators']))



class branc_point:
    
    def __init__(self,f, main_polygons):
        self.f=f
        self.main_polygons=main_polygons
        self.b1, self.b2= self.calculate_branch()

    def calculate_branch(self):
        x=[]
        y=[]
   
        for p in self.main_polygons:
          x_interior_points=spread_points(Constants.pts_per_polygon, p['points'])
          ind_x=x_interior_points[:, 0].argsort()
          p['x_points']=x_interior_points[ind_x]
         #  plt.scatter(p['points'][:,0], p['points'][:,1], color='black')
         #  plt.scatter(p['x_points'][:,0], p['x_points'][:,1], color='red')
         #  plt.show()
         
          x.append(list(map(self.f, p['x_points'][:,0],p['x_points'][:,1]))  )
          y.append(p['eigen'])
        x=np.hstack(x).reshape((len(x), len(x[0])))
        y=np.hstack(y).reshape((len(y), len(y[0])))

        return x.transpose(), y.transpose()  

def create_main_polygons(control_polygons, polygons_dir):
   x=[]

   for filename in control_polygons:
        f = os.path.join(polygons_dir, filename)

        if os.path.isfile(f):
           
           df=torch.load(f)
           x.append(df)
   return x        

        
def create_data_points(train_polygons,control_polygons, polygons_dir):
    data_names=[]
    main_polygons=create_main_polygons(control_polygons, polygons_dir)

    for filename in train_polygons:
        f = os.path.join(polygons_dir, filename)
        if os.path.isfile(f):
           
           df=torch.load(f)
           for i in range(df['points'].shape[0]):
             y=df['points'][i].reshape([Constants.dim,1])
             ev_y=df['eigen'].reshape([Constants.ev_per_polygon,1])
             f_x=branc_point(df['gauss'], main_polygons).b1
             ev_x=branc_point(df['gauss'],main_polygons).b2
             output=df['u'][i]
             name= str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
             
             data_names.append(name+'.pt')
             
             save_file(np_to_torch(y),Constants.path+'y/', name)
             save_file(np_to_torch(ev_y),Constants.path+'ev_y/', name)
             save_file(np_to_torch(f_x),Constants.path+'f_x/', name)
             save_file(np_to_torch(ev_x),Constants.path+'ev_x/', name)
             save_file(np_to_torch(output),Constants.path+'output/', name)
             save_file(data_names,Constants.path+'data_names/','data_names')

             
            #  input1.append(torch.tensor(df['points'][i].reshape([Constants.dim,1]),dtype=Constants.dtype  ))
            #  input2.append(torch.tensor(df['eigen'].reshape([Constants.ev_per_polygon,1]),dtype=Constants.dtype ))
            #  input3.append(torch.tensor(branc_point(df['gauss'], main_polygons).b1, dtype=Constants.dtype))
            #  input4.append(torch.tensor(branc_point(df['gauss'],main_polygons).b2, dtype=Constants.dtype))
            #  out.append(torch.tensor(df['u'][i], dtype=Constants.dtype))
    
    return         
# create_data_points(train_polygons,control_polygons, polygons_dir)

def load_data(dir=['y/', 'ev_y/', 'f_x/', 'ev_x/', 'output/']):
    filenames = torch.load(Constants.path+'data_names/data_names.pt')

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
        return self.x1[idx], self.x2[idx], self.x3[idx], self. x4[idx], self.y[idx]
    

y, ev_y, f_x, ev_x, output  =load_data()
my_dataset = SonarDataset([y, ev_y, f_x, ev_x], output)

train_size = int(0.7 * len(my_dataset))
test_size = len(my_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])
test_dataloader = DataLoader (test_dataset, batch_size=Constants.batch_size, shuffle=True)

my_dataset = train_dataset
train_size = int(0.8 * len(my_dataset))
val_size = len(my_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, [train_size, val_size])
val_dataloader = DataLoader (val_dataset, batch_size=Constants.batch_size, shuffle=True)

train_dataloader = DataLoader (train_dataset, batch_size=Constants.batch_size, shuffle=True)

# print(len(my_dataset))
# print(output.shape)
# torch.save(my_dataset, Constants.path+'data_sets/my_dataset.pt')
# train_dataset = torch.load(Constants.path+'data_sets/my_dataset.pt')


# for i, data in enumerate(train_loader):
#     x1,x2,x3,x4,y=data
#     print(x1)
#     print(x2)
#     print(y)


