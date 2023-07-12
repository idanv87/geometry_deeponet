import os
import pickle
from random import sample
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.optim as optim
import numpy as np

from utils import *
# from dataset import train_dataloader, train_dataset, val_dataset, val_dataloader, test_dataloader, test_dataset
from tqdm import tqdm
import argparse
import time


def validate(model, dataloader, dataset, criterion):
    # print('Validating')
    model.eval()
    val_running_loss = 0.0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size))
    with torch.no_grad():
        for i, data in prog_bar:
            counter += 1
            x1,x2,x3,x4,output=data
            x1,x2,x3,x4,output = x1.to(Constants.device), x2.to(Constants.device),x3.to(Constants.device),x4.to(Constants.device),output.to(Constants.device)
            total += output.size(0)
            outputs = model(x3,x4,x1,x2)
            loss = criterion(outputs, output)
            
            val_running_loss += loss.item()
            # _, preds = torch.max(outputs.data, 1)
            # val_running_correct += (preds == output).sum().item()
        
        val_loss = val_running_loss / counter
        
        return val_loss




class branch(nn.Module):
    def __init__(self, n, p):
      super().__init__()
      self.linear=nn.Linear(in_features=n*n, out_features=p, bias=True)
      self.activation=torch.nn.Tanh()
      self.n=n
    def forward(self,x):
         
         s=torch.matmul(x, torch.transpose(x,1,2))
         s=torch.flatten(s,start_dim=1)
         return self.activation(self.linear(s))

         

class trunk(nn.Module):
    def __init__(self, n, p):
      super().__init__()
      self.linear=nn.Linear(in_features=n*n, out_features=p, bias=True)
      self.activation=torch.nn.Tanh()
    def forward(self,x):
         s=torch.matmul(x, torch.transpose(x,1,2))
         s=torch.flatten(s,start_dim=1)
         self.activation(self.linear(s))
         return self.activation(self.linear(s))
    





class deeponet(nn.Module):
    def __init__(self, pts_per_polygon, ev_per_polygon, dim, p):
      super().__init__()
      self.branch1=branch(pts_per_polygon,p)
      self.branch2=branch(ev_per_polygon,p)
      
      self.trunk1=trunk(dim,p)
      self.trunk2=trunk(ev_per_polygon,p)

    def forward(self,x,lx,y,ly):
      #  s1=self.trunk1(y)
      #  s2=self.trunk2(ly)
       s1=torch.cat(( self.trunk1(y),self.trunk2(ly/10)), dim=-1)
       s2=torch.cat(( self.branch1(x),self.branch2(lx/10)), dim=-1)
     
       return torch.sum(s1*s2, dim=-1)





p=2
dim=Constants.dim
pts_per_polygon=Constants.pts_per_polygon
ev_per_polygon=Constants.ev_per_polygon
model=deeponet(pts_per_polygon, ev_per_polygon, dim, p)
# best_model=torch.load(Constants.path+'best_model/'+'best.pth')
# model.load_state_dict(best_model['model_state_dict'])
# print(count_trainable_params(model))

if __name__=='__main__':
   pass
   

   # best_model=torch.load(Constants.path+'best_model/'+'2023-07-10_10.00.42.817019.pth')
   # model.load_state_dict(best_model['model_state_dict'])
   # model.eval()
   # from dataset import train_dataloader

   # test_epoch_loss = validate(
   #      model, test_dataloader, test_dataset, torch.nn.MSELoss()
   #  )
   # print(test_epoch_loss)
# def func(x,y):
#     return (x-np.sqrt(math.pi)/2)*(x+np.sqrt(math.pi)/2)*(y-np.sqrt(math.pi)/2)*(y+np.sqrt(math.pi)/2)
   
# for j,l in enumerate(train_dataloader):
#        X=l[0]
#        U=l[-1]
#        x=[X[i,0] for i in range(X.shape[0])]
#        y=[X[i,1] for i in range(X.shape[0])]
#        u=np.array([U[i] for i in range(U.shape[0])])
#        print(u)

# #       
# #        u_an=np.array(list(map(func, x,y)))

# print(u)

       
      #  plt.scatter(x,y)
      #  plt.show()

# print(type(test_dataloader))



# print(count_trainable_params(model))







# for i, data in enumerate(train_dataloader):
#    x1,x2,x3,x4,output=data
#    print(model(x3,x4,x1,x2))


