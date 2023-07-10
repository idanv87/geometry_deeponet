import os
import pickle
from random import sample

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils import *
# from dataset import train_dataloader, train_dataset, val_dataset, val_dataloader, test_dataloader, test_dataset
from tqdm import tqdm
import argparse
import time



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
print(count_trainable_params(model))







# for i, data in enumerate(train_dataloader):
#    x1,x2,x3,x4,output=data
#    print(model(x3,x4,x1,x2))


