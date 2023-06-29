import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *


class branch(nn.Module):
    def __init__(self, n,m):
      super().__init__()
      self.fc = nn.Linear(in_features=n, out_features=m)
      self.c=nn.Linear(in_features=self.fc.out_features, out_features=1, bias=False)
      self.activation=torch.relu
    def forward(self,x):
       return self.c(self.activation(self.fc(x)))


class trunk(nn.Module):
    def __init__(self, n,m):
      super().__init__()
      self.fc = nn.Linear(in_features=n, out_features=m)
      self.activation=torch.relu
    def forward(self,x):
       return self.activation(self.fc(x))


class deeponet(nn.Module):
    def __init__(self, num_branches, y_dim, u_dim, n):
      super().__init__()
      self.branches=nn.ModuleList([])
      self.trunk=nn.ModuleList([])
      for i in range(num_branches):
         self.branches.append(branch(u_dim,n))
         self.trunk.append(trunk(y_dim,1))
    def forward(self,x,y):
       sol=0
       for i in range(num_branches):
          sol=self.branches[i](x)*self.trunk[i](y)
       return sol
    
# x=np.random.rand(10,2)
# x=torch.tensor(x, dtype=torch.float64)
num_branches=2
u_dim=10
y_dim=2
x=torch.rand((4,1,u_dim))
y=torch.rand((4,1,y_dim))
m=deeponet(num_branches,y.shape[2],x.shape[2],3)
m(x,y)
# count_trainable_params(m)



# b=branch(x.shape[2],3)
# t=trunk(y.shape[2],1)

# print(t(y).shape)
# print(b(x).shape)

# my_nn = Net()
# model_parameters = filter(lambda p: p.requires_grad, my_nn.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)