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


class branc_point:
    
    def __init__(self,f, main_polygons):
        self.f=f
        self.main_polygons=main_polygons
        self.b1, self.b2= self.calculate_branch()

    def calculate_branch(self):
        x=[]
        y=[]
        Constants.pts_per_polygon=np.min([len(p['points']) for p in self.main_polygons])
        for p in self.main_polygons:
         #  print(len(p['points']))
          x.append(list(map(self.f, p['points'][:Constants.pts_per_polygon,0],p['points'][:Constants.pts_per_polygon,1]))  )
          y.append(p['eigen'])
        x=np.hstack(x).reshape((len(x), len(x[0])))
        y=np.hstack(y).reshape((len(y), len(y[0])))

        return x.transpose(), y.transpose()  

def create_main_polygons(dir_path):
   x=[]

   for filename in os.listdir(dir_path):
        f = os.path.join(dir_path, filename)

        if os.path.isfile(f) and  f.endswith('.pkl'):
           
           df=extract_pickle(f)
           x.append(df)
   return x        

        
def create_data_points(dir_train, dir_main_polygons):
    input1=[]
    input2=[]
    input3=[]
    input4=[]
    output=[]
    main_polygons=create_main_polygons(dir_main_polygons)

    for filename in os.listdir(dir_train):
        f = os.path.join(dir_train, filename)
        if os.path.isfile(f) and f.endswith('.pkl'):
           df=extract_pickle(f)
           for i in range(df['points'].shape[0]):
             
             input1.append(df['points'][i].reshape([Constants.dim,1])  )
             input2.append(df['eigen'].reshape([Constants.ev_per_polygon,1]) )
             input3.append(branc_point(df['gauss'], main_polygons).b1)
             input4.append(branc_point(df['gauss'],main_polygons).b2)
             output.append(df['u'][i])
    return input1, input2, input3, input4, output          

y, ev_y, f_x, ev_x, output  =create_data_points(Constants.path+'train',Constants.path+'main_polygons')



num_data=len(y)
indices_batches=create_batches(num_data, Constants.batch_size)


batched_data1=[]
batched_data2=[]
batched_data3=[]
batched_data4=[]
batched_output=[]
for batch_index in indices_batches: 
  batched_data1.append(torch.tensor(np.array([y[k] for k in batch_index]), dtype=torch.float32))
  batched_data2.append(torch.tensor(np.array([ev_y[k] for k in batch_index]), dtype=torch.float32))
  batched_data3.append(torch.tensor(np.array([f_x[k] for k in batch_index]), dtype=torch.float32))
  batched_data4.append(torch.tensor(np.array([ev_x[k] for k in batch_index]), dtype=torch.float32))
  batched_output.append(torch.tensor(np.array([output[k] for k in batch_index]), dtype=torch.float32))


class branch(nn.Module):
    def __init__(self, n, p):
      super().__init__()
      self.linear=nn.Linear(in_features=n*n, out_features=p, bias=True)
      self.activation=torch.nn.ReLU()
      self.n=n
    def forward(self,x):
         
         s=torch.matmul(x, torch.transpose(x,1,2))
        

         return self.activation(self.linear(torch.flatten(s,start_dim=1)))

         

class trunk(nn.Module):
    def __init__(self, n, p):
      super().__init__()
      self.linear=nn.Linear(in_features=n*n, out_features=p, bias=True)
      self.activation=torch.nn.ReLU()
    def forward(self,x):
         s=torch.matmul(x, torch.transpose(x,1,2))
         return self.activation(self.linear(torch.flatten(s,start_dim=1)))




class deeponet(nn.Module):
    def __init__(self, pts_per_polygon, ev_per_polygon, dim, p):
      super().__init__()
      self.branch1=branch(pts_per_polygon,p)
      self.branch2=branch(ev_per_polygon,p)
      
      self.trunk1=trunk(dim,p)
      self.trunk2=trunk(ev_per_polygon,p)
      self.loss=torch.nn.MSELoss()

    def forward(self,x,lx,y,ly):
       s1=torch.cat(( self.trunk1(y),self.trunk2(ly)), dim=-1)
       s2=torch.cat(( self.branch1(x),self.branch2(lx)), dim=-1)
     
       return torch.sum(s1*s2, dim=-1)

p=40
dim=Constants.dim
num_ctrl_polygons=2
pts_per_polygon=Constants.pts_per_polygon
ev_per_polygon=Constants.ev_per_polygon

# x1=torch.randn(4,pts_per_polygon, num_ctrl_polygons)

# l1=torch.randn(4,ev_per_polygon, 7)


# y1=torch.randn(4,dim, 1)

# ly=torch.randn(4,ev_per_polygon, 1)

model=deeponet(pts_per_polygon, ev_per_polygon, dim, p)

optimizer=optim.Adam(model.parameters(), lr=0.01)
loss_tot=[]
loss_val_tot=[]
for epoch in range (Constants.num_epochs) :
   for i in range(len(indices_batches)-1):
      y_batch=batched_output[i]
      y_pred=model(batched_data3[i],batched_data4[i],batched_data1[i],batched_data2[i])
      loss = model.loss(y_pred, y_batch)
      optimizer.zero_grad()
      loss.backward()
        # update weights
      optimizer.step()
      loss_tot.append(loss.item())
      
   with torch.no_grad(): # validation step
      y_batch=batched_output[-1]
      y_pred=model(batched_data3[-1],batched_data4[-1],batched_data1[-1],batched_data2[-1])
      loss_val = model.loss(y_pred, y_batch)   
      loss_val_tot.append(loss_val.item())   
         
   
  
# plt.plot(loss_tot)
# plt.plot(loss_val_tot, 'red')
# plt.show()  
# print(count_trainable_params(model))
