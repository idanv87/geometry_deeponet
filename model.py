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
from geometry import circle

from utils import *


# initializer=nn.init.zeros_

class fc(torch.nn.Module):
    def __init__(self, input_shape, output_shape, num_layers):
        super().__init__() 
        self.input_shape=input_shape
        self.output_shape=output_shape
        n=40
        self.activation=torch.nn.ReLU()
        self.layers=torch.nn.ModuleList([torch.nn.Linear(in_features=self.input_shape,out_features=n,bias=True)])
        output_shape=n

        for j  in range(num_layers):
            layer=torch.nn.Linear(in_features=output_shape,out_features=n,bias=True)
            # initializer(layer.weight)
            output_shape=n
            self.layers.append(layer)
        
        self.layers.append(torch.nn.Linear(in_features=output_shape,out_features=self.output_shape,bias=True))


    def forward(self,y):
        s=torch.squeeze(y)
        for layer in self.layers:
            s=layer(self.activation(s))


        return s
    

class deeponet(nn.Module):
    def __init__(self, dim, num_hot_spots, pts_per_circle, ev_per_polygon, p):
        super().__init__()
        self.branch=fc(num_hot_spots,p,4)
        self.trunk=fc(dim,p,4)

    def forward(self, input):
        y, ly, f_circle, f_polygon = input
        s1=self.branch(f_polygon)
        s2=self.trunk(y)
        if len(s1.size())==1:
            return [torch.squeeze(torch.bmm(s1.view(1, 1, s1.shape[0]),
                          s2.view(1, s2.shape[0], 1)
                           ))]
        else:    
            return [torch.squeeze(torch.bmm(s1.view(s1.shape[0], 1, s1.shape[1]),
                          s2.view(s2.shape[0], s2.shape[1], 1)
                           ))]

    
p = 60
dim = Constants.dim
num_hot_spots = int((int(1/Constants.h)-2)**2/(Constants.hot_spots_ratio**2))
pts_per_circle=len(circle().hot_points)
ev_per_polygon = Constants.ev_per_polygon



model = deeponet(dim, num_hot_spots, pts_per_circle, ev_per_polygon, p)

# best_model=torch.load(Constants.path+'best_model/'+'best.pth')
# model.load_state_dict(best_model['model_state_dict'])
print("number of model parameters: " + str(count_trainable_params(model)))

if __name__ == "__main__":

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


# def validate(model, dataloader, dataset, criterion):
#     # print('Validating')
#     model.eval()
#     val_running_loss = 0.0
#     counter = 0
#     total = 0
#     prog_bar = tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size))
#     with torch.no_grad():
#         for i, data in prog_bar:
#             counter += 1
#             x1,x2,x3,x4,output=data
#             x1,x2,x3,x4,output = x1.to(Constants.device), x2.to(Constants.device),x3.to(Constants.device),x4.to(Constants.device),output.to(Constants.device)
#             total += output.size(0)
#             outputs = model(x3,x4,x1,x2)
#             loss = criterion(outputs, output)

#             val_running_loss += loss.item()
#             # _, preds = torch.max(outputs.data, 1)
#             # val_running_correct += (preds == output).sum().item()

#         val_loss = val_running_loss / counter

#         return val_loss
