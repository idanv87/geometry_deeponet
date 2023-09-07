import matplotlib.pyplot as plt
import torch
import torch.nn as nn


import numpy as np
import os
import sys






class fc(torch.nn.Module):
    def __init__(self, input_shape, output_shape, num_layers, activation_last):
        super().__init__()
        self.activation_last=activation_last
        self.input_shape = input_shape
        self.output_shape = output_shape

        n = max(input_shape,20)
        self.activation = torch.nn.ELU()
        # self.activation = torch.nn.LeakyReLU()
        # self.activation = torch.nn.Tanh()

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=self.input_shape, out_features=n, bias=True)])
        output_shape = n

        for j in range(num_layers):
            layer = torch.nn.Linear(
                in_features=output_shape, out_features=n, bias=True)
            # initializer(layer.weight)
            output_shape = n
            self.layers.append(layer)

        self.layers.append(torch.nn.Linear(
            in_features=output_shape, out_features=self.output_shape, bias=True))

    def forward(self, y):
        s=y
        for layer in self.layers:
            s = layer(self.activation(s))
        if self .activation_last:
            return self.activation(s)
        else:
            return s


class deeponet(nn.Module):
    # good parameters: n_layers in deeponet=4,n_layers in geo_deeponet=10, infcn=100, ,n=5*p, p=100

    def __init__(self, dim, num_hot_spots, p):
        super().__init__()
        n_layers = 4
        self.n = p
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.branch1 = fc(num_hot_spots, self.n, n_layers,activation_last=False)
        self.trunk1 = fc(dim, p,  n_layers, activation_last=True)
        self.c_layer = fc( self.n, p, n_layers, activation_last=False)
        self.c2_layer =fc( num_hot_spots+dim, 1, n_layers, False) 


    def forward(self, X):
        y,f,m_x,m_y, angle=X
        branch = self.c_layer(self.branch1(f))
        trunk = self.trunk1(y)
        # alpha = torch.squeeze(self.c2_layer(torch.cat((f,y),dim=1)))
        return torch.sum(branch*trunk, dim=-1, keepdim=False)+self.alpha
        

class geo_deeponet(nn.Module):
    # good parameters: n_layers in deeponet=4,n_layers in geo_deeponet=10, infcn=100, ,n=5*p, p=100

    def __init__(self, dim, num_hot_spots,num_fourier_domain, p):
        super().__init__()
        n_layers = 4
        self.n = p
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.branch1 = fc(num_hot_spots, self.n, n_layers,activation_last=False)
        self.branch2= fc(num_fourier_domain, self.n, n_layers,activation_last=False)
        self.trunk1 = fc(dim, p,  n_layers, activation_last=True)
        self.c_layer = fc( 2*self.n, p, n_layers, activation_last=False)
        self.c2_layer =fc( num_fourier_domain, 1, n_layers, False) 


    def forward(self, X):
        y,f,m_x,m_y, angle=X
        branch = self.c_layer(  torch.cat((self.branch1(f), self.branch2(angle)), dim=1))
        trunk = self.trunk1(y)
        alpha=torch.squeeze(self.c2_layer(angle),dim=1)
        # alpha = torch.squeeze(self.c2_layer(torch.cat((f,y),dim=1)))
        return torch.sum(branch*trunk, dim=-1, keepdim=False)+alpha
        














