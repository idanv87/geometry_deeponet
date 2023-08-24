import numpy as np
import scipy
from scipy.linalg import circulant
from scipy.sparse import  kron, identity, csr_matrix
from scipy.stats import qmc
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import torch
from two_d_data_set import *
from two_d_model import deeponet, geo_deeponet
from draft import create_data, expand_function

current_path=os.path.abspath(__file__)
sys.path.append(current_path.split('deeponet')[0]+'deeponet/')

from utils import count_trainable_params, extract_path_from_dir, save_uniqe
from constants import Constants


train_names=extract_path_from_dir(Constants.path+'polygons/')[0:2]
test_names=[train_names[0]]
hint_names=[train_names[0]]
      
def plot_surface(xi,yi,Z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xi, yi, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()


def generate_sample(M, F, F_hot, psi):
    #  np.random.uniform(-10,10,len(F))
     x1=np.array([M[i]*F[i] for i  in range(len(F))])
     x2=np.array([M[i]*psi[i] for i  in range(len(F))])
     x3=np.array([M[i]*F_hot[i] for i  in range(len(F_hot))])
     return np.sum(x1, axis=0), np.sum(x2, axis=0), np.sum(x3, axis=0)


def generate_data(names,  save_path, number_samples=10):
    X=[]
    Y=[]
    for name in names:
        domain=torch.load(name)
        xi,yi,F, F_hot,psi, moments_x, moments_y, angle_fourier=create_data(domain)
        sampler = qmc.Halton(d=len(F), scramble=False)
        sample = 20*sampler.random(n=number_samples)-10
        for i in range(number_samples):
            s0,s1, s_hot=generate_sample(sample[i], F, F_hot,psi)

            a=expand_function(s0, domain)
            #  plot_surface(xi.reshape(18,18),yi.reshape(18,18),F[12].reshape(18,18))
            for j in range(len(xi)):
                X1=[
                    torch.tensor([xi[j],yi[j]], dtype=torch.float32),
                    torch.tensor(a, dtype=torch.float32),
                    torch.tensor(0, dtype=torch.float32),
                    torch.tensor(0, dtype=torch.float32),
                    torch.tensor(angle_fourier, dtype=torch.float32)
                    ]
                Y1=torch.tensor(s1[j], dtype=torch.float32)
                save_uniqe([X1,Y1],save_path)
                X.append(X1)
                Y.append(Y1)
    return X,Y        

X,Y=generate_data(train_names, Constants.train_path, 20)
X_test, Y_test=generate_data(test_names,Constants.test_path,1)

train_data=extract_path_from_dir(Constants.train_path)
test_data=extract_path_from_dir(Constants.test_path)
X_train=[torch.load(f)[0] for f in train_data]
Y_train=[torch.load(f)[1] for f in train_data]
X_test=[torch.load(f)[0] for f in test_data]
Y_test=[torch.load(f)[1] for f in test_data]

train_dataset = SonarDataset(X_train, Y_train)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)
test_dataset = SonarDataset(X_test, Y_test)
val_dataloader=create_loader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
train_dataloader = create_loader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
test_dataloader=create_loader(test_dataset, batch_size=4, shuffle=False, drop_last=True)

inp, out=next(iter(test_dataset))
model=geo_deeponet( 2, inp[1].shape[0],inp[4].shape[0], 100)
inp, out=next(iter(train_dataloader))

model(inp)
print(f" num of model parameters: {count_trainable_params(model)}")
# model([X[0].to(Constants.device),X[1].to(Constants.device)])

