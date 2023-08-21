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
from two_d_model import deeponet
from draft import create_data, expand_function

current_path=os.path.abspath(__file__)
sys.path.append(current_path.split('deeponet')[0]+'deeponet/')

from utils import count_trainable_params, extract_path_from_dir
from constants import Constants
names=list(set(extract_path_from_dir(current_path.split('deeponet')[0]+'data_deeponet/polygons/'))-set(
  '/Users/idanversano/Documents/clones/data_deeponet/polygons/rect.pt'   
)
)
names=[names[0]] 
      
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# create train data

X=[]
Y=[]

for name in names:

    domain=torch.load(name)
    xi,yi,F, F_hot,psi, moments_x, moments_y=create_data(domain)
    
    number_samples=100
    sampler = qmc.Halton(d=len(F), scramble=False)
    sample = 20*sampler.random(n=number_samples)-10
    for i in range(number_samples):
        s0,s1, s_hot=generate_sample(sample[i], F, F_hot,psi)

        # a=expand_function(s0, domain)
        #  plot_surface(xi.reshape(18,18),yi.reshape(18,18),F[12].reshape(18,18))
        for j in range(len(xi)):
            X.append([
                torch.tensor([xi[j],yi[j]], dtype=torch.float32),
                torch.tensor(s_hot, dtype=torch.float32),
                torch.tensor(moments_x, dtype=torch.float32),
                torch.tensor(moments_y, dtype=torch.float32),
                ])
            Y.append(torch.tensor(s1[j], dtype=torch.float32))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# torch.save(X,Constants.path+'X.pt')
# torch.save(Y,Constants.path+'Y.pt')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# create test data
X_test=[]
Y_test=[]
xi,yi,F, F_hot, psi,moments_x, moments_y=create_data(torch.load(names[0]))
number_samples=1
sampler = qmc.Halton(d=len(F), scramble=False)
sample = 20*sampler.random(n=number_samples)-10
for i in range(number_samples):
    s0,s1, s_hot=generate_sample(sample[i], F, F_hot, psi)
    try:
        # a=expand_function(s0, domain)
        #  plot_surface(xi.reshape(18,18),yi.reshape(18,18),F[12].reshape(18,18))
        for j in range(len(xi)):
                X_test.append([
                    torch.tensor([xi[j],yi[j]], dtype=torch.float32),
                    torch.tensor(s_hot, dtype=torch.float32),
                    torch.tensor(moments_x, dtype=torch.float32),
                    torch.tensor(moments_y, dtype=torch.float32),
                    ])        
                Y_test.append(torch.tensor(s1[j], dtype=torch.float32))
    except:
         pass            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# torch.save(X_test,Constants.path+'X_test.pt')
# torch.save(Y_test,Constants.path+'Y_test.pt')
my_dataset = SonarDataset(X, Y)
train_size = int(0.8 * len(my_dataset))
val_size = len(my_dataset) - train_size





train_dataset, val_dataset = torch.utils.data.random_split(
    my_dataset, [train_size, val_size]
)
test_dataset = SonarDataset(X_test, Y_test)
val_dataloader=create_loader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
train_dataloader = create_loader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
test_dataloader=create_loader(test_dataset, batch_size=4, shuffle=False, drop_last=True)

inp, out=next(iter(train_dataset))
print(inp[1].shape[0])


model=deeponet( 2, inp[1].shape[0], 100)
print(f" num of model parameters: {count_trainable_params(model)}")
# model([X[0].to(Constants.device),X[1].to(Constants.device)])

