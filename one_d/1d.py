import numpy as np
import scipy
from scipy.linalg import circulant
import matplotlib.pyplot as plt
import os
import sys
import torch
from one_d_data_set import *
from one_d_model import deeponet
sys.path.append(os.getcwd())

from constants import Constants

def create_D2(x):
    Nx = len(x[1:-1])
    dx = x[1] - x[0]
    kernel = np.zeros((Nx, 1))
    kernel[-1] = 1
    kernel[0] = -2
    kernel[1] = 1
    D2 = circulant(kernel)
    D2[0, -1] = 0
    D2[-1, 0] = 0

    return scipy.sparse.csr_matrix(D2/dx/dx)
x=np.linspace(-1,1,40)
M=create_D2(x)

ev,V=scipy.sparse.linalg.eigs(-M,k=6,return_eigenvectors=True,which="SR")
A = (-M - Constants.k* scipy.sparse.identity(M.shape[0]))
F=[V[:,i].real for i in range(V.shape[1])]
psi=[scipy.sparse.linalg.spsolve(A,b) for b in F]

def generate_sample(x):
     M=np.random.uniform(-1,1,len(F))
     input=np.array([M[i]*F[i] for i  in range(len(F))])
     output=np.array([M[i]*psi[i] for i  in range(len(F))])

     return np.sum(input, axis=0), np.sum(output, axis=0)

number_samples=10
X=[]
Y=[]
for _ in range(number_samples):
     s=generate_sample(x)
     for j,y in enumerate(list(x[1:-1])):
        X.append([torch.tensor(y, dtype=torch.float32),torch.tensor(s[0], dtype=torch.float32)])
        Y.append(torch.tensor(s[1][j], dtype=torch.float32))
   

my_dataset = SonarDataset(X, Y)
train_size = int(0.8 * len(my_dataset))
val_size = len(my_dataset) - train_size





train_dataset, val_dataset = torch.utils.data.random_split(
    my_dataset, [train_size, val_size]
)
test_dataset = val_dataset
val_dataloader=create_loader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
train_dataloader = create_loader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
test_dataloader=create_loader(test_dataset, batch_size=100, shuffle=False, drop_last=True)
     

model=deeponet( 1, 38, 100)

# model([X[0].to(Constants.device),X[1].to(Constants.device)])