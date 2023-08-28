import numpy as np
import scipy
from scipy.linalg import circulant
import math
import matplotlib.pyplot as plt
import os
import sys
import torch
from one_d_data_set import *
from one_d_model import deeponet
from scipy.stats import qmc

sys.path.append(os.getcwd())
from utils import count_trainable_params
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
x=np.linspace(-1,1,50)
domain=x
L=create_D2(x)

# ev,V=scipy.sparse.linalg.eigs(-L,k=20,return_eigenvectors=True,which="SR")
V1=np.array([((1-x[1:-1])*(1+x[1:-1]))*scipy.special.legendre(j)(x[1:-1])for j in range(20)]).T
V2=np.array([np.sin(j*math.pi*x[1:-1])for j in range(20)]).T
V3=np.array([((1-x[1:-1])*(1+x[1:-1]))*scipy.special.chebyc(j)(x[1:-1])for j in range(20)]).T
V=np.hstack((V1,V2,V3))
F=[V[:,i].real for i in range(V.shape[1])]
# V=np.array([np.sin(math.pi*(j+1)*x[1:-1]) for j in range(5)]).T
A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))

psi=[scipy.sparse.linalg.spsolve(A,b) for b in F]

# F=F+[b-A@g for g,b in zip(psi,F)]
# psi=[scipy.sparse.linalg.spsolve(A,b) for b in F]
# F=F+psi
# psi=[scipy.sparse.linalg.spsolve(A,b) for b in F]

def generate_sample(M=np.random.uniform(-1,1,len(F))):

     
     x1=np.array([M[i]*F[i] for i  in range(len(F))])
     x2=np.array([M[i]*psi[i] for i  in range(len(F))])


     return np.sum(x1, axis=0), np.sum(x2, axis=0)

number_samples=300
sampler = qmc.Halton(d=len(F), scramble=False)
sample = 2*sampler.random(n=number_samples)-1
X=[]
Y=[]
X_test=[]
Y_test=[]
for i in range(number_samples):

     s0,s1=generate_sample(sample[i])
     for j,y in enumerate(list(x[1:-1])):
        X.append([torch.tensor(y, dtype=torch.float32),torch.tensor(s0, dtype=torch.float32)])
        Y.append(torch.tensor(s1[j], dtype=torch.float32))

# s0,s1=generate_sample()
for j,y in enumerate(list(x[1:-1])):
           
            # s0=np.sin(math.pi*domain[1:-1])
            # s1=scipy.sparse.linalg.spsolve(A,s0) 
            s0,s1=generate_sample()
            X_test.append([torch.tensor(y, dtype=torch.float32),torch.tensor(s0, dtype=torch.float32)])
            Y_test.append(torch.tensor(s1[j], dtype=torch.float32))
   

# plt.plot([z[0] for z in X_test],Y_test)
# plt.show()

my_dataset = SonarDataset(X, Y)
train_size = int(0.8 * len(my_dataset))
val_size = len(my_dataset) - train_size





train_dataset, val_dataset = torch.utils.data.random_split(
    my_dataset, [train_size, val_size]
)
test_dataset = SonarDataset(X_test, Y_test)
val_dataloader=create_loader(val_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
train_dataloader = create_loader(train_dataset, batch_size=Constants.batch_size, shuffle=True, drop_last=True)
test_dataloader=create_loader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
     

model=deeponet( 1, x.shape[0]-2, 80)
print(f" num of model parameters: {count_trainable_params(model)}")
# model([X[0].to(Constants.device),X[1].to(Constants.device)])