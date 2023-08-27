import os
import sys
import math
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.utils.data import Dataset, DataLoader
import sys
from scipy.interpolate import Rbf



from utils import np_to_torch, count_trainable_params
from constants import Constants
from functions.functions import gaussian
from one_d.main import test_dataset, Y_test, L, SonarDataset, F, domain, generate_sample
from one_d.one_d_data_set import create_loader

class g(nn.Module):
    def __init__(self,A, model,b,factor): 
        super().__init__()
        self.model=model
        for param in self.model.parameters():
            param.requires_grad = False
        self.A=A
        self.y=torch.tensor(domain[1:-1],dtype=torch.float32).reshape(28,)
        self.b=b
        self.factor=factor
    def forward(self, x):
        s0=(self.b-torch.matmul(self.A,x).reshape(1,self.A.shape[0]))*self.factor
        s1=s0.repeat(self.A.shape[0],1)
        
        return x+self.model([self.y,s1])/self.factor
    
class interpolation_2D:
    def __init__(self, X,Y,values):
        self.rbfi = Rbf(X, Y, values)

    def __call__(self, x,y):
        return list(map(self.rbfi,x,y  ))
    
    
def Gauss_zeidel(A, b, x, theta):
    ITERATION_LIMIT = 2
    # x = b*0
    for it_count in range(1, ITERATION_LIMIT):
        x_new = np.zeros_like(x, dtype=np.float_)
        # print(f"Iteration {it_count}: {x}")
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
          
            x_new[i] = (1-theta)*x[i]+ theta*(b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(A@x-b)/np.linalg.norm(b)<1e-8:
             x = x_new
             return [x, it_count, np.linalg.norm(A@x-b)/np.linalg.norm(b)]
            
        x = x_new

    return [x, it_count, np.linalg.norm(A@x-b)/np.linalg.norm(b)]





def deeponet(model, func):
    X_test=[]
    Y_test=[]
    x=domain[1:-1]
    s1=func(x)

    for j,y in enumerate(list(x)):
        X_test.append([torch.tensor(y, dtype=torch.float32),torch.tensor(s1, dtype=torch.float32)])
        Y_test.append(torch.tensor(s1, dtype=torch.float32))

    
    test_dataset = SonarDataset(X_test, Y_test)
    test_dataloader=create_loader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    coords=[]
    prediction=[]
    with torch.no_grad():    
        for input,output in test_dataloader:
            coords.append(input[0])
            prediction.append(model(input))

    coords=np.squeeze(torch.cat(coords,axis=0).numpy())
    prediction=torch.cat(prediction,axis=0).numpy()

    return prediction

def network(model,func, with_net):

    A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))
    ev,V=scipy.sparse.linalg.eigs(A,k=15,return_eigenvectors=True,which="SR")
    # print(ev)
    # # func=scipy.interpolate.interp1d(domain[1:-1],np.sin(4*math.pi*domain[1:-1]), kind='cubic')
    # func=scipy.special.legendre(4)
    b=func(domain[1:-1])
    dx=domain[1]-domain[0]
    solution=scipy.sparse.linalg.spsolve(A, b)

    # mod1=[np.dot(solution,V[:,i]) for i in range(14)]
    # mod2=[np.dot(deeponet(model, func),V[:,i]) for i in range(15)]
    # plt.plot(mod1,'r')
    # plt.plot(mod2)
    # plt.plot(solution,'r')
    # plt.plot(deeponet(model, func))
    # plt.show()


    if with_net:
        x=deeponet(model, func)
    else:
        x=deeponet(model, func)*0
    # x=torch.load(Constants.path+'pred.pt')
    tol=[]
    res_err=[]
    err=[]
    fourier_err=[]
    k_it=0
    x_k=[]

    for i in range(800):
        x_0 = x
        k_it += 1
        theta=1
        
        # if False:
        if ((k_it%10) ==0) and with_net:  
            x_k.append(x_0)
            factor=np.max(abs(b))/np.max(abs(A@x_0-b))
            x_temp = x_0*factor + \
            deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(b-A@x_0)*factor, kind='cubic' )) 
            x=x_temp/factor
            
            
            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:    
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]


       

        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        fourier_err.append([np.dot(x-solution,V[:,i]) for i in range(15)])
        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
        tol.append(np.linalg.norm(x-x_0))

    # torch.save(x, Constants.path+'pred.pt')
    return err, res_err, fourier_err, x_k

from one_d.main import model

experment_path=Constants.path+'runs/'
best_model=torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])


def main1(func):
    err_net, res_err_net, f_net, x_k=network(model, func,with_net=True)
    return (err_net, res_err_net, f_net, x_k)
    # torch.save([ err_net, res_err_net], Constants.path+'hints_fig.pt')
def main2(func): 
    err_gs, res_err_gs, iter=network(model, func,with_net=False)
    return err_gs, res_err_gs
    # torch.save([ err_gs, res_err_gs], Constants.path+'gs_fig.pt')

e_deeponet=[]
r_deeponet=[]
fourier_deeponet=[]
e_gs=[]
r_gs=[]

func=scipy.interpolate.interp1d(domain[1:-1],(1-domain[1:-1]**2), kind='cubic')
# temp1, temp2, temp3, temp4=main1(func)
# torch.save((temp1,temp2,temp3,temp4), Constants.path+'modes_error.pt')
e_deeponet, r_deeponet, fourier_deeponet, x_k= torch.load( Constants.path+'modes_error.pt')

def plot_hints():
    fig,ax=plt.subplots(2)
    D=np.array(fourier_deeponet)
    for j in range(D.shape[-1]):
        if j<7:
            ax[0].plot(D[:,j], label=str(j))
            ax[0].legend()
            ax[0].text(40,0,'relative error='+str(r_deeponet[-1]))
        else:
            ax[1].plot(D[:,j], label=str(j))
            ax[1].legend()
            ax[1].text(40,0,'relative error='+str(r_deeponet[-1]))

    plt.show()


def non_linear(func,x_k):
    all_ev=[]
    t= np.linspace(0,1,10)
    A=-L - Constants.k* scipy.sparse.identity(L.shape[0])
    b=func(domain[1:-1])
    solution=scipy.sparse.linalg.spsolve(A, b)
    for xk in x_k: 
        evk=[]
        Xi=[(1-t[i])*solution+t[i]*xk for i in range(t.shape[0])]
        factor=np.max(abs(b))/np.max(abs(A@xk-b)) 
        G=g(torch.tensor(A.todense(),dtype=torch.float32),model,torch.tensor(b,dtype=torch.float32),factor)
        for xi in Xi:
            yi=torch.tensor(xi,dtype=torch.float32,requires_grad=True)
            J=jacobian(G,yi)
            with torch.no_grad():
                evk.append(torch.linalg.eigvals(J).numpy())
        all_ev.append(evk)        
    return all_ev            
# plot_hints()
all_ev=non_linear(func,x_k)
for l in all_ev[-1]:
    plt.scatter(l.real,l.imag)
plt.show()







# func=scipy.interpolate.interp1d(domain[1:-1],A@b, kind='cubic')
# print(deeponet(model, func))


# ax[0].plot(e_deeponet) 
# ax[1].plot(r_deeponet)
# ax[0].set_title('relative error')
# ax[1].set_title('residual error')

 

# print(r_deeponet[0][-1])    
# plt.show()


  
  