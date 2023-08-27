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
        self.y=torch.tensor(domain[1:-1],dtype=torch.float32).reshape(A.shape[0],)
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
    x=domain[1:-1]
    s1=func(x)
    with torch.no_grad():
        y=torch.tensor(domain[1:-1],dtype=torch.float32).reshape(s1.shape[0],)
        s_temp=torch.tensor(s1.reshape(1,s1.shape[0]),dtype=torch.float32).repeat(s1.shape[0],1)
        pred2=model([y,s_temp])
    return pred2.numpy()

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
    count=0
    for i in range(600):
        x_0 = x
        k_it += 1
        theta=1
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        fourier_err.append([np.dot(x-solution,V[:,i]) for i in range(15)])
        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))

        
        # if False:
        if (((k_it%4) ==0))and with_net :  
            count+=1
            x_k.append(x_0)
            factor=np.max(abs(b))/np.max(abs(A@x_0-b))
            x_temp = x_0*factor + \
            deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(b-A@x_0)*factor, kind='cubic' )) 
            x=x_temp/factor
            
            
            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:    
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]


       
        print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
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

func=scipy.interpolate.interp1d(domain[1:-1],(1-domain[1:-1]**2)*np.exp(domain[1:-1]), kind='cubic')


# temp1, temp2, temp3, temp4=main1(func)
# torch.save((temp1,temp2,temp3,temp4), Constants.path+'modes_error.pt')
e_deeponet, r_deeponet, fourier_deeponet, x_k= torch.load( Constants.path+'modes_error.pt')

def plot_hints():
    fig,ax=plt.subplots(2,2)
    D=np.array(fourier_deeponet)
    for j in range(D.shape[-1]):
        if j<7:
            ax[0,0].plot(D[:200,j], label=str(j))
            ax[0,0].legend(loc='lower right')
            # ax[0,0].text(40,0,'relative error='+str(r_deeponet[-1]))
            ax[0,0].set_title('relative error of low modes')
        else:
            ax[1,0].plot(D[:200,j], label=str(j))
            ax[1,0].legend(loc='lower right')
            # ax[1,0].text(40,0,'relative error='+str(r_deeponet[-1]))
            ax[1,0].set_title('relative error of high modes')
    ax[0,1].plot(e_deeponet)
    
    ax[0,1].set_title(f'relative error={"{:.3e}".format(e_deeponet[-1])}')
    ax[1,1].plot(r_deeponet)
    ax[1,1].set_title(f'residual error={"{:.3e}".format(r_deeponet[-1])}')


def plot_comparison():
    A=-L - Constants.k* scipy.sparse.identity(L.shape[0])
    b=func(domain[1:-1])
    solution=scipy.sparse.linalg.spsolve(A, b)
    plt.figure(0)
    plt.plot(deeponet(model,func),'r',label='deeponet solution')
    plt.plot(solution,'b',label='analytic  solution')
    plt.legend()
def non_linear(func,x_k):
    all_ev=[]
    t= np.linspace(0,1,30)
    A=-L - Constants.k* scipy.sparse.identity(L.shape[0])
    b=func(domain[1:-1])
    solution=scipy.sparse.linalg.spsolve(A, b)
    plt.figure(0)
    plt.plot(deeponet(model,func),'r',label='deeponet solution')
    plt.plot(solution,'b',label='analytic  solution')
    plt.legend()
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
    torch.save(all_ev, Constants.path+'all_ev.pt')    
    return all_ev            

plot_hints()
plt_comparison()
# non_linear(func,x_k)
all_ev=torch.load(Constants.path+'all_ev.pt')

fig,ax=plt.subplots(1)
for j,l in enumerate(all_ev):
    spectral_radii=np.max([np.max(l[s].real**2+l[s].imag**2) for s in range(len(l))])
    ax.scatter(j,np.sqrt(spectral_radii))
    ax.set_title('spectral radius')
ax.text(j,1,spectral_radii)
plt.show(block=True)







# func=scipy.interpolate.interp1d(domain[1:-1],A@b, kind='cubic')
# print(deeponet(model, func))


# ax[0].plot(e_deeponet) 
# ax[1].plot(r_deeponet)
# ax[0].set_title('relative error')
# ax[1].set_title('residual error')

 

# print(r_deeponet[0][-1])    
# plt.show()


  
  