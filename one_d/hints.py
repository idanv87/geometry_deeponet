import os
import sys
import math
sys.path.append(os.getcwd())

from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import gmres
import scipy
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

import sys
from scipy.interpolate import Rbf



from utils import np_to_torch, count_trainable_params
from constants import Constants
from functions.functions import gaussian
from one_d.main import test_dataset, Y_test, create_D2, SonarDataset, F, domain, generate_sample, sample
from one_d.one_d_data_set import create_loader

main_domain=domain.copy()
domain=np.linspace(-1,1,300)

L=create_D2(domain)
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

        if np.linalg.norm(A@x-b)/np.linalg.norm(b)<1e-15:
             x = x_new
             return [x, it_count, np.linalg.norm(A@x-b)/np.linalg.norm(b)]
            
        x = x_new

    return [x, it_count, np.linalg.norm(A@x-b)/np.linalg.norm(b)]





def deeponet(model, func):
    x=domain[1:-1]
    s1=func(main_domain[1:-1])
    with torch.no_grad():
        y=torch.tensor(domain[1:-1],dtype=torch.float32).reshape(x.shape[0],)
        s_temp=torch.tensor(s1.reshape(1,s1.shape[0]),dtype=torch.float32).repeat(x.shape[0],1)
        pred2=model([y,s_temp])
    return pred2.numpy()

def network(model,func, J, J_in, hint_init):

    A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))
    
   
    ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")
    # print(ev)

    b=func(domain[1:-1])
    solution=scipy.sparse.linalg.spsolve(A, b)
    gmres_solution,exit_cod=gmres(A, b, x0=None, tol=1e-13, restart=None, maxiter=4000)
    print(np.linalg.norm(A@gmres_solution-b)/np.linalg.norm(b))
    print(exit_cod)
    solution_expansion=[np.dot(solution,V[:,s]) for s in range(V.shape[1])]


    if hint_init:
        x=deeponet(model, func)
        
    else:
        x=deeponet(model, func)*0

    tol=[]
    res_err=[]
    err=[]
    fourier_err=[]
    x_expansion=[]
    k_it=0
    x_k=[]
    count=0
    # plt.plot(solution,'r');plt.plot(x,'b');plt.show()

    for temp in range(4000):
        fourier_err.append([abs(np.dot(x-solution,V[:,i])) for i in range(15)])
        x_k.append(x)
        x_expansion.append([np.dot(x,V[:,s]) for s in range(V.shape[1])])
        
        x_0 = x
        k_it += 1
        theta=1.1
        
        if ( ((k_it%J) in J_in) and (k_it>J_in[-1])  ) :  

            factor=np.max(abs(generate_sample(sample[1])[0]))/np.max(abs(A@x_0-b))
            
            x_temp = x_0*factor + \
            deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(b-A@x_0)*factor, kind='cubic' )) 
            x=x_temp/factor
            
            
            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:    
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]



        print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))

        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))

        tol.append(np.linalg.norm(x-x_0))
    
    # torch.save(x, Constants.path+'pred.pt')
    print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
    return (err, res_err, fourier_err, x_k, solution, solution_expansion, x_expansion, J, J_in, hint_init)

from one_d.main import model

experment_path=Constants.path+'runs/'
best_model=torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])


def run_hints(func,J,J_in, hint_init):
    return network(model, func, J, J_in, hint_init)
    # torch.save([ err_net, res_err_net], Constants.path+'hints_fig.pt')


def plot_fourier(fourier_deeponet):
    modes=[1,5,10]
    colors=['blue', 'green', 'red']
    fig,ax=plt.subplots(3)
    [ax[k].scatter(list(range(len(fourier_deeponet))),[f[modes[k]-1] for f in fourier_deeponet],color=colors[k], label=str(modes[k])) for k in range(3)]
    plt.show()

def plot_solution_and_fourier(times,path, eps_name):
    times=np.asarray(times)
    e_deeponet, r_deeponet, fourier_deeponet, x_k, solution,  solution_expansion, x_expansion, J, J_in, hint_init= torch.load( path)
    all_modes=list(range(1,len(fourier_deeponet[0])+1))
    fig1,ax1=plt.subplots(5,5)   
    fig2,ax2=plt.subplots(5,5)   
    fig3,ax3=plt.subplots()   # should be J+1
    fig1.supxlabel('x')
    fig2.supxlabel('mode')
    fig1.suptitle('solution')
    fig2.suptitle('fourier coefficients')
    fig3.suptitle('relative error')
    ind=times.reshape((5,5))
    counter=0
    for i in range(ind.shape[0]):
        for j in range(ind.shape[1]):
            ax1[i,j].plot(domain[1:-1],x_k[times[counter]],'b', label='hints') 
            ax1[i,j].plot(domain[1:-1],solution, color='r', label='analytic', linestyle='dashed') 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            ax1[i,j].text(0.8, 0.9, f'iter={times[counter]}', transform=ax1[i,j].transAxes, fontsize=6, ha='left', va='top', bbox=bbox)
            ax2[i,j].text(0.8, 0.9, f'iter={times[counter]}', transform=ax2[i,j].transAxes, fontsize=6, ha='left', va='top', bbox=bbox)
            ax2[i,j].plot(all_modes,x_expansion[times[counter]],'b')  
            ax2[i,j].plot(all_modes,solution_expansion, color='r', linestyle='dashed')
            if j>0:
                ax1[i,j].set_yticks([])
                ax2[i,j].set_yticks([])
            if (i+1)<ind.shape[0]:
                ax1[i,j].set_xticks([])
                ax2[i,j].set_xticks([])
            if counter==0 and hint_init:
                ax1[i,j].text(0.5, 0.5, 'Hints', transform=ax2[i,j].transAxes, fontsize=6, ha='left', va='top'
                    , bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
                ax2[i,j].text(0.5, 0.5, 'Hints', transform=ax2[i,j].transAxes, fontsize=6, ha='left', va='top'
                    , bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
            if counter==0 and (not hint_init):
                    ax1[i,j].text(0.5, 0.5, 'zero_init', transform=ax2[i,j].transAxes, fontsize=6, ha='left', va='top'
                    , bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
                    ax2[i,j].text(0.5, 0.5, 'zero_init', transform=ax2[i,j].transAxes, fontsize=6, ha='left', va='top'
                    , bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))    
            if (counter>J_in[-1]) and ((counter% J) in J_in):
            
                ax1[i,j].text(0.5, 0.5, 'Hints', transform=ax2[i,j].transAxes, fontsize=6, ha='left', va='top'
                    , bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax2[i,j].text(0.5, 0.5, 'Hints', transform=ax2[i,j].transAxes, fontsize=6, ha='left', va='top'
                    , bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            counter+=1
    ax3.plot(e_deeponet,'g')  
    # ax3.plot(r_deeponet,'r',label='res.err') 
    # ax3.legend()
    ax3.set_xlabel('iteration')    
    ax3.set_ylabel('error')  
    ax3.text(0.9, 0.1, f'final_err={e_deeponet[-1]:.2e}', transform=ax3.transAxes, fontsize=6,ha='left', va='top'
                    , bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    fig1.savefig(eps_name+'sol.eps', format='eps',bbox_inches='tight')
    fig2.savefig(eps_name+'four.eps', format='eps',bbox_inches='tight')
    fig3.savefig(eps_name+'errors.eps', format='eps',bbox_inches='tight')
    plt.show(block=False)       
    return 1
    
# func=scipy.interpolate.interp1d(domain[1:-1],F[21], kind='cubic')

func=scipy.interpolate.interp1d(main_domain[1:-1], generate_sample(sample[1])[0], kind='cubic')
func=scipy.interpolate.interp1d(domain, (1-domain**2)*scipy.special.legendre(10)(domain) , kind='cubic')
# func=scipy.interpolate.interp1d(domain[1:-1],
#                                 10*np.sin(    10*(math.pi/2)*(domain[1:-1]+1))+
#                                 10*np.sin(6*(math.pi/2)*(domain[1:-1]+1))+
#                                 1*np.sin((math.pi/2)*(domain[1:-1]+1))
#                                 , kind='cubic')
# func=scipy.special.legendre(10)    

torch.save(run_hints(func, J=30, J_in=[0], hint_init=True), Constants.path+'modes_error.pt')
plot_solution_and_fourier(list(range(0,0+25)),Constants.path+'modes_error.pt', Constants.tex_fig_path+ 'one_d_x0_J=8_Jin=012_modes=1')

# plot_fourier(fourier_deeponet[:25])

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
        factor=np.max(abs(b))/np.max(abs(A@xk-b))*8 
        G=g(torch.tensor(A.todense(),dtype=torch.float32),model,torch.tensor(b,dtype=torch.float32),factor)
        for xi in Xi:
            yi=torch.tensor(xi,dtype=torch.float32,requires_grad=True)
            J=jacobian(G,yi)
            with torch.no_grad():
                evk.append(torch.linalg.eigvals(J).numpy())
                # evk.append(torch.linalg.eigvals(J).numpy()[int(A.shape[0]/2):])
        all_ev.append(evk)        
    torch.save(all_ev, Constants.path+'all_ev.pt')    
    return all_ev            












  
  