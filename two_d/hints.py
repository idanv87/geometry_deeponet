import os
import sys
import math
from matplotlib.ticker import ScalarFormatter


from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import sys
from scipy.interpolate import Rbf


current_path=os.path.abspath(__file__)
sys.path.append(current_path.split('deeponet')[0]+'deeponet/')

from constants import Constants
from utils import save_eps, plot_figures
from two_d.main import hint_names, SonarDataset, generate_sample
from two_d.two_d_data_set import create_loader
from draft import create_data, expand_function
from two_d.geometry.geometry import Polygon



class interpolation_2D:
    def __init__(self, X,Y,values):
        self.rbfi = Rbf(X, Y, values)

    def __call__(self, x,y):
        return self.rbfi(x,y)

   

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


#


def deeponet(model, func,polyg, domain, domain_hot, moments_x, moments_y, angle_fourier):
    X_test_i=[]
    Y_test_i=[]
    s0=func(domain_hot[0], domain_hot[1])
    a=expand_function(func(domain[0], domain[1]), polyg)

    for j in range(len(domain[0])):
        X_test_i.append([
                        torch.tensor([domain[0][j],domain[1][j]], dtype=torch.float32), 
                         torch.tensor(a, dtype=torch.float32),
                         torch.tensor(0, dtype=torch.float32),
                         torch.tensor(0, dtype=torch.float32),
                         torch.tensor(angle_fourier, dtype=torch.float32)
                         ])
        Y_test_i.append(torch.tensor(s0, dtype=torch.float32))

    
    test_dataset = SonarDataset(X_test_i, Y_test_i)
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

def network(model, with_net, polyg, J):
    domain=[polyg['interior_points'][:,0],polyg['interior_points'][:,1]]
    domain_hot=[polyg['hot_points'][:,0],polyg['hot_points'][:,1]]
    moments=polyg['moments'][:2*len(polyg['generators'])]
    moments_x=[m.real/len(polyg['generators']) for m in moments]
    moments_y=[m.imag/len(polyg['generators']) for m in moments]
    angle_fourier=polyg['angle_fourier']
    L=polyg['M']
    A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))
    ev,V=scipy.sparse.linalg.eigs(-L,k=15,return_eigenvectors=True,which="SR")
    print(ev)

    xi,yi,F, F_hot,psi, temp1, temp2, temp3=create_data(polyg)
    sampler = qmc.Halton(d=len(F), scramble=False)
    sample = 20*sampler.random(n=2)-10
    
    np.random.seed(1)
    b=generate_sample(np.random.uniform(0,1,len(F))*20-10,F, F_hot, psi)[0]# func=scipy.special.legendre(4)  
  
    func=interpolation_2D(domain[0],domain[1], b)


    solution=scipy.sparse.linalg.spsolve(A, b)
    predicted=deeponet(model, func, polyg, domain, domain_hot, moments_x, moments_y, angle_fourier)
    # print(np.linalg.norm(solution-predicted)/np.linalg.norm(solution))


    if with_net:
        x=deeponet(model, func, polyg, domain, domain_hot, moments_x, moments_y, angle_fourier)
    else:
        x=deeponet(model, func, polyg, domain, domain_hot, moments_x, moments_y, angle_fourier)

    solution_expansion= [np.dot(solution,V[:,s]) for s in range(V.shape[1])]
    fourier_error=[]
    res_err=[]
    err=[]
    k_it=0

    for i in range(400):
        x_0 = x
        fourier_expansion=[np.dot(x,V[:,s]) for s in range(V.shape[1])]
        fourier_error.append([abs(f-g) for f,g in zip(fourier_expansion, solution_expansion)])
        k_it += 1
        theta=2/3

        
        # if False:
        if ( k_it%J==0) and with_net:  
            # print(np.max(abs(generate_sample(sample[4])[0])))
            xi,yi,F, F_hot,psi, temp1, temp2, temp3=create_data(polyg)
            factor=np.max(abs(b))/np.max(abs(A@x_0-b))*np.max(abs(generate_sample(sample[0],F, F_hot, psi)[0]))
            # factor=99./np.max(abs(A@x_0-b))
            x_temp = x_0*factor + \
            deeponet(model, interpolation_2D(domain[0],domain[1],(b-A@x_0)*factor ), polyg, domain, domain_hot, moments_x, moments_y, angle_fourier) 
            x=x_temp/factor
            
            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:    
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]


       
        print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
   


   
    return err, res_err,fourier_error 






def main1(model, polyg, file_name, J):
    err_net, res_err_net, fourier_error =network(model,True, polyg,J)
    torch.save([ err_net, res_err_net, fourier_error ], Constants.fig_path+file_name)

def main2(model,polyg): 
    err_gs, res_err_gs, iter=network(model,False, polyg)
    torch.save([ err_gs, res_err_gs], Constants.fig_path+'gs_fig.pt')










def figure1():


    fig, ax = plt.subplots()
    err, res_err,fourier_error=torch.load(Constants.fig_path+'hints_fig3.pt')
    ax=plot_figures(ax, err, title='relative error', label='J=3', color='red',xlabel='iterations',
                    ylabel='error', text_hight=0.3
                    )
    err, res_err,fourier_error=torch.load(Constants.fig_path+'hints_fig5.pt')
    ax=plot_figures(ax, err, title='relative error', label='J=5', color='green',xlabel='iterations',
                    ylabel='error', text_hight=0.4
                    )
    err, res_err,fourier_error=torch.load(Constants.fig_path+'hints_fig10.pt')
    ax=plot_figures(ax, err, title='relative error', label='J=10', color='blue',xlabel='iterations',
                    ylabel='error', text_hight=0.5
                    )

    ax.annotate('',
                xy = (400,0),
                xytext =(320,0.3),
                arrowprops = dict(facecolor = 'black', width = 0.2, headwidth = 8),
                horizontalalignment = 'center') 
    
    save_eps('hints3250.eps')
    plt.show(block=True)

def figure2():
    

    fig, ax = plt.subplots(2,2)
    fig.suptitle('fourier_error J=3')
    err, res_err, fourier_error=torch.load(Constants.fig_path+'hints_fig3.pt')
    plot_figures(ax[0,0], [s[0] for s in fourier_error], title='mode=1', color='black',xlabel='',
                    ylabel='error'
                    )
    plot_figures(ax[0,1], [s[1] for s in fourier_error], title='mode=2', color='black',xlabel='',
                    ylabel='error'
                    )
    plot_figures(ax[1,0], [s[9] for s in fourier_error], title= 'mode=10', color='black',xlabel='iterations',
                    ylabel='error'
                    )
    plot_figures(ax[1,1], [s[10] for s in fourier_error], title='mode=11', color='black',xlabel='iterations',
                    ylabel='error'
                    )

                    
    # ax.annotate('',
    #             xy = (400,0),
    #             xytext =(320,0.2),
    #             arrowprops = dict(facecolor = 'black', width = 0.2, headwidth = 8),
    #             horizontalalignment = 'center') 

    


    
    save_eps('hints3250_fourier.eps')

    plt.show(block=True)


from two_d.main import model
experment_path=Constants.path+'runs/'
best_model=torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])
polyg=torch.load(hint_names[0])
# domain=Polygon(polyg['generators'])
# domain.create_mesh(1/20)
# domain.save(Constants.path+'hints_polygons/'+str(3250)+'.pt')
# polyg2=torch.load(Constants.path+'hints_polygons/'+str(3250)+'.pt')
# main1(model, polyg, 'super_hints_fig5.pt', J=10)
figure2()




