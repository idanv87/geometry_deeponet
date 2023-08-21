import os
import sys
import math



import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from scipy.interpolate import Rbf


current_path=os.path.abspath(__file__)
sys.path.append(current_path.split('deeponet')[0]+'deeponet/')
from utils import np_to_torch
from constants import Constants
from functions.functions import gaussian
from two_d.main import names, SonarDataset, generate_sample, plot_surface, sample
from two_d.two_d_data_set import create_loader
from draft import create_data



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





def deeponet(model, func, domain, domain_hot, moments_x, moments_y):
    X_test_i=[]
    Y_test_i=[]
    s0=func(domain_hot[0], domain_hot[1])

    for j in range(len(domain[0])):
        X_test_i.append([
                        torch.tensor([domain[0][j],domain[1][j]], dtype=torch.float32), 
                         torch.tensor(s0, dtype=torch.float32),
                         torch.tensor(moments_x, dtype=torch.float32),
                         torch.tensor(moments_y, dtype=torch.float32)
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

def network(model, with_net, polyg):
    domain=[polyg['interior_points'][:,0],polyg['interior_points'][:,1]]
    domain_hot=[polyg['hot_points'][:,0],polyg['hot_points'][:,1]]
    moments=polyg['moments'][:2*len(polyg['generators'])]
    moments_x=[m.real/len(polyg['generators']) for m in moments]
    moments_y=[m.imag/len(polyg['generators']) for m in moments]
    L=polyg['M']
    A = (-L - Constants.k* scipy.sparse.identity(L.shape[0]))
    # ev,V=scipy.sparse.linalg.eigs(A,k=15,return_eigenvectors=True,which="SR")
    # print(ev)


    func=interpolation_2D(domain[0],domain[1], np.sin(4*math.pi*domain[0])*np.sin(4*math.pi*domain[1]))
    # func=interpolation_2D(domain[0],domain[1], F[0]*100)

    # func=scipy.special.legendre(4)
    b=func(domain[0],domain[1])
    solution=scipy.sparse.linalg.spsolve(A, b)
    predicted=deeponet(model, func, domain, domain_hot, moments_x, moments_y)
    print(np.mean(abs(solution-predicted)))
    
    # plot_surface(domain[0].reshape(18,18),domain[1].reshape(18,18),b.reshape(18,18))
    # plot_surface(domain[0].reshape(18,18),domain[1].reshape(18,18),predicted.reshape(18,18))


    # mod1=[np.dot(solution,V[:,i]) for i in range(14)]
    # mod2=[np.dot(deeponet(model, func),V[:,i]) for i in range(15)]
    # plt.plot(mod1,'r')
    # plt.plot(mod2)
    # plt.plot(solution,'r')
    # plt.plot(deeponet(model, func))
    # plt.show()


    if with_net:
        x=deeponet(model, func,domain, domain_hot, moments_x, moments_y)
    else:
        x=deeponet(model, func,domain, domain_hot, moments_x, moments_y)*0
    # x=torch.load(Constants.path+'pred.pt')
    tol=[]
    res_err=[]
    err=[]
    k_it=0

    for i in range(400):
        x_0 = x
        k_it += 1
        theta=1
        
        # if False:
        if ((k_it%20) ==0) and with_net:  
            # print(np.max(abs(generate_sample(sample[4])[0])))
            factor=np.max(abs(b))/np.max(abs(A@x_0-b))*np.max(abs(generate_sample(sample[4])[0]))
            # factor=b/(b-A@x_0)
            # mod1=[np.dot((b-A@x_0)*factor,V[:,i]) for i in range(14)]
            # mod2=[np.dot(b,V[:,i]) for i in range(14)]
            # plt.plot(mod1,'r')
            # plt.plot(mod2,'b')


            # func=scipy.interpolate.interp1d(domain[1:-1],(b-A@x_0)*factor, kind='cubic')
            # plt.plot(deeponet(model, func),'r')
            # plt.plot(scipy.sparse.linalg.spsolve(A, (b-A@x_0)*factor))
            # plt.show()


            # plt.plot(b,'r')
            # plt.plot((b-A@x_0)*factor)
            # plt.show()
            x_temp = x_0*factor + \
            deeponet(model, interpolation_2D(domain[0],domain[1],(b-A@x_0)*factor ),domain, domain_hot, moments_x, moments_y) 
            x=x_temp/factor
            
            # x = x_0 + deeponet(model, scipy.interpolate.interp1d(domain[1:-1],(A@x_0-b)*factor ))/factor

        else:    
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]


       
        print(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
        tol.append(np.linalg.norm(x-x_0))

    # torch.save(x, Constants.path+'pred.pt')
   
    return err, res_err, k_it    

from two_d.main import model
experment_dir='geo_deeponet/'
experment_path=Constants.path+'runs/'+experment_dir
best_model=torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])
# err_net, res_err_net, iter=network(model,with_net=True)
polyg=torch.load(names[0])
def main1():
    err_net, res_err_net, iter=network(model,True, polyg)
    torch.save([ err_net, res_err_net], Constants.path+'hints_fig.pt')
def main2(): 
    err_gs, res_err_gs, iter=network(model,False, polyg)
    torch.save([ err_gs, res_err_gs], Constants.path+'gs_fig.pt')
def main():
    l1=torch.load(Constants.path+'hints_fig.pt')[1]

    l2=torch.load(Constants.path+'gs_fig.pt')[1]


    plt.plot(l1, 'b',  label='hints')
    # plt.plot(l2,'r', label='GS')
    # print(l1[-1])
    print(l2[-1])

    plt.legend()
    # plt.show()
    # print(fourier_error1)
    # print(fourier_error2)

# main1()
main2()
main()  
# from multiprocessing import Process
# if __name__ == "__main__":
#     p1 = Process(target=main1)
#     p1.start()
#     p2 = Process(target=main2)
#     p2.start()
#     p1.join()
#     p2.join()    


  
    # solution=scipy.sparse.linalg.spsolve(A, b)




# def plot_results(x,y,y_test, y_pred):
#     error=torch.linalg.norm(y_test-y_pred)/torch.linalg.norm(y_test)
#     fig, ax=plt.subplots(1,2)
#     fig.suptitle(f'relative L2 Error: {error:.3e}')
#     im0=ax[0].scatter(x,y,c=y_test)
#     fig.colorbar(im0, ax=ax[0])
#     im1=ax[1].scatter(x,y,c=y_pred)
#     fig.colorbar(im1, ax=ax[1])
#     # im2=ax[2].scatter(x,y,c=abs(y_pred-y_test))
#     # fig.colorbar(im2, ax=ax[2])
#     ax[0].set_title('test')
#     ax[1].set_title('pred')