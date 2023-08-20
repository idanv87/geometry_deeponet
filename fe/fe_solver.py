import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, DataLoader
import sys


from utils import np_to_torch
from utils import spread_points
from constants import Constants
from functions.functions import gaussian
from dataset import create_loader


def fourier_expand(u,V):
    expansion=[]
    for v in V:
        expansion.append(np.dot(u,v)*v)
    return expansion    

class interplation_block:
    def __init__(self, points, values):
        assert points.shape[-1] == 2

        self.tck = scipy.interpolate.bisplrep(points[:, 0], points[:, 1], values)

    def __call__(self, X, Y):
        return scipy.interpolate.bisplev(X, Y, self.tck)

class SonarDataset(Dataset):
    def __init__(self, X):
        self.data_len = len(X)
        self.x = X
    def __len__(self):
        # this should return the size of the dataset
        return self.data_len
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx]


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





def deeponet(model, func, domain=torch.load(Constants.path + "polygons/1.pt")):

    x2 = np.array(list(map(func, domain['hot_points'][:,0], domain['hot_points'][:,1])))
    data=[]
    for j in range(0, domain['interior_points'].shape[0], 1):
            y = np.expand_dims(domain['interior_points'][j], -1)
            f_domain = np.expand_dims(x2, -1)
            data.append([np_to_torch(y),np_to_torch(y),np_to_torch(y),np_to_torch(f_domain)])

    data_set=SonarDataset(data)
    test_dataloader=create_loader(data_set, batch_size=100, shuffle=False, drop_last=False)
    coords=[]
    prediction=[]
    with torch.no_grad():    
        for input in test_dataloader:
            coords.append(input[0])
            prediction.append(model(input)[0])

    coords=np.squeeze(torch.cat(coords,axis=0).numpy())
    prediction=torch.cat(prediction,axis=0).numpy()
    X=domain['interior_points'][:,0]
    Y=domain['interior_points'][:,1]
    return prediction

def network(model, with_net):
    point=np.array([0,-0.3])
    func=gaussian(point)

    domain=torch.load(Constants.path + "polygons/1.pt")
    X=domain['interior_points'][:,0]
    Y=domain['interior_points'][:,1]
    b=np.array(list(map(func,X, Y)))
    M=domain['M']

    A = (-M - Constants.k* scipy.sparse.identity(M.shape[0]))
    ev,V=scipy.sparse.linalg.eigs(A,k=10,return_eigenvectors=True,which="SR")



    
    solution=scipy.sparse.linalg.spsolve(A, b)
    solution_expansion=fourier_expand(solution, [V[:,k] for k in range(10)])

    if with_net:
        x=deeponet(model, func)/100
    else:
        x=deeponet(model, func)/100*0
    # x=torch.load(Constants.path+'pred.pt')
    tol=[]
    res_err=[]
    err=[]
    k_it=0

    for i in range(1000):
        x_0 = x
        k_it += 1
        theta=1
        
        # if False:
        if ((k_it%40) ==0) and with_net:     
            factor=np.max(abs(A@x_0-b))
            x = x_0 + deeponet(model, 
                               interplation_block(domain['interior_points'],(A@x_0-b)/factor  ))/100*factor

        else:    
            x = Gauss_zeidel(A.todense(), b, x_0, theta)[0]


       
        x_expansion=fourier_expand(x, [V[:,k] for k in range(10)])
        fourier_err=[np.linalg.norm(x_expansion[i]-solution_expansion[i]) for i in range(10)]
        res_err.append(np.linalg.norm(A@x-b)/np.linalg.norm(b))
        err.append(np.linalg.norm(x-solution)/np.linalg.norm(solution))
        tol.append(np.linalg.norm(x-x_0))

    # torch.save(x, Constants.path+'pred.pt')
    return fourier_err,err, res_err, k_it    

from model import model
experment_dir='geo_deeponet/'
experment_path=Constants.path+'runs/'+experment_dir
best_model=torch.load(experment_path+'best_model.pth')
model.load_state_dict(best_model['model_state_dict'])
fourier_error, err_net, res_err_net, iter=network(model,with_net=True)

def main1():
    fourier_error, err_net, res_err_net, iter=network(model,with_net=True)
    torch.save([fourier_error, err_net, res_err_net], Constants.path+'hints_fig.pt')
def main2(): 
    fourier_error, err_gs, res_err_gs, iter=network(model,with_net=False)
    torch.save([fourier_error, err_gs, res_err_gs], Constants.path+'gs_fig.pt')
def main():
    l1=torch.load(Constants.path+'hints_fig.pt')[2]

    l2=torch.load(Constants.path+'gs_fig.pt')[2]


    plt.plot(l1[200:], 'b',  label='hints')
    plt.plot(l2[200:],'r', label='GS')

    plt.legend()
    plt.show()
    # print(fourier_error1)
    # print(fourier_error2)


# from multiprocessing import Process
# if __name__ == "__main__":
#     p1 = Process(target=main1)
#     p1.start()
#     p2 = Process(target=main2)
#     p2.start()
#     p1.join()
#     p2.join()    
main()  

  
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