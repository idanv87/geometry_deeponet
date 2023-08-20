import os
import sys
import math


import dmsh
import meshio
import optimesh
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy.stats import qmc


current_path=os.path.abspath(__file__)
sys.path.append(current_path.split('deeponet')[0]+'deeponet/')
from two_d.geometry.geometry import Polygon
from pydec.dec import simplicial_complex
from utils import np_to_torch
from constants import Constants
from functions.functions import gaussian
# from two_d.main import test_dataset, Y_test, L, SonarDataset, F, domain, generate_sample, plot_surface, sample
from two_d.two_d_data_set import create_loader 


def loss(a,*args):
        basis,f, x,y=args
        assert len(a)==len(basis)
        return np.linalg.norm(np.sum(np.array([a[i]*func(np.array([x, y]).T) for i,func in enumerate(basis)]),axis=0)-f)**2/len(basis)



     
def generate_domains():
    num_domains=15
    for j in range(num_domains):
        theta1=np.random.uniform(low=0.0, high=math.pi/2)
        theta2=np.random.uniform(low=math.pi/2, high=math.pi)
        theta3=np.random.uniform(low=math.pi, high=3*math.pi/2)
        theta4=np.random.uniform(low=3*math.pi/2, high=2*math.pi)
        theta=[theta1, theta2, theta3, theta4]

        p=Polygon(np.array([[np.cos(theta[i]), np.sin(theta[i])] for i in range(len(theta))]))
        try:
            p.create_mesh(0.1)
            p.save(current_path.split('deeponet')[0]+'data_deeponet/polygons/'+str(j)+'.pt')
        except:
             pass

        


def create_data(domain):
    x=domain['interior_points'][:,0]
    y=domain['interior_points'][:,1]

    # x_hot=domain['hot_points'][:,0]
    # y_hot=domain['hot_points'][:,1]


    M=domain['M']
    A = (-M - Constants.k* scipy.sparse.identity(M.shape[0]))
    test_functions=domain['radial_basis']
    V=[func(np.array([x, y]).T) for func in test_functions]
    F=[v for v in V]
    psi=[scipy.sparse.linalg.spsolve(A,b) for b in F]

    # V_hot=[func(np.array([x_hot, y_hot]).T) for func in test_functions]
    # F_hot=[v for v in V_hot]

    
    moments=domain['moments'][:2*len(domain['generators'])]
    moments_x=[m.real/len(domain['generators']) for m in moments]
    moments_y=[m.imag/len(domain['generators']) for m in moments]


    return x,y,F, psi, moments_x, moments_y

def expand_function(f,domain):
    
    rect=torch.load(current_path.split('deeponet')[0]+'data_deeponet/polygons/rect.pt')
    x=domain['interior_points'][:,0]
    y=domain['interior_points'][:,1]
    basis=rect['radial_basis']
    x0=np.random.rand(len(basis),1)
    res = minimize(loss, x0, method='nelder-mead',args=(basis,f,x,y), options={'xatol': 1e-4, 'disp': True})
    return res.x

    
    

# rect=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]))
# rect.create_mesh(0.2)
# rect.save(current_path.split('deeponet')[0]+'data_deeponet/polygons/rect.pt')




if __name__=='__main__':        
    generate_domains()


        


 



# x,y=np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))
# values=x.ravel()*0
# values[12]=1
# p=scipy.interpolate.RBFInterpolator(np.array([x.ravel(), y.ravel()]).T,values)
# x,y=np.meshgrid(np.linspace(0,1,20),np.linspace(0,1,20))
# plot_surface(x,y,p(np.array([x.ravel(), y.ravel()]).T).reshape(20,20))
