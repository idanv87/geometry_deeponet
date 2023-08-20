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


# rect=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]))
# rect.create_mesh(0.2)
# rect.save(current_path.split('deeponet')[0]+'data_deeponet/polygons/rect.pt')

# small_rect=Polygon(np.array([[0,0],[0.5,0],[0.5,0.5],[0,0.5]]))
# small_rect.create_mesh(0.1)
# small_rect.save(current_path.split('deeponet')[0]+'data_deeponet/polygons/small_rect.pt')

# rect=torch.load(current_path.split('deeponet')[0]+'deeponet/two_d/polygons/rect.pt')
# small_rect=torch.load(current_path.split('deeponet')[0]+'deeponet/two_d/polygons/small_rect.pt')

def create_data(domain):
    x=domain['interior_points'][:,0]
    y=domain['interior_points'][:,1]
    x_hot=domain['interior_points'][:,0]
    y_hot=domain['interior_points'][:,1]
    M=domain['M']
    A = (-M - Constants.k* scipy.sparse.identity(M.shape[0]))
    test_functions=domain['radial_basis']
    V=[func(np.array([x, y]).T) for func in test_functions]
    F=[v for v in V]
    V_hot=[func(np.array([x_hot, y_hot]).T) for func in test_functions]
    F_hot=[v for v in V_hot]
    psi=[scipy.sparse.linalg.spsolve(A,b) for b in F]
    moments=domain['moments'][:2*len(domain['generators'])]
    moments_x=[m.real/len(domain['generators']) for m in moments]
    moments_y=[m.imag/len(domain['generators']) for m in moments]


    return x,y,F_hot, psi, moments_x, moments_y

# create_data(rect)
# xi=rect['interior_points'][:,0]
# yi=rect['interior_points'][:,1]


        


 



# x,y=np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))
# values=x.ravel()*0
# values[12]=1
# p=scipy.interpolate.RBFInterpolator(np.array([x.ravel(), y.ravel()]).T,values)
# x,y=np.meshgrid(np.linspace(0,1,20),np.linspace(0,1,20))
# plot_surface(x,y,p(np.array([x.ravel(), y.ravel()]).T).reshape(20,20))
