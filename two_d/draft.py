import os
import sys
import math

# from shapely.geometry import Polygon as Pol2
import dmsh
import meshio
import optimesh
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from sklearn.cluster import KMeans
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy.stats import qmc
import pandas as pd


current_path=os.path.abspath(__file__)
sys.path.append(current_path.split('deeponet')[0]+'deeponet/')
from two_d.geometry.geometry import Polygon
from utils import extract_path_from_dir

from constants import Constants

from two_d.two_d_data_set import create_loader 


def loss(a,*args):
        basis,f, x,y=args
        assert len(a)==len(basis)
        return np.linalg.norm(np.sum(np.array([a[i]*func(np.array([x, y]).T) for i,func in enumerate(basis)]),axis=0)-f)**2


def create_data(domain):
    x=domain['interior_points'][:,0]
    y=domain['interior_points'][:,1]

    x_hot=domain['hot_points'][:,0]
    y_hot=domain['hot_points'][:,1]


    M=domain['M']
    A = (-M - Constants.k* scipy.sparse.identity(M.shape[0]))
    test_functions=domain['radial_basis']
    V=[func(np.array([x, y]).T) for func in test_functions]
    F=[v for v in V]
    psi=[scipy.sparse.linalg.spsolve(A,b) for b in F]

    V_hot=[func(np.array([x_hot, y_hot]).T) for func in test_functions]
    F_hot=[v for v in V_hot]

    
    moments=domain['moments'][:2*len(domain['generators'])]
    moments_x=[m.real/len(domain['generators']) for m in moments]
    moments_y=[m.imag/len(domain['generators']) for m in moments]
    angle_fourier=domain['angle_fourier']


    return x,y,F,F_hot, psi, moments_x, moments_y, angle_fourier

def expand_function(f,domain):
    # f is a vector of f evaluated on the domain points
    
    base_rect=torch.load(Constants.path+'base_polygon/base_rect.pt')
    # base_rect=torch.load(Constants.path+'/base_polygon/base_rect.pt')
    x=domain['interior_points'][:,0]
    y=domain['interior_points'][:,1]
    basis=base_rect['hot_radial_basis']
  
    phi=np.array([func(np.array([x, y]).T) for func in basis]).T
    return np.linalg.solve(phi.T@phi,phi.T@f)

    #   x0=np.random.rand(len(basis),1)
    # res = minimize(loss, x0, method='BFGS',args=(basis,f,x,y), options={'xatol': 1e-4, 'disp': True})
    # return res.x

    
    

# rect=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]))
# rect.create_mesh(0.1)
# rect.save(Constants.path+'polygons/rect.pt')

# rect=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]))
# rect.create_mesh(1/40)
# rect.save(Constants.path+'base_polygon/base_rect.pt')

# figures:
# import urllib
# selig_url = 'http://airfoiltools.com/airfoil/seligdatfile?airfoil=n0012-il'
# selig_path = 'naca0012-selig.dat'
# urllib.request.urlretrieve(selig_url, selig_path)
# Load coordinates from file.
def generate_domains():

    fourier_coeff=[]
    for i,name in enumerate(os.listdir(Constants.path+'naca/')):
      

        
       
        
            
        with open(Constants.path+'naca/'+name, 'r') as infile:
            x1, y1 = np.loadtxt(infile, unpack=True, skiprows=1)
            lengeths=[np.sqrt((x1[(k+1)%x1.shape[0]]-x1[k])**2+ (y1[(k+1)%x1.shape[0]]-y1[k])**2) for k in range(x1.shape[0])]
            X=[]
            Y=[]
            for j in range(len(lengeths)):
                    if lengeths[j]>1e-4:
                        X.append(x1[j])
                        Y.append(y1[j])
            try:                   
                domain=Polygon(np.vstack((np.array(X),np.array(Y)/np.max(abs(y1)))).T)
                domain.create_mesh(0.1)
                domain.save(Constants.path+'polygons/'+str(i)+'.pt')
            except:
                pass    


def analyze_data():
    names=extract_path_from_dir(Constants.path+'polygons/')
    C=[]
    for i,name in enumerate(names):
        domain=torch.load(name)
        C.append(domain['angle_fourier'])
        # Polygon.plot(domain['generators'], str(i))
    return C
    



if __name__=='__main__':
     domain_coeff=analyze_data()
    # base_domain=Polygon(np.array([[0,0],[1,0],[1,1],[0,1]]))
    # base_domain.create_mesh(0.1)
    # base_ domain.save(Constants.path+'base_polygon/base_rect.pt')
    # generate_domains()







# if __name__=='__main__':
#     polygon = Pol2(shell=((0,0),(1,0),(1,1),(0,1)),
# holes=None
# fig, ax = plt.subplots()
#     plot_polygon(ax, polygon, facecolor='white', edgecolor='red')
# plt.show()
####################################################################################################################################################################
    # p=torch.load(Constants.path+'polygons/rect.pt')
    # plt.scatter(p['interior_points'][:,0], p['interior_points'][:,1],c='b')
    # plt.scatter(p['hot_points'][:,0], p['hot_points'][:,1],c='r')
    # plt.title('interior points and hot points')
    # plt.show()
####################################################################################################################################################################







