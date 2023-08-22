import os
from typing import Any
import torch
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import Rbf

class Constants:
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dtype = torch.float32

    # path = '/Users/idanversano/Documents/clones/deeponet_data/'

    path = '/Users/idanversano/Documents/clones/data_exp/'
    output_path = '/Users/idanversano/Documents/clones/deeponet_output/'
    k=12.1

    num_samples = 5
    num_edges = 7
    var_center = 0
    var_angle = 0.4
    radius = 3
    h = 1/30
    gauss_points = 5

    num_control_polygons = 1
    batch_size =64
    num_epochs = 400
    hot_spots_ratio = 1
    num_moments = 8

    pts_per_polygon = 10
    points_on_circle = []
    ev_per_polygon = 2
    for r in list(np.linspace(0, 0.99, 50)):
        for theta in list(np.linspace(0, 2*math.pi, 50)):
            z = r*cmath.exp(theta*1j)
            points_on_circle.append([z.real, z.imag])

    points_on_circle = np.array(points_on_circle)




    dim = 2
    # num_ev=4

    isExist = os.path.exists(path+'polygons')
    if not isExist:
        os.makedirs(path+'polygons')

    isExist = os.path.exists(path+'hints_polygons')
    if not isExist:
        os.makedirs(path+'hints_polygons')

    l = []
    for i in range(1, 5):
        for j in range(1, 5):
            l.append((i, j))




# class interpolation_2D:
#     def __init__(self, X,Y,values):
#         self.rbfi = Rbf(X, Y, values)

#     def __call__(self, x,y):
#         return list(map(self.rbfi,x,y  ))
    
# x,y=np.meshgrid(np.linspace(-1,1,20)[1:-1], np.linspace(-1,1,20)[1:-1], indexing='ij')
# positions = np.vstack([x.ravel(), y.ravel()])   
# X=positions[0]
# Y=positions[1]
# values=np.array([0 for i in range(len(X))])
# values[81]=1

# func=Rbf(X, Y, values)
# xi,yi = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), indexing='ij')
# Z=func(xi,yi)
# from matplotlib import cm
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(xi, yi, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# fig.colorbar(surf)
# plt.show()

# domain=np.linspace(0,math.pi,20)
# x,y=np.meshgrid(domain,domain,indexing='ij')
# X=[]
# Y=[]
# values=[]
# for i in range(x.shape[0]):
#     for j in range(x.shape[0]):
#         X.append(x[i,j])
#         Y.append(y[i,j])
        # values.append(np.sin(x[i,j])*np.sin(y[i,j]))
# positions = np.vstack([x.ravel(), y.ravel()])           


# X=np.array(X)
# Y=np.array(Y)
# values=np.array(values)
# f=interpolation(X,Y,values)

# import numpy as np

# rng = np.random.default_rng()
#  # radial basis function interpolator instance
# xi,yi = np.meshgrid(np.linspace(0, math.pi, 100), np.linspace(0, math.pi, 100), indexing='ij')

# X=[]
# Y=[]
# values=[]
# for i in range(xi.shape[0]):
#     for j in range(yi.shape[1]):
#         X.append(xi[i,j])
#         Y.append(yi[i,j])
#         values.append(np.sin(xi[i,j])*np.sin(yi[i,j]))
# di = np.array(f(X,Y))   # interpolated values
# values=np.array(values)
# print(np.max(abs(di-values)))
# from matplotlib import cm
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(x, y, di, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# fig.colorbar(surf)
# plt.show()

# def mydec(f):
#     def wrapper(x):
#         print("I am wrapping")
#         return 2*    f(x)
#     return wrapper

# @mydec         
# def sin(x):
#     return np.sin(x)     
# x=1

# print(sin(x))


# x=torch.tensor([[[[1],[2],[3]]],[[[4],[5],[6]]]])
# y=torch.tensor([[1],[2],[3]])


        
        