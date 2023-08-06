import os
from typing import Any
import torch
import numpy as np
import math
import cmath


class Constants:
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dtype = torch.float32

    path = '/Users/idanversano/Documents/clones/deeponet_data/'
    # path = '/Users/idanversano/Documents/clones/exp_data/'
    output_path = '/Users/idanversano/Documents/clones/deeponet_output/'

    num_samples = 5
    num_edges = 7
    var_center = 0
    var_angle = 0.4
    radius = 3
    h = 1/20
    gauss_points = 5

    num_control_polygons = 1
    batch_size = 32
    num_epochs = 1
    hot_spots_ratio = 2
    num_moments = 5

    pts_per_polygon = 10
    points_on_circle = []
    ev_per_polygon = 2
    for r in list(np.linspace(0, 0.99, 50)):
        for theta in list(np.linspace(0, 2*math.pi, 50)):
            z = r*cmath.exp(theta*1j)
            points_on_circle.append([z.real, z.imag])

    points_on_circle = np.array(points_on_circle)


    k = 3.11

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




        
        