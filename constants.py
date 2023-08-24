import os
import sys
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
    current_path=os.path.abspath(__file__)

    # sys.path.append(current_path.split('deeponet')[0]+'deeponet/')
    # path='/content/drive/MyDrive/clones'

    path = current_path.split('deeponet')[0]+'/data_exp3/'
    train_path= '/Users/idanversano/Documents/clones/data_exp3/train_set/'
    test_path= '/Users/idanversano/Documents/clones/data_exp3/test_set/'

    fig_path=path+'figures/'
    k=25


    h = 1/30



    batch_size =16
    num_epochs = 400
    hot_spots_ratio = 1


    isExist = os.path.exists(train_path)
    if not isExist:
        os.makedirs(train_path)

    isExist = os.path.exists(test_path)
    if not isExist:
        os.makedirs(test_path)

    isExist = os.path.exists(fig_path)
    if not isExist:
        os.makedirs(fig_path)

    isExist = os.path.exists(path+'polygons')
    if not isExist:
        os.makedirs(path+'polygons')
    isExist = os.path.exists(path+'base_polygon')
    if not isExist:
        os.makedirs(path+'base_polygon')    

    isExist = os.path.exists(path+'hints_polygons')
    if not isExist:
        os.makedirs(path+'hints_polygons')






