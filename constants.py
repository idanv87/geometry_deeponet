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

from airfoils import Airfoil

class foil_function:
    def __init__(self,m,p, t):
        assert p>0 and p<1
        assert m>0 and m<0.1
        self.m=m
        self.p=p
        self.t=t
    def __call__(self,x):
        assert (x>=0).all() and (x<=1).all()
        LU=(self.m/self.p**2)*(2*self.p*x-x**2)
        LV=(self.m/((1-self.p)**2))*(1-2*self.p + 2*self.p*x-x**2)

        thetaU=np.arctan(2*(self.m/self.p**2)*(self.p-x))
        thetaL=np.arctan(2*(self.m/((1-self.p)**2))*(self.p-x))

        val1=np.array((x<=self.p)*1)
        val2=np.array((x>self.p)*1)
      
        yt=5*self.t*(0.2969*np.sqrt(x) -0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
        # ytL=-5*self.t*(0.2969*np.sqrt(x) -0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
        yc=val1*LU+val2*LV
   
        theta=val1*thetaU+val2*thetaL

        xu=x-yt*np.sin(theta)
        yu=yc+yt*np.cos(theta)
        xl=x+yt*np.sin(theta)
        yl=yc-yt*np.cos(theta)

     
        return xu,yu,xl,yl
        
         





class Constants:
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    dtype = torch.float32
    current_path=os.path.abspath(__file__)

    # sys.path.append(current_path.split('deeponet')[0]+'deeponet/')
    # path='/content/drive/MyDrive/clones'

    path = current_path.split('deeponet')[0]+'data_exp3/'
    train_path= path+'train_set/'
    test_path= path+'test_set/'

    fig_path=path+'figures/'
    k=25


    h = 1/30



    batch_size =64
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

def generate_my_naca():
        k=50
        for  i in range(1,10):
            for j in range(1,10):
            # foil = Airfoil.NACA4('5820')
            
                f=foil_function(i/100,j/10,k/100)
                x=np.linspace(0,1,100)
                xu,yu,xl,yl=f(x)

                X=np.hstack((xu,np.flip(xl[1:-1])))
                Y=np.hstack((yu,np.flip(yl[1:-1])))
                torch.save((X,Y),Constants.path+'my_naca/'+str(i)+str(j)+str(k)+'.pt')
            
# generate_my_naca()   


# foil=Airfoil.NACA4('8950')

# d=foil.all_points
# foil.plot()
# plt.show()
if __name__=='__main__':
    m=8
    p=9
    k=50
    f=foil_function(m/100,p/10,k/100)
    x=np.linspace(0,1,500)
    xu,yu,xl,yl=f(x)

    X=np.hstack((xu,xl[1:-1]))
    Y=np.hstack((yu,yl[1:-1]))
    plt.scatter(X,Y)

    plt.show()

# foil.plot()
# plt.scatter(p/10,0,c='g')
# plt.scatter(d[0],d[1],c='r')