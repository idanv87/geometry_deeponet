
import math
from typing import Any
import numpy as np
import torch
import sys
import os
import scipy
from sympy import *
from sympy import sin, cos, exp
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils import spread_points
from constants import Constants

class sin_function:
    def __init__(self, n, m, a, b):
        self.n = n
        self.m = m
        self.a = a
        self.b = b
        self.wn = math.pi**2*(n**2/a**2+m**2/b**2)

    def __call__(self, x, y):
        try:
            return ((self.wn-Constants.k))*torch.sin(math.pi*self.n*x/self.a)*torch.sin(math.pi*self.m*y/self.b)
        except:
            return ((self.wn-Constants.k))*np.sin(math.pi*self.n*x/self.a)*np.sin(math.pi*self.m*y/self.b)

    def solve_helmholtz(self, domain):
        f = self.call
        mode = self.wn
        b = f(domain['interior_points'][:, 0], domain['interior_points'][:, 1])

        assert abs(mode-Constants.k) > 1e-6
        return 1/(mode-Constants.k)*b


class sum_sin:
    def __init__(self, modes, a, b, weights=None):
        self.modes = modes
        self.a = a
        self.b = b
        self.funcs = [sin_function(mode[0], mode[1], self.a, self.b)
                      for mode in self.modes]
        self.weights = [1/len(modes)]*len(modes)
        if weights is not None:
            self.weights = weights

    def __call__(self, X):
        x=X[0]
        y=X[1]
        count = 0
        for weight, f in zip(self.weights, self.funcs):
            count += weight*f(x, y)
        return count

    def solve_helmholtz(self, domain):
        count = 0
        for weight, f in zip(self.weights, self.funcs):
            mode = f.wn
            b = f(domain['interior_points'][:, 0],
                       domain['interior_points'][:, 1])

            assert abs(mode-Constants.k) > 1e-6
            count += weight/(mode-Constants.k)*b

        return count


class Test_function:
    def __init__(self, domain, index):
        self.index=index
        self.domain=domain
        self.v = [domain['generators'][i] for i in range(len(domain['generators']))]
        self.n=len(self.v)
        self.x = symbols('x')
        self.y = symbols('y')

        self.edges=[]
        self.length=[]
        for i in range(self.n):
            E=self.v[(i+1)%self.n]-self.v[i]
            self.length.append(np.linalg.norm(E))
            N=[-E[1],E[0]]
            term=(self.x-self.v[i][0])*N[0]+(self.y-self.v[i][1])*N[1]
            self.edges.append(term)
        self.expr=1
        for e,l in zip(self.edges, self.length):
            self.expr*=sin(math.pi*self.index*e/l)
            # self.expr*=self.index*e
        self.d_expr=self.diff(self.expr)    
  

    def diff(self, expr):
        return -diff(expr, 'x', 2)-diff(expr, 'y', 2)-Constants.k*expr

    def __call__(self,x,y):

        return float(((self.d_expr).subs([('x', x), ('y', y)])).evalf())

    def solve_helmholtz(self, domain):
        return np.array([float(((self.expr).subs([('x', x), ('y', y)])).evalf()) for 
         x, y in zip(domain['interior_points'][:, 0], domain['interior_points'][:, 1])
         ])
    


class fourier_bessel:
    def __init__(self, domain, n):
        self.n=n
        self.domain=domain
        self.ev=[np.sqrt(l) for l in domain['ev']]
        self.vertices=list(domain['generators'])
        # self.vertices=[v-self.temp_vertices[0] for v in self.temp_vertices]
        # self.radi=[np.sqrt(v[0] ** 2 + v[1] ** 2) for v in self.vertices]
        self.theta = [np.arctan2(v[1], v[0]) for v in self.vertices]
        self.delta_phi=[theta-self.theta[0] for theta in self.theta[1:]]
        self.m=[math.pi*self.n/d_phi for d_phi in self.delta_phi]
    def __call__(self,x,y):
        # v=[x-self.temp_vertices[0][0],y-self.temp_vertices[0][1]]
        v=[x,y]
        r=np.sqrt(v[0] ** 2 + v[1] ** 2)
        theta=np.arctan2(v[1], v[0])
        k=np.sqrt(self.ev[-1])
        value=1
        
        for i in range(len(self.delta_phi)):
            value*=scipy.special.jv(self.m[i],k*r)*np.sin(self.m[i]*(theta-self.theta[0]))
        return scipy.special.jv(self.m[0],k*r)*np.sin(self.m[0]*(theta-self.theta[0]))    

       
        
        return value*np.sin(self.m*(theta-self.theta[0]))




        
class gaussian:
    def __init__(self, point):
        self.point=point
        self.r=Constants.h*4
    def __call__(self, x,y): 
        return np.exp(-((x-self.point[0])**2+(y-self.point[1])**2)/self.r**2)    
    
    def solve_helmholtz(self, domain):
        return solve_helmholtz(domain, self.__call__)



def solve_helmholtz(domain, f):
        b=f(domain['interior_points'][:,0], domain['interior_points'][:,1])
        M=domain['M']
        A = -M - Constants.k * scipy.sparse.identity(M.shape[0])
        return scipy.sparse.linalg.spsolve(A, b)*100
    


# domain=torch.load(Constants.path + "polygons/1.pt")
# # print(domain['hot_points'][::10,:].shape)

# x=domain['hot_points'][:,0]
# y=domain['hot_points'][:,1]
# n=20

# all_ind=set(list(range(y.shape[0])))
# ind1=set(y.argsort()[n:-n])
# ind2=set(x.argsort()[n:-n])
# good_ind=list(ind1.intersection(ind2))
# sources=spread_points(15,domain['hot_points'][good_ind])
# x=M[:,0]
# y=M[:,1]



# plt.scatter(x[1:],y[1:])
# plt.scatter(x[0],y[0],c='r')
# plt.show()
# func=gaussian(domain['hot_points'][521])
# g=func.solve_helmholtz(domain)
# u=np.array(list(map(func,x,y)))
# plt.scatter(x, y,c=u)
# plt.colorbar()
# plt.show()


