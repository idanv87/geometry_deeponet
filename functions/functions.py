import math
from typing import Any
import numpy as np
import torch
import sys
import os
import scipy
from sympy import *
from sympy import sin, cos

sys.path.append(os.getcwd())
from constants import Constants


class sin_function:
    def __init__(self,n,m,a,b): 
        self.n=n
        self.m=m
        self.a=a
        self.b=b
        self.wn=math.pi**2*(n**2/a**2+m**2/b**2)
    def call(self,x,y, solve=False):
        if True:
            try:
                return ((self.wn-Constants.k))*torch.sin(math.pi*self.n*x/self.a)*torch.sin(math.pi*self.m*y/self.b) 
            except:
                return ((self.wn-Constants.k))*np.sin(math.pi*self.n*x/self.a)*np.sin(math.pi*self.m*y/self.b) 

        else:    
            try:
                return torch.sin(math.pi*self.n*x/self.a)*torch.sin(math.pi*self.m*y/self.b) 
            except:
                return np.sin(math.pi*self.n*x/self.a)*np.sin(math.pi*self.m*y/self.b) 
            



class Test_function:
    def __init__(self, generators, solution=True):
        self.v=set(generators[:,0])
        self.solution=solution
        x=symbols('x')
        y=symbols('y')
        self.function=100
        for v in self.v:
            self.function*=sin(x-v)*sin(y-v)

    
    def diff(self, expr):
        return -diff(expr,'x',2)-diff(expr,'y',2)-Constants.k*expr
    
    def __call__(self, X, Y):
        if self.solution== True:
            return float((self.function.subs([('x', X),('y', Y)])).evalf())
        if self.solution== False:
            return float(((self.diff(self.function)).subs([('x', X),('y', Y)])).evalf())


                  
class christofel:
    def __init__(self, n): 
        self.n= n
    def __call__(self, z):
        n=self.n
        try:
            assert abs(z)<1
   
            value1 = scipy.special.hyp2f1(1 / n, 2 / n, 1 + 1 / n, z ** n,
                                  out=None)
            value = z * (1 - z ** n) ** (2 / n) * (z ** n - 1) ** (-2 / n) * value1
            return value
        except:
            print('z has modulus greater than 1' )
            assert abs(z)<1


def Gauss_zeidel(A, b, x):
    ITERATION_LIMIT = 2
    # x = b*0
    for it_count in range(1, ITERATION_LIMIT):
        x_new = np.zeros_like(x, dtype=np.float_)
        # print(f"Iteration {it_count}: {x}")
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])

            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        # if np.allclose(x, x_new, rtol=1e-10):
        #     break
        x = x_new

    # print(f"Solution: {x}")
    # error = np.linalg.norm(abs(np.dot(A, x) - b))
    # print(error)
    #  it_count, np.max(abs(np.dot(A, x) - b))
    return x
          






# domain=torch.load(Constants.path + "polygons/lshape.pt")  
# f=Test_function(domain['generators'],False)
# # print(np.array(list((map(f,domain['hot_points'][:,0], domain['hot_points'][:,1])))))
      
# print(dtype(f(3,2)))



