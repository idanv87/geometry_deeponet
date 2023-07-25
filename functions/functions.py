import math
import numpy as np
import torch
import sys
import os

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
        self.function=1
        for v in self.v:
            self.function*=sin(x-v)*sin(y-v)

    
    def diff(self, expr):
        return -diff(expr,'x',2)-diff(expr,'y',2)-Constants.k*expr
    
    def __call__(self, X, Y):
        if self.solution== True:
            return float((self.function.subs([('x', X),('y', Y)])).evalf())
        if self.solution== False:
            return float(((self.diff(self.function)).subs([('x', X),('y', Y)])).evalf())


                  






# domain=torch.load(Constants.path + "polygons/lshape.pt")  
# f=Test_function(domain['generators'],False)
# print(np.array(list((map(f,domain['hot_points'][:,0], domain['hot_points'][:,1])))))
      



