from constants import Constants
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
    def __init__(self, domain):
        self.domain=domain
        self.v = [domain['generators'][i] for i in range(len(domain['generators']))]
        self.n=len(self.v)
        self.x = symbols('x')
        self.y = symbols('y')

        self.edges=[]
        for i in range(self.n):
            E=self.v[(i+1)%self.n]-self.v[i]
            N=[-E[1],E[0]]
            term=(self.x-self.v[i][0])*N[0]+(self.y-self.v[i][1])*N[1]
            self.edges.append(term)
        self.expr=1
        for e in self.edges:
            self.expr*=math.pi*sin(e)    
        self.d_expr=self.diff(self.expr)    
  

    def diff(self, expr):
        return -diff(expr, 'x', 2)-diff(expr, 'y', 2)-Constants.k*expr

    def __call__(self,X):

        return float(((self.d_expr).subs([('x', X[0]), ('y', X[1])])).evalf())

    def solve_helmholtz(self, domain):
        return np.array([float(((self.expr).subs([('x', x), ('y', y)])).evalf()) for 
         x, y in zip(domain['interior_points'][:, 0], domain['interior_points'][:, 1])
         ])



class christofel:
    def __init__(self, n):
        self.n = n

    def __call__(self, z):
        n = self.n
        try:
            assert abs(z) < 1

            value1 = scipy.special.hyp2f1(1 / n, 2 / n, 1 + 1 / n, z ** n,
                                          out=None)
            value = z * (1 - z ** n) ** (2 / n) * \
                (z ** n - 1) ** (-2 / n) * value1
            return value
        except:
            print('z has modulus greater than 1')
            assert abs(z) < 1


def Gauss_zeidel(A, b, x):
    ITERATION_LIMIT = 2
    # x = b*0
    for it_count in range(1, ITERATION_LIMIT):
        x_new = np.zeros_like(x, dtype=np.float_)
        # print(f"Iteration {it_count}: {x}")
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])

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
