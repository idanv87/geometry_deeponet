import datetime
import os
import sys

from shapely.geometry import Polygon as Pol2
from pylab import figure
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from scipy.interpolate import Rbf
from scipy.spatial.distance import euclidean, cityblock
import cmath
current_path=os.path.abspath(__file__)
sys.path.append(current_path.split('deeponet')[0]+'deeponet/')
from utils import *
from constants import Constants
# from coords import Map_circle_to_polygon
from pydec.dec import simplicial_complex
from functions.functions import Test_function









    

    
class mesh:
    ''''
        points=np.random.rand(5,2)
        points=[points[i] for i in range(5)]
        mesh(points)
    '''
    def __init__(self, points):
        #  points (50,2) as list of arrays[[],[],[]..]

        self.points=points.copy()
        self.p=[]
        for i in range(len(self.points)):
            l=self.points.copy()
            point=l[i].reshape(1,2)
            l.pop(i)
            nbhd=np.array(l)
            #  50,2
            X=np.vstack((point,nbhd))
            values=np.hstack((1,np.zeros(nbhd.shape[0])))
            self.p.append(scipy.interpolate.RBFInterpolator(X,values))

def calc_coeff(vertices):
    n = len(vertices)
    a = []
    Z = []
    for v in vertices:
        Z.append(complex_version(v))
    for i in range(len(Z)):
        exp1 = Z[(i-1) % len(Z)]-Z[i]
        exp2 = Z[(i) % len(Z)]-Z[(i+1) % len(Z)]
        a.append((exp1.conjugate()/exp1-exp2.conjugate()/exp2)*(1j/2))
    return a, Z


def calc_moment(k, a, z):
    return np.sum([a[i]*(z[i]**k) for i in range(len(a))])



class Polygon:
    def __init__(self, generators):
        self.generators = generators
        self.n=self.generators.shape[0]
        self.geo = dmsh.Polygon(self.generators)
        self.moments = [calc_moment(k, calc_coeff(self.generators)[
                                    0], calc_coeff(self.generators)[1]) for k in range(300)]

        self.fourier_coeff = self.fourier()
        
        

    def create_mesh(self, h):
        # if np.min(calc_min_angle(self.geo)) > (math.pi / 20):
            
        X, cells = dmsh.generate(self.geo, h)

        X, cells = optimesh.optimize_points_cells(
            X, cells, "CVT (full)", 1.0e-6, 120
        )
        self.X=X
        self.cells=cells
    # else:
    #     self.plot()    
        # dmsh.show(X, cells, self.geo)

        self.vertices = X
        self.sc = simplicial_complex(X, cells)
        self.M = (
            (self.sc[0].star_inv)
            @ (-(self.sc[0].d).T)
            @ (self.sc[1].star)
            @ self.sc[0].d
        )

        self.boundary_indices = [i for i in range(self.generators.shape[0])]
        self.calc_boundary_indices()
        self.interior_indices = list(
            set(range(self.vertices.shape[0])) - set(self.boundary_indices)
        )
        self.interior_points = self.vertices[self.interior_indices]

        self.hot_points = spread_points(30, self.interior_points)
        self.hot_indices=[]
        for i in range(self.vertices.shape[0]):
            for j in range(self.hot_points.shape[0]):
                if (self.vertices[i]== self.hot_points[j]).all()==True:
                     self.hot_indices.append(i)
        self.hot_indices=list(set(self.hot_indices))
        

        self.ev = self.laplacian().real
        self.radial_functions=self.radial_basis()
        self.hot_radial_functions=self.hot_radial_basis()

    def calc_boundary_indices(self):
        for i in range(self.generators.shape[0], (self.vertices).shape[0]):
            if on_boundary(self.vertices[i], self.geo):
                self.boundary_indices.append(i)

    def laplacian(self):
        return scipy.sparse.linalg.eigs(
            -self.M[self.interior_indices][:, self.interior_indices],
            k=5,
            return_eigenvectors=False,
            which="SR",
        )



    def is_legit(self):
        if np.min(abs(self.sc[1].star.diagonal())) > 0:
            return True
        else:
            return False

    def save(self, path):
        assert self.is_legit()
        data = {
            "vertices":self.vertices,
            "ev": self.ev,
            "principal_ev": self.ev[-1], 
            "interior_points": self.interior_points,
            # "interior_points": self.interior_points[np.lexsort(np.fliplr(self.interior_points).T)],
            # "hot_points": self.hot_points,
            "hot_points": self.hot_points[np.lexsort(np.fliplr(self.hot_points).T)],
            "generators": self.generators,
            "M": self.M[self.interior_indices][:, self.interior_indices],
            'moments': self.moments,
            'radial_basis':self.radial_functions,
            'hot_radial_basis':self.hot_radial_functions,
             'angle_fourier':self.fourier_coeff,
            "legit": True,
            'type': 'polygon'
        }
        torch.save(data, path)

    def plot2(self):
        plt.scatter(self.interior_points[:, 0],
                    self.interior_points[:, 1], color='black')
        plt.scatter(self.hot_points[:, 0], self.hot_points[:, 1], color='red')
        plt.show()

    def radial_basis(self):
        m=mesh([self.vertices[i] for i in range(self.vertices.shape[0])])
        return [m.p[i] for i in self.interior_indices]
    
    def hot_radial_basis(self):
        m=mesh([self.vertices[i] for i in range(self.vertices.shape[0])])
        return [m.p[i] for i in self.hot_indices]
    
    def fourier(self):
        x1=self.generators[:,0]
        y1=self.generators[:,1]
        dx=[np.linalg.norm(np.array([y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]])) for k in range(x1.shape[0])]

        theta=[np.arctan2(y1[(k+1)%y1.shape[0]]-y1[k],x1[(k+1)%x1.shape[0]]-x1[k]) for k in range(x1.shape[0])]

        l=[h/np.sum(dx) for h in dx]

        coeff=step_fourier(l,theta)
        return coeff
    @classmethod
    def plot(cls,generators, title='no title was given'):
        assert generators.shape[1]==2
        x1=generators[:,0]
        y1=generators[:,1]
        polygon = Pol2(shell=[[x1[k],y1[k]] for k in range(x1.shape[0])],holes=None)
        fig, ax = plt.subplots()
        ax.set_title(title)
        plot_polygon(ax, polygon, facecolor='white', edgecolor='red')
        

    def plot_moments(self):
        X=[self.moments[i].real for i in range(len(self.moments))]
        Y=[self.moments[i].imag for i in range(len(self.moments))]
        fig,ax=plt.subplots(1)
        ax.plot(range(len(self.moments)), X, 'r', label='real part')
        ax.plot(range(len(self.moments)), Y, 'b', label='imaginary part')
        plt.legend()
        
    
    def plot_geo(self):
        dmsh.show(self.X, self.cells, self.geo)










    # for name in train_domains_path:
    #     analyze_momnets(name,'r')
    # for name in test_domains_path:
    #     analyze_momnets(name,'b')

    # plt.show()






 