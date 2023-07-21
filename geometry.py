import datetime
import time

from pylab import figure, cm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh
from pathlib import Path


from utils import *
from constants import Constants
from coords import Map_circle_to_polygon
from pydec.dec import simplicial_complex

# sin_function(ind[0], ind[1], df['a'], df['b']).call
# u=(1/(ev_s[j]-Constants.k))*np.array(list(map(funcs[j], df['interior_points'][:, 0], df['interior_points'][:, 1])))
class circle:
    def __init__(self,R=1):
        vertices=[]
        for i,r in enumerate(list(np.linspace(0,R,20))):
            for j,theta in enumerate(list(np.linspace(0,2*math.pi,20))):
                if i%4==0 and j%2 ==0:
                    vertices.append([r*math.cos(theta), r*math.sin(theta)])
        self.hot_points=np.array(vertices)
       

    def plot(self):
        plt.scatter(self.hot_points[:,0], self.hot_points[:,1],color='red')
        plt.title('unit circle')
        plt.show()    




class rectangle:
    
    def __init__(self, a,b): 
        self.a= a*np.sqrt(math.pi/a/b)
        self.b= b*np.sqrt(math.pi/a/b)
        self.generators=np.array([[0,0],[a,0],[a,b],[0,b]])
        self.area=self.a*self.b
        indices=[(1,1), (1,2),(2,1),(2,2)]
        self.ev=np.flip((math.pi**2)*np.array([(ind[0]/self.a)**2+(ind[1]/self.b)**2 for ind in indices]))
        self.principal_ev=self.ev[-1]
        self.interior_points=None
       

    def plot(self):
        plt.scatter(self.interior_points[:,0], self.interior_points[:,1],color='black')
        plt.scatter(self.hot_points[:,0], self.hot_points[:,1],color='red')
        plt.show()

    def create_mesh(self,h):
        n=int(1/h)
        x,y=np.meshgrid(np.linspace(0,self.a,n)[1:-1], np.linspace(0,self.b,n)[1:-1])
        vertices=[]
        hot_points=[]
        for i in range(n-2):
            for j  in range(n-2):
                vertices.append([x[i,j],y[i,j]])
                if i %Constants.hot_spots_ratio ==0 and j%Constants.hot_spots_ratio==0:
                    hot_points.append([x[i,j],y[i,j]])

        self.interior_points=np.array(vertices)   
        self.hot_points =np.array(hot_points)

    def apply_function_on_rectangle(self, func, plot=False):
        
        value=func(self.interior_points[:, 0], self.interior_points[:, 1])
        if plot:
            x,y=np.meshgrid(np.linspace(0,self.a), np.linspace(0,self.b))
            plt.imshow(func(x,y),extent=[0,self.a,0,self.b], cmap=cm.jet, origin='lower')
            plt.colorbar()
            plt.show()
        return value 

    
    def save(self, path):
            data={
                "ev": self.ev,
                "principal_ev":self.principal_ev,
                "interior_points": self.interior_points,
                "hot_points": self.hot_points,
                "generators": self.generators,
                "legit": True,
                'a':self.a,
                'b':self.b,
                'type':'rectangle'
             }
            torch.save(data, path)

    @classmethod
    def solve_helmholtz_equation(cls,f,*args):
        domain,mode= args
        b=f(domain['interior_points'][:,0], domain['interior_points'][:,1])

        assert abs(mode-Constants.k)>1e-6
        return 1/(mode-Constants.k)*b




class Polygon:
    def __init__(self, generators ):
        
        self.generators= (np.sqrt(math.pi) / np.sqrt(polygon_centre_area(generators))) * generators
        # v[:, 0] -= np.mean(v[:, 0])
        # v[:, 1] -= np.mean(v[:, 1])
        self.geo = dmsh.Polygon(self.generators)

    def create_mesh(self, h):
      
        if np.min(calc_min_angle(self.geo)) > (math.pi / 8):
            X, cells = dmsh.generate(self.geo, h )

            X, cells = optimesh.optimize_points_cells(
                X, cells, "CVT (full)", 1.0e-6, 120
            )
        self.vertices=X
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
        self.interior_points=self.vertices[self.interior_indices]

        self.hot_points=spread_points(int((int(1/Constants.h)-2)**2/(Constants.hot_spots_ratio**2)), self.interior_points)
        self.ev = self.laplacian().real
        
   

    def calc_boundary_indices(self):
        for i in range(self.generators.shape[0], (self.vertices).shape[0]):
            if on_boundary(self.vertices[i], self.geo):
                self.boundary_indices.append(i)

    def laplacian(self):
        return scipy.sparse.linalg.eigs(
            -self.M[self.interior_indices][:, self.interior_indices],
            k=Constants.ev_per_polygon,
            return_eigenvectors=False,
            which="SM",
        )
    


    def is_legit(self):
        if np.min(abs(self.sc[1].star.diagonal())) > 0:
            return True
        else:
            return False

    def save(self, path):
            assert self.is_legit()
            data={
                "ev": self.ev,
                "principal_ev":self.ev[-1],
                "interior_points": self.interior_points,
                "hot_points": self.hot_points,
                "generators": self.generators,
                "M":self.M[self.interior_indices][:, self.interior_indices],
                "legit": True,
                'type':'polygon'
             }
            torch.save(data, path)

    def plot(self):
        plt.scatter(self.interior_points[:,0], self.interior_points[:,1],color='black')
        plt.scatter(self.hot_points[:,0], self.hot_points[:,1],color='red')
        plt.show()

    @classmethod
    def solve_helmholtz_equation(self,func,domain):
        M=domain['M']
        b=func(domain['interior_points'][:,0], domain['interior_points'][:,1])
        A = -M - Constants.k * scipy.sparse.identity(
        len(domain['interior_points'])
    )

        return scipy.sparse.linalg.spsolve(A, b)


# rect=rectangle(1,2)
# rect.create_mesh(Constants.h)
# rect.save(Constants.path + "polygons/rect_train.pt")
# # p=Polygon(np.array([[0, 0], [1, 0], [1, 1 / 4], [1 / 4, 1 / 4], [1 / 4, 1], [0, 1]]))
# # p.create_mesh(Constants.h)
# # p.save(Constants.path + "polygons/lshape.pt")
# domain=torch.load(Constants.path + "polygons/rect_train.pt")
# plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1], color='red')
# plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1],color='black')
# plt.show()
# print(len(domain['hot_points']))
# print(int((int(1/Constants.h)-2)**2/(Constants.hot_spots_ratio**2)))
# domain=torch.load(Constants.path + "polygons/rect_train.pt")
# print(len(domain['hot_points']))

# u=Polygon.solve_helmholtz_equation(np.sin,domain)
# plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1])
# # plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1],color='red')

# plt.show()











# rect=rectangle(1,2)
# # print(torch.tensor(rect.apply_function_on_rectangle(np.sin)))
# # rect.save(Constants.path + "polygons/rect3.pt")
# rect.create_mesh(1/20)
# print(rect.hot_points.shape)
# print(rect.interior_points.shape)

# rect.plot()
# rect.plot()