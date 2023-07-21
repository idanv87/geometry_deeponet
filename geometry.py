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

class L_shape:
    def __init__(self,R=1):
        vertices=[]
        for i,r in enumerate(list(np.linspace(0,R,20))):
            for j,theta in enumerate(list(np.linspace(0,2*math.pi,20))):
                if i%4==0 and j%2 ==0:
                    vertices.append([r*math.cos(theta), r*math.sin(theta)])
        self.hot_points=np.array(vertices)


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


# rect=rectangle(1,2)
# # print(torch.tensor(rect.apply_function_on_rectangle(np.sin)))
# # rect.save(Constants.path + "polygons/rect3.pt")
# rect.create_mesh(1/20)
# print(rect.hot_points.shape)
# print(rect.interior_points.shape)

# rect.plot()
# rect.plot()