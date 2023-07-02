import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh
import pickle
import datetime

from utils import *
from constants import Constants


class Polygon:
        def __init__(self, vertices, triangles, generators):
                self.geo=dmsh.Polygon(generators)
                self.vertices=vertices
                self.sc=simplicial_complex(vertices, triangles)
                self.M=(self.sc[0].star_inv)@(-(self.sc[0].d).T)@(self.sc[1].star)@self.sc[0].d
                self.generators=generators
                self.boundary_indices=[i for i in range(generators.shape[0])] 
                self.calc_boundary_indices()
                self.interior_indices=list(set(range(self.vertices.shape[0]))-set(self.boundary_indices))
                self.ev=self.laplacian().real

                
        def calc_boundary_indices(self):
              for i in range(self.generators.shape[0], (self.vertices).shape[0]):
                  if on_boundary(self.vertices[i],self.geo):
                     self.boundary_indices.append(i)    
        

        def laplacian(self):
                return scipy.sparse.linalg.eigs(-self.M[self.interior_indices][:,self.interior_indices], k=Constants.ev_per_polygon+1,return_eigenvectors=False, which='SM')[:-1]
        
        def solve_helmholtz(self,f):
               A=-self.M[self.interior_indices][:,self.interior_indices]-scipy.sparse.identity(len(self.interior_indices))
               return scipy.sparse.linalg.spsolve(A,f[self.interior_indices])


        def is_legit(self):
                if  np.min(abs(self.sc[1].star.diagonal()))>0:
                        return True
                else:
                        return False
                

        def plot_polygon(self):
               plt.scatter(self.vertices[:,0], self.vertices[:,1], color='black')
               plt.scatter(self.vertices[self.boundary_indices,0], self.vertices[self.boundary_indices,1], color='red')
               plt.show()

class data_point:

       def __init__(self, path):
            v=np.array(generate_polygon((0.,0.), Constants.radius, Constants.var_center,Constants.var_angle,Constants.num_edges ))
            v=(1/np.sqrt(polygon_centre_area(v)))*v
            geo = dmsh.Polygon(v)
            X, cells = dmsh.generate(geo, 0.2)
            X, cells = optimesh.optimize_points_cells(X, cells, "CVT (full)", 1.0e-10, 80)
            self.polygon=Polygon(X, cells, v)
          
            self.f=np.array(list(map(gaussian, X[:,0],X[:,1])))  
            self.path=path
            self.value={'eigen':None, 'points':None, 'u':None, 'gauss':gaussian}
            if self.polygon.is_legit():
                
                self.value['v']=self.polygon.generators
                self.value['eigen']=self.polygon.ev
                self.u=self.polygon.solve_helmholtz(self.f)
                interior_points=X[self.polygon.interior_indices]
                ind=interior_points[:, 0].argsort()
                self.value['points']=interior_points[ind]
                plt.scatter(X[:,0], X[:,1], color='red')
                plt.scatter( interior_points[:,0],  interior_points[:,1], color='black')
               
                plt.show()
                # print(len(ind))
                self.value['u']=self.u[ind]
                self.save_data()
                #
                
       def save_data(self):
              with open(self.path, 'wb') as file:
                 pickle.dump(self.value, file)
        
def creat_train_data(num_samples):
      for i in range(num_samples):
            uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
            path=Constants.path+'train/'+uniq_filename+'.pkl'
            data_point(path)

def creat_main_polygons_data(num_samples):
      for i in range(num_samples):
            uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
            path=Constants.path+'main_polygons/'+uniq_filename+'.pkl'
            data_point(path)            
            
      
# creat_train_data(2)  
# creat_main_polygons_data(1)











# v=np.array(generate_polygon((0.,0.), Constants.radius, Constants.var_center,Constants.var_angle,Constants.num_edges ))
# v[:,0]-=np.mean(v[:,0])
# v[:,1]-=np.mean(v[:,1])
# v=(1/np.sqrt(polygon_centre_area(v)))*v
# geo = dmsh.Polygon(v)
# X, cells = dmsh.generate(geo, 0.2)
# X, cells = optimesh.optimize_points_cells(X, cells, "CVT (full)", 1.0e-10, 80)
# p=Polygon(X, cells, v)  
# dmsh.show(X, cells, geo)
# x=np.linspace(-1,1,20)
# plt.plot(x,[gaussian(x[i]*20,0) for i in range(len(x))])
# plt.show()

              

               
  



# p.plot_polygon()
# print(p.interior_indices)
# print(p.boundary_indices)
