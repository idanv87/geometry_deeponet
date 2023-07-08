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
                self.triangles=triangles
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
                return scipy.sparse.linalg.eigs(-self.M[self.interior_indices][:,self.interior_indices],k=6,return_eigenvectors=False, which='SM')[:-1]
        
        def solve_helmholtz(self,f):
               A=-self.M[self.interior_indices][:,self.interior_indices]-scipy.sparse.identity(len(self.interior_indices))
               return scipy.sparse.linalg.spsolve(A,f[self.interior_indices])


        def is_legit(self):
                if  np.min(abs(self.sc[1].star.diagonal()))>0:
                        return True
                else:
                        return False
                

        # def plot_polygon(self):
        #       dmsh.show(self.vertices, self.triangles, self.geo)

        def plot_polygon(self):
              plt.scatter(self.vertices[self.boundary_indices,0], self.vertices[self.boundary_indices,1], color='red')

              dmsh.show(self.vertices, self.triangles, self.geo)
            #   plt.scatter(self.vertices[self.interior_indices,0], self.vertices[self.interior_indices,1], color='black')
              
              plt.show()      


class data_point:

       def __init__(self, path):
            v=np.array(generate_polygon((0.,0.), Constants.radius, Constants.var_center,Constants.var_angle,Constants.num_edges ))
            v=(np.sqrt(math.pi)/np.sqrt(polygon_centre_area(v)))*v
            v[:,0]-=np.mean(v[:,0])
            v[:,1]-=np.mean(v[:,1])
            geo = dmsh.Polygon(v)
            X, cells = dmsh.generate(geo, Constants.h)
            X, cells = optimesh.optimize_points_cells(X, cells, "CVT (full)", 1.0e-10, 80)
            self.polygon=Polygon(X, cells, v)
         
            self.path=path
            self.value={'eigen':None, 'interior_points':X, 'generators':v, 'X':X, 'cells':cells,'ind':None}
           
            if self.polygon.is_legit():
                
                self.value['eigen']=self.polygon.ev
                interior_points=X[self.polygon.interior_indices]
                ind=interior_points[:, 0].argsort()
                self.value['ind']=ind
                self.value['interior_points']=interior_points[ind]
                self.save_data()
                #
                
       def save_data(self):
              torch.save(self.value, self.path)
            #   with open(self.path, 'wb') as file:
            #      pickle.dump(self.value, file)
        
#         
            
def creat_polygons_data(num_samples):
      for i in range(num_samples):
            uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
            path=Constants.path+'polygons/'+uniq_filename+'.pt'
            data_point(path) 

if __name__=='__main__':
      pass
#    creat_polygons_data(5) 

polygons_dir=Constants.path+'polygons/'
polygons_raw_names=next(os.walk(polygons_dir), (None, None, []))[2]
polygons_files_names=[n for n in polygons_raw_names if n.endswith('.pt')]
all_eigs=[torch.load(polygons_dir+name)['eigen'][-1] for name in polygons_files_names]
points=spread_points(Constants.num_control_polygons, np.vstack((all_eigs,all_eigs)).T)[:,0]

control_ind=[all_eigs.index(points[i]) for i in range(len(points))]

control_polygons=set([polygons_files_names[i] for i in control_ind])
test_polygons=set(random.sample(polygons_files_names,2))
train_polygons=set(polygons_files_names)-test_polygons-control_polygons
# Mu=create_mu()[:2]
Mu=[(0,0)]
# print(Mu)



# plt.scatter(all_eigs, np.array(all_eigs)*0, color='black')
# plt.scatter(points,np.array(points)     *0,color='red')
# plt.show()

if __name__=='__main__':
   fig, axs = plt.subplots(2,len(control_polygons))
   for j, name in enumerate(control_polygons):
        
        p=torch.load(polygons_dir+name)
        coord =[p['generators'][i] for i in range(p['generators'].shape[0])]
        coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        xs, ys = zip(*coord) #create lists of x and y values
        axs[0,j].plot(xs,ys) 
        
        # p=torch.load(polygons_dir+list(train_polygons)[j])
        # coord =[p['generators'][i] for i in range(p['generators'].shape[0])]
        # coord.append(coord[0]) #repeat the first point to create a 'closed loop'
        # xs, ys = zip(*coord) #create lists of x and y values
        # axs[1,j].plot(xs,ys) 
        
   plt.show()




# v=np.array(generate_polygon((0.,0.), 3, 0,0,10))
# # v=np.array([0,0])
# v=(np.sqrt(math.pi)/np.sqrt(polygon_centre_area(v)))*v
# v[:,0]-=np.mean(v[:,0])
# v[:,1]-=np.mean(v[:,1])

# geo = dmsh.Polygon(v)
# X, cells = dmsh.generate(geo, 0.2)
# X, cells = optimesh.optimize_points_cells(X, cells, "CVT (full)", 1.0e-10, 80)
# dmsh.show(X, cells, geo)









# creat_train_data(1)  
# creat_main_polygons_data(1)




# def creat_train_data(num_samples):
#       for i in range(num_samples):
#             uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
#             path=Constants.path+'train/'+uniq_filename+'.pkl'
#             data_point(path)

# def creat_main_polygons_data(num_samples):
#       for i in range(num_samples):
#             uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
#             path=Constants.path+'main_polygons/'+uniq_filename+'.pkl'
#             data_point(path)    











# p=Polygon(X, cells, v)  
# print(p.is_legit())



              

               
  



# p.plot_polygon()
# print(p.interior_indices)
# print(p.boundary_indices)
