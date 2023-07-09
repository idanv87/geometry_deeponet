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
                return scipy.sparse.linalg.eigs(-self.M[self.interior_indices][:,self.interior_indices],k=6,return_eigenvectors=False, which='SM')
        
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

       def __init__(self, path, special=None):
            if special is not None:
                  v=special  
            else:
                  v=np.array(generate_polygon((0.,0.), Constants.radius, Constants.var_center,Constants.var_angle,Constants.num_edges ))

                  
            v=(np.sqrt(math.pi)/np.sqrt(polygon_centre_area(v)))*v
            v[:,0]-=np.mean(v[:,0])
            v[:,1]-=np.mean(v[:,1])
            geo = dmsh.Polygon(v)
            if np.min(calc_min_angle(geo))>(math.pi/8):
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

def create_special_polygons():
     path=Constants.path+'special_polygons/rect.pt'
     data_point(path, np.array([[0,0],[1,0],[1,1],[0,1]])) 
     path=Constants.path+'special_polygons/special1.pt'
     data_point(path, np.array(generate_polygon((0.,0.), Constants.radius, 0,0,15 ))
) 

          
               
  




def create_data_point(X,func,p):
            assert p.is_legit
            f=np.array(list(map(func, X[:,0],X[:,1])))  
            u=p.solve_helmholtz(f)
            return f,u

class branc_point:
    
    def __init__(self,f, main_polygons):
        self.f=f
        self.main_polygons=main_polygons
        self.b1, self.b2= self.calculate_branch()

    def calculate_branch(self):
        x=[]
        y=[]
   
        for p in self.main_polygons:
          x_interior_points=spread_points(Constants.pts_per_polygon, p['interior_points'])
         
          x.append(list(map(self.f, x_interior_points[:,0],x_interior_points[:,1]))  )
          y.append(p['eigen'])
        #   print(p['eigen'].shape)
        x=np.hstack(x).reshape((len(x), len(x[0])))
        # print(np.hstack(y).shape)
        y=np.hstack(y).reshape((len(y), len(y[0])))

        return x.transpose(), y.transpose()  

def create_main_polygons(control_polygons, polygons_dir):
   x=[]

   for filename in control_polygons:
        f = os.path.join(polygons_dir, filename)

        if os.path.isfile(f):
           
           df=torch.load(f)
        
           x.append(df)
   return x        


def create_data_points(control_polygons, train_polygons, train_or_test):
    if train_or_test=='train':
        funcs=[Gaussian(mu).call for mu in create_mu()]
        polygons_dir=Constants.path+'polygons/'
    else:
         funcs=[Test_function().call]    
         polygons_dir=Constants.path+'special_polygons/'
    
    data_names=[]
    main_polygons=create_main_polygons(control_polygons, Constants.path+'polygons/')

    for filename in train_polygons:
        fil= os.path.join(polygons_dir, filename)
        if os.path.isfile(fil):
           df=torch.load(fil)
           
           for func in funcs:
                # func=Gaussian(mu).call
                
                p=Polygon(df['X'], df['cells'], df['generators'])  
                f,u=create_data_point(df['X'],func,p)
               
                u=u[df['ind']]

                f_x=branc_point(func, main_polygons).b1
                ev_x=branc_point(func,main_polygons).b2
                for i in range(df['interior_points'].shape[0]):
                    
                   
                        
                    y=df['interior_points'][i].reshape([Constants.dim,1])
                    # ev_y=df['eigen'].reshape([Constants.ev_per_polygon,1])
                    ev_y=df['eigen'].reshape([df['eigen'].shape[0],1])
                    output=u[i]
                    # sort indices


                    name= str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
                    
                    data_names.append(name+'.pt')
                    
                    save_file(np_to_torch(y),Constants.path+'y/', name)
                    save_file(np_to_torch(ev_y),Constants.path+'ev_y/', name)
                    save_file(np_to_torch(f_x),Constants.path+'f_x/', name)
                    save_file(np_to_torch(ev_x),Constants.path+'ev_x/', name)
                    save_file(np_to_torch(output),Constants.path+'output/', name)
        if train_or_test=='train':
          save_file(data_names,Constants.path+'train_data_names/','train_data_names')
        else:  
          save_file(data_names,Constants.path+'test_data_names/','test_data_names')  


    
    return      
if __name__=='__main__':
      pass
      create_special_polygons()
      creat_polygons_data(5) 

polygons_dir=Constants.path+'polygons/'
polygons_raw_names=next(os.walk(polygons_dir), (None, None, []))[2]
polygons_files_names=[n for n in polygons_raw_names if n.endswith('.pt')]
all_eigs=[torch.load(polygons_dir+name)['eigen'][-1] for name in polygons_files_names]
points=spread_points(Constants.num_control_polygons, np.vstack((all_eigs,all_eigs)).T)[:,0]
control_ind=[all_eigs.index(points[i]) for i in range(len(points))]
control_polygons=set([polygons_files_names[i] for i in control_ind])
train_polygons=set(polygons_files_names)

polygons_dir=Constants.path+'special_polygons/'
polygons_raw_names=next(os.walk(polygons_dir), (None, None, []))[2]
polygons_files_names=[n for n in polygons_raw_names if n.endswith('.pt')]
test_polygons=set(polygons_files_names)

if __name__=='__main__':   
  pass
  create_data_points(control_polygons, train_polygons, train_or_test='train')
  create_data_points(control_polygons, test_polygons, train_or_test='test')


# print(Mu)










# v=np.array(generate_polygon((0.,0.), 3, 0,0,10))
# v=np.array([[0,0],[1,0],[1,1],[0,1]])

# v=(np.sqrt(math.pi)/np.sqrt(polygon_centre_area(v)))*v
# v[:,0]-=np.mean(v[:,0])
# v[:,1]-=np.mean(v[:,1])

# geo = dmsh.Polygon(v)
# X, cells = dmsh.generate(geo, 0.2)
# X, cells = optimesh.optimize_points_cells(X, cells, "CVT (full)", 1.0e-10, 80)


          
      
# p=Polygon(X, cells, v)

# print(p.s)
# print(polygon_centre_area(v))
# print(math.pi*2)
# # print(p.ev)
# dmsh.show(X, cells, geo)

# plt.show()







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

# if __name__=='__main__':
#    pol_type=test_polygons
#    polygons_dir=Constants.path+'special_polygons/'
#    fig, axs = plt.subplots(2,len(pol_type))
#    for j, name in enumerate(pol_type):
#         print(name)
        
#         p=torch.load(polygons_dir+name)
#         geo=dmsh.Polygon(p['generators'])
#         print(calc_min_angle(geo))
#         coord =[p['generators'][i] for i in range(p['generators'].shape[0])]
#         coord.append(coord[0]) #repeat the first point to create a 'closed loop'
#         xs, ys = zip(*coord) #create lists of x and y values
#         axs[j,0].plot(xs,ys) 
        
# #    x=list(np.linspace(-1,1,5))
# #    y=list(np.linspace(-1,1,5) )    
# #    for X in x:
# #          for Y in y:
# #                axs[j].scatter(X,Y) 
#    plt.show()    