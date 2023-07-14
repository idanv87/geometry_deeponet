import datetime
import time 


import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh



from utils import *
from constants import Constants
from pathlib import Path



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
                return scipy.sparse.linalg.eigs(-self.M[self.interior_indices][:,self.interior_indices],k=Constants.ev_per_polygon+1,return_eigenvectors=False, which='SM')
        
        # def solve_helmholtz(self,f):
        #        A=-self.M[self.interior_indices][:,self.interior_indices]-Constants.k*scipy.sparse.identity(len(self.interior_indices))
        #        return scipy.sparse.linalg.spsolve(A,f[self.interior_indices])


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
                self.value={'eigen':None, 'interior_points':None, 'generators':v, 'X':X, 'cells':cells,'ind':None, 'interior_indices':self.polygon.interior_indices, 'M':self.polygon.M, 'legit':self.polygon.is_legit()}
                
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

# def create_special_polygons():
#      path=Constants.path+'special_polygons/rect.pt'
#      data_point(path, np.array([[0,0],[1,0],[1,1],[0,1]])) 
#      path=Constants.path+'special_polygons/special1.pt'
#      data_point(path, np.array(generate_polygon((0.,0.), Constants.radius, 0,0,15 ))
# ) 

#      data_point(path, np.array([[0,-1],[0.5,1],[0,1],[-0.5,0]])) 
          
               
  




def create_data_point(X,func,M, indices,legit):
            assert legit
            f=np.array(list(map(func, X[:,0],X[:,1])))  
            
                 
            u=solve_helmholtz(M,indices,f)

            
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

def create_main_polygons(control_polygons):
   x=[]

   for f in control_polygons:
       

        if os.path.isfile(f):
           
           df=torch.load(f)
        
           x.append(df)
   return x        


def create_data_points(control_polygons, train_polygons, train_or_test, func=None):
        assert train_or_test in set(['train', 'test', 'hints'])

        if train_or_test=='train':
                funcs=[Gaussian(mu).call for mu in create_mu()]
        if train_or_test=='test':
                funcs=[Test_function().call]    
        if train_or_test=='hints':
                funcs=[func]   
        
        main_polygons=create_main_polygons(control_polygons)
 
        data=[]
        for file in train_polygons:
        
                if os.path.isfile(file):
                        df=torch.load(file)
                
                for func in funcs:

                        f,u=create_data_point(df['X'],func,df['M'], df['interior_indices'],df['legit'])
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
                                
                                # data_names.append(name+'.pt')
                                
                                out_path=Constants.path+train_or_test
                                if train_or_test=='hints':
                                        data.append((np_to_torch(y),np_to_torch(ev_y[-Constants.ev_per_polygon:]),np_to_torch(f_x),np_to_torch(ev_x[-Constants.ev_per_polygon:])))
                                else:
                                        save_file(np_to_torch(y),out_path+'/y/', name)
                                        save_file(np_to_torch(ev_y),out_path+'/ev_y/', name)
                                        save_file(np_to_torch(f_x),out_path+'/f_x/', name)
                                        save_file(np_to_torch(ev_x),out_path+'/ev_x/', name)
                                        save_file(np_to_torch(output),out_path+'/output/', name)
                                
                                
                        
        return data


   
     

def create_special_polygons():
     path=Constants.path+'polygons/rect.pt'
     data_point(path, np.array([[0,0],[1,0],[1,1],[0,1]])) 
     path=Constants.path+'polygons/special1.pt'
     data_point(path, np.array(generate_polygon((0.,0.), Constants.radius, 0,0,8 ))) 


      
if __name__=='__main__':
        pass
        # create_special_polygons()
        # creat_polygons_data(5)


polygons_files_names=extract_path_from_dir(Constants.path+'polygons/')
test_polygons=[Constants.path+'polygons/rect.pt']
train_polygons= list(set(polygons_files_names)-set(test_polygons))

all_eigs=[torch.load(name)['eigen'][-1] for name in train_polygons]
points=spread_points(Constants.num_control_polygons, np.vstack((all_eigs,all_eigs)).T)[:,0]
control_ind=[all_eigs.index(points[i]) for i in range(len(points))]
control_polygons=set([train_polygons[i] for i in control_ind])


def add_new_polygon(train_or_test='train'):
     path=Constants.path+'polygons/rect2.pt'
     data_point(path, np.array([[0,0],[1,0],[1,1],[0,1]])) 
     create_data_points(control_polygons, [path], train_or_test='test')



if __name__=='__main__':
        pass
        create_data_points(control_polygons, train_polygons, train_or_test='train')
        create_data_points(control_polygons, test_polygons, train_or_test='test')
        print('finished creating data')
       




def plot_eigs():
   ev=[[],[],[]]
   lab=['train_polygons','control polygons', 'test' ]
   for i,type in enumerate([ train_polygons, control_polygons, test_polygons]):
     for name in type:
                p=torch.load(name)
                ev[i].append(p['eigen'][-1])

   plt.figure(figsize=(10, 7))  
   for i in range(3):
       plt.scatter(ev[i], np.array(ev[i])*0+i, label=lab[i])
   plt.legend()    
   plt.show()
   plt.title('principal eigenvalue distribution')


def plot_polygons(dir, name):
        print(len(dir))
        fig, axs = plt.subplots(len(dir))
        fig.suptitle(name)
        for j, name in enumerate(dir):
                p=torch.load(name)
                
                coord =[p['generators'][i] for i in range(p['generators'].shape[0])]
                print(' ')
                print(name)
                print(p['eigen'][-1])

                coord.append(coord[0]) #repeat the first point to create a 'closed loop'
                xs, ys = zip(*coord) #create lists of x and y values
                

                control_points=spread_points(Constants.pts_per_polygon, p['interior_points'])
                if len(dir)==1:
                  axs.plot(xs,ys) 
                  
                  axs.scatter(control_points[:,0], control_points[:,1])
                else:
                  axs[j].plot(xs,ys) 
                  axs[j].scatter(control_points[:,0], control_points[:,1])
                       
        # plt.title(str(name))        
           
# add_new_polygon()
# print(train_polygons)
# if __name__=='__main__': 
#   pass
#   plot_eigs()
#   plot_polygons(control_polygons, 'control_polygons')   
#   plot_polygons(test_polygons, 'test_polygons')   
#   plot_polygons(train_polygons, 'train_polygons')  
#   plt.show()

#         
        
 

 


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