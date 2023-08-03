import datetime
import time

import cmath
from pylab import figure, cm
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh



from utils import *
from constants import Constants
from geometry import rectangle, circle, Polygon
from functions.functions import sin_function, Test_function


# circle_hot_points=circle().hot_points





def generate_domains():
      
        x_length=[1]
        y_length=[1]
        for i,a in enumerate(x_length):
                for j,b in enumerate(y_length):

                    rect=rectangle(a,b)
                    rect.create_mesh(Constants.h)
                    rect.save(Constants.path + "polygons/rect"+str(i)+str(j)+".pt")
        # rect=rectangle(1,1)
        # rect.create_mesh(Constants.h)
        # rect.save(Constants.path + "polygons/rect_train_"+str(0)+".pt")
        # p=Polygon(np.array([[0, 0], [1, 0], [1, 1 / 2], [1 / 2, 1 / 2], [1 / 2, 1], [0, 1]]))
        # p.create_mesh(Constants.h/2)
        # p.save(Constants.path + "polygons/lshape.pt")

        # p=Polygon(np.array([[0, 0], [1, 0], [1, 1 ], [0, 1 ]]))
        # p.create_mesh(Constants.h)
        # p.save(Constants.path + "polygons/rect_1.pt")
        # p=Polygon(np.array([[0, 0], [0.5, 0], [0.25, 1]]))
        # p.create_mesh(Constants.h)
        # p.save(Constants.path + "polygons/triangle.pt")
#


# generate_domains()
# for large data
def create_data(domains, Functions, modes, dir_path):

    for domain, funcs in zip(domains, Functions):
        
            
            # transform = Map_circle_to_polygon(domain['generators']).call
            # xi=np.array(list(map(transform, circle_hot_points[:,0], circle_hot_points[:,1])))
            for i,f in enumerate(funcs):
                x2=np.array(list(map(f.call,domain['hot_points'][:,0], domain['hot_points'][:,1])))

                # x1=np.array(list(map(f,xi[:,0], xi[:,1])))
                x1=np.array([ [domain['moments'][l].real,domain['moments'][l].imag] for l in range(len(domain['moments']))])
             
                if domain['type']=='rectangle':
                        
                        x3=rectangle.solve_helmholtz_equation(f.call,domain, f.wn)
                        
                else:
                        x3= Polygon.solve_helmholtz_equation(f.call,domain)

                for j in range(0,domain['interior_points'].shape[0],1):
                  
                    y=np.expand_dims(domain['interior_points'][j],-1)
                    y_ev=np.expand_dims(domain['ev'],-1)
                    f_gs=x1
                    f_domain=np.expand_dims(x2,-1)
                    output_value=x3[j]
                    name = (
                    str(datetime.datetime.now().date())
                    + "_"
                    + str(datetime.datetime.now().time()).replace(":", ".")
                        )
                    save_file([np_to_torch(y),np_to_torch(y_ev),np_to_torch(f_gs),np_to_torch(f_domain)],dir_path+'input/', name)
                    save_file([np_to_torch(output_value)],dir_path+'output/', name)

   
if __name__=="__main__": 
   
    polygons_files_names = extract_path_from_dir(Constants.path + "polygons/")
    
    # polygons_files_names=list(set(polygons_files_names)-set([Constants.path + "polygons/lshape.pt"]))
    test_domains_path=[polygons_files_names[0] ]
    train_domains_path=[polygons_files_names[0] ]

    # train_domains_path = list(set(polygons_files_names) ) 
    # train_domains_path = list(set(polygons_files_names) - set(test_domains_path)) 
    # train_domains_path = [polygons_files_names[2] ]


    # test_domains_path = [Constants.path + "polygons/lshape.pt"]
    # train_domains_path = list(set(polygons_files_names) - set(test_domains_path))      

    # train_domains_path=[Constants.path + "polygons/rect_train_"+str(0)+".pt"]
    # test_domains_path=[Constants.path + "polygons/rect_train_"+str(0)+".pt"]
    # train_domains_path=[Constants.path + "polygons/lshape.pt"]
    # test_domains_path=[Constants.path + "polygons/lshape.pt"]

    # train_modes=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)]
    train_modes=[]
    for i in range(1,5,1):
            for j in range(1,5,1):    
                    train_modes.append((i,j))

    test_modes=[(5,5)]                

    train_domains=[torch.load(name) for name in train_domains_path]
    train_functions=[[sin_function(ind[0], ind[1], rect['a'], rect['b']) for ind in train_modes] for rect in train_domains ]

    # rect version:

    test_domains=[torch.load(name) for name in test_domains_path]
    test_functions=[[sin_function(ind[0], ind[1], rect['a'], rect['b']) for ind in test_modes] for rect in test_domains ]


    # lshape version:
    # train_domains=[torch.load(name) for name in train_domains_path]
    # train_functions=[[Test_function(domain['generators'], solution=False)] for domain in train_domains ]
    # train_modes=[[sin_function(ind[0], ind[1], 1, 1).wn for ind in test_modes] for rect in train_domains ]
    # 
    # test_domains=[torch.load(name) for name in test_domains_path]
    # test_functions=[[Test_function(domain['generators'], solution=False)] for domain in test_domains ]
    # test_modes=[]
    create_data(test_domains, test_functions, test_modes, Constants.path+'test/')
    create_data(train_domains, train_functions, train_modes, Constants.path+'train/')







# if __name__=="__main__":        
          

    # fig,ax=plt.subplots(1)
    # ev_train=[domain['principal_ev'] for domain in train_domains]
    # ev_test=[domain['principal_ev'] for domain in test_domains]
    # ax.scatter(ev_train,np.zeros(len(train_domains)), color='black')
    # ax.scatter(ev_test, np.ones(len(test_domains)), color='red')
    # ev_train=[domain['ev'] for domain in train_domains]
    # ev_test=[domain['ev'] for domain in test_domains]
    # for l in ev_train:
    #        ax.scatter(l[1],l[2], color='black')
    # for l in ev_test:
    #        ax.scatter(l[1],l[2], color='red')           
    # ax.set_title(f'eigenvalues')
    # plt.show()

# p=Polygon(np.array([[0, 0], [1, 0], [1, 1 / 4], [1 / 4, 1 / 4], [1 / 4, 1], [0, 1]]))
# p.create_mesh(Constants.h)
# p.save(Constants.path + "polygons/lshape.pt")
# domain=torch.load(Constants.path + "polygons/rect_train_"+str(3)+".pt")
# domain=torch.load(Constants.path + "polygons/triangle.pt")
# # domain=torch.load(Constants.path + "polygons/triangle.pt")
# n=len(domain['generators'])
# a=abs(cmath.exp(1 * 2 * math.pi * 1j / n)-1)
# area=(n*a**2/4)*(1/np.tan(math.pi/n))
# reg_vertices = [np.sqrt(math.pi/area)*cmath.exp(k * 2 * math.pi * 1J / n) for k in
#                 range(n)]
# x = [reg_vertices[i].real  for i in range(n)]
# y = [reg_vertices[i].imag  for i in range(n)]

# # # eta=np.array(list(map(g, Constants.points_on_circle[:,0], Constants.points_on_circle[:,1])))

# transform = Map_circle_to_polygon(domain['generators']).call2

# xi=np.array(list(map(transform, circle_hot_points[:,0], circle_hot_points[:,1])))

# # x2=np.array(list(map(f,domain['hot_points'][:,0], domain['hot_points'][:,1])))

# # x1=np.array(list(map(f,xi[:,0], xi[:,1])))

# # plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1],color='red')

# plt.scatter(xi[:,0], xi[:,1],color='black')
# plt.scatter(domain['generators'][:,0], domain['generators'][:,1],color='red')
# # plt.scatter(circle_hot_points[:,0], circle_hot_points[:,1],color='red')
# plt.scatter(x, y,color='green')
# # print(max(abs(circle_hot_points[:,0])))

# plt.show()

