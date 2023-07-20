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
from geometry import rectangle


def generate_domains():
      rect=rectangle(1,2)
      rect.create_mesh(Constants.h)
      rect.save(Constants.path + "polygons/rect_train.pt")

      
generate_domains()
train_modes=[]
for i in range(1,5):
        for j in range(1,5):
                train_modes.append((i,j))

test_modes=[(1,2)]                

train_domains=[torch.load(Constants.path + "polygons/rect_train.pt")]
train_functions=[[sin_function(ind[0], ind[1], rect['a'], rect['b']).call for ind in train_modes] for rect in train_domains ]
train_modes=[[sin_function(ind[0], ind[1], rect['a'], rect['b']).wn for ind in train_modes] for rect in train_domains ]


test_domains=[torch.load(Constants.path + "polygons/rect_train.pt")]
test_functions=[[sin_function(ind[0], ind[1], rect['a'], rect['b']).call for ind in test_modes] for rect in test_domains ]
test_modes=[[sin_function(ind[0], ind[1], rect['a'], rect['b']).wn for ind in test_modes] for rect in test_domains ]


circle_hot_points=np.array([[0,0],[0,0.5]])

def create_train_data(train_domains, train_functions, train_modes, dir_path):

    for domain, funcs in zip(train_domains, train_functions):
        
           
           transform = Map_circle_to_polygon(domain['generators']).call
           xi=np.array(list(map(transform, circle_hot_points[:,0], circle_hot_points[:,0])))
           for i,f in enumerate(funcs):
                
                x2=f(domain['hot_points'][:,0], domain['hot_points'][:,1])
                x1=f(xi[:,0], xi[:,1])

                x3=rectangle.solve_helmholtz_equation(f,domain, train_modes[0][i])

                for j in range(domain['interior_points'].shape[0]):
                    y=np.expand_dims(domain['interior_points'][j],-1)
                    y_ev=np.expand_dims(domain['ev'],1)
                    f_circle=np.expand_dims(x1,-1)
                    f_domain=np.expand_dims(x2,-1)
                    output=x3[j]
                    name = (
                    str(datetime.datetime.now().date())
                    + "_"
                    + str(datetime.datetime.now().time()).replace(":", ".")
                        )
                    save_file([np_to_torch(y),np_to_torch(y_ev),np_to_torch(f_circle),np_to_torch(f_domain)],dir_path+'input/', name)
                    save_file([np_to_torch(output)],dir_path+'output/', name)

    return 
if __name__=="__main__":              
    create_train_data(train_domains, train_functions, train_modes, Constants.path+'train/')
    create_train_data(test_domains, test_functions, test_modes, Constants.path+'test/')
pass

    # self.transform = Map_circle_to_polygon(target_polygon).call
    #                 list(
    #                 map(
    #                     self.transform, x_interior_points[:, 0], x_interior_points[:, 1]
    #                 )
    #             )




# rect=rectangle(1,1)
# value=rect.apply_function_on_rectangle(sin_function(3,2,rect.a,rect.b).call)
# pass




