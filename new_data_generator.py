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
from geometry import rectangle, circle, Polygon


def generate_domains():
        # for a in list(np.linspace(0.5,3,50)):
        #     rect=rectangle(a,1/a)
        #     rect.create_mesh(Constants.h)
        #     rect.save(Constants.path + "polygons/rect_train_"+str(a)+".pt")
        rect=rectangle(1,1)
        rect.create_mesh(Constants.h)
        rect.save(Constants.path + "polygons/rect_train_"+str(1)+".pt")
        p=Polygon(np.array([[0, 0], [1, 0], [1, 1 / 4], [1 / 4, 1 / 4], [1 / 4, 1], [0, 1]]))
        p.create_mesh(Constants.h)
        p.save(Constants.path + "polygons/lshape.pt")

# generate_domains()

# for large data
# polygons_files_names = extract_path_from_dir(Constants.path + "polygons/")
# test_domains = [Constants.path + "polygons/lshape.pt"]
# train_domains = list(set(polygons_files_names) - set(test_domains))      

train_domains=[Constants.path + "polygons/rect_train_"+str(1)+".pt"]
test_domains=[Constants.path + "polygons/rect_train_"+str(1)+".pt"]

train_modes=[]
for i in range(1,10,2):
        for j in range(1,10,2):
                train_modes.append((i,j))

test_modes=[(1,1)]                

train_domains=[torch.load(name) for name in train_domains]
train_functions=[[sin_function(ind[0], ind[1], rect['a'], rect['b']).call for ind in train_modes] for rect in train_domains ]
train_modes=[[sin_function(ind[0], ind[1], rect['a'], rect['b']).wn for ind in train_modes] for rect in train_domains ]

test_domains=[torch.load(name) for name in test_domains]
test_functions=[[sin_function(ind[0], ind[1], rect['a'], rect['b']).call for ind in test_modes] for rect in test_domains ]
test_modes=[[sin_function(ind[0], ind[1], rect['a'], rect['b']).wn for ind in test_modes] for rect in test_domains ]


# lshape version:
# test_domains=[torch.load(name) for name in test_domains]
# test_functions=[[Test_function() for ind in test_modes] for rect in test_domains ]
# test_modes=[[sin_function(ind[0], ind[1], 1, 1).wn for ind in test_modes] for rect in test_domains ]


circle_hot_points=circle().hot_points

def create_train_data(train_domains, train_functions, train_modes, dir_path):

    for domain, funcs in zip(train_domains, train_functions):
            
            transform = Map_circle_to_polygon(domain['generators']).call
            xi=np.array(list(map(transform, circle_hot_points[:,0], circle_hot_points[:,0])))
            for i,f in enumerate(funcs):
                
                x2=f(domain['hot_points'][:,0], domain['hot_points'][:,1])

                x1=f(xi[:,0], xi[:,1])

                if domain['type']=='rectangle':
                        x3=rectangle.solve_helmholtz_equation(f,domain, train_modes[0][i])
                else:
                        x3= Polygon.solve_helmholtz_equation(f,domain)

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
    fig,ax=plt.subplots(1)
    ev_train=[domain['principal_ev'] for domain in train_domains]
    ev_test=[domain['principal_ev'] for domain in test_domains]
    ax.scatter(ev_train,np.zeros(len(train_domains)), color='black')
    ax.scatter(ev_test, np.ones(len(test_domains)), color='red')
    ax.set_title(f'eigenvalues')
    plt.show()


    # self.transform = Map_circle_to_polygon(target_polygon).call
    #                 list(
    #                 map(
    #                     self.transform, x_interior_points[:, 0], x_interior_points[:, 1]
    #                 )
    #             )




# rect=rectangle(1,1)
# value=rect.apply_function_on_rectangle(sin_function(3,2,rect.a,rect.b).call)
# pass




