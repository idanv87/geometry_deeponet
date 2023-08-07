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
from functions.functions import sin_function, Test_function, sum_sin, Poly_function


def generate_domains():
    # p=rectangle(0.3,1)
    # p.create_mesh(Constants.h)
    # p.save(Constants.path + "polygons/rect.pt")
    num_domains=1
    for j in range(num_domains):
        theta1=np.random.uniform(low=0.0, high=math.pi/2)
        theta2=np.random.uniform(low=math.pi/2, high=math.pi)
        theta3=np.random.uniform(low=math.pi, high=3*math.pi/2)
        theta4=np.random.uniform(low=3*math.pi/2, high=2*math.pi)
        theta=[theta1, theta2, theta3, theta4]
        X=np.array([[1+np.cos(theta[i]), 1+np.sin(theta[i])] for i in range(len(theta))])
        # plt.scatter(X[:,0],X[:,1])
        # plt.show()
    
        p=Polygon(np.array([[1+np.cos(theta[i]), 1+np.sin(theta[i])] for i in range(len(theta))]))
        p.create_mesh(Constants.h)
        p.save(Constants.path + "polygons/"+str(j)+".pt")
       

    # for theta in np.linspace(0.0001,math.pi/2-0.0001,50):
    #     rect = rectangle(np.cos(theta), np.sin(theta))
        
    #     rect.create_mesh(Constants.h)
    #     rect.save(Constants.path + "polygons/rect"+str(theta)+".pt")

    # p1=Polygon(np.array([[0, 0], [0.6, 0], [0.6, 0.3 ], [0.3,0.3]])) 
    # p2=Polygon(np.array([[0, 0], [0.6, 0], [0.3, 0.3 ], [0.3,0.6]])) 
    # p1.create_mesh(Constants.h/2)
    # p2.create_mesh(Constants.h/2)
    # p1.save(Constants.path + "hints_polygons/trapz.pt")
    # p2.save(Constants.path + "hints_polygons/lshape.pt")


# generate_domains()

def create_data(domains, Functions, dir_path):

    for domain, funcs in zip(domains, Functions):

        for i, f in enumerate(funcs):
            x2 = np.array(list(map(f, domain['hot_points'])))

            x1 = np.array([[domain['moments'][l].real, domain['moments'][l].imag]
                          for l in range(len(domain['moments']))])

            x3 = f.solve_helmholtz(domain)

            for j in range(0, domain['interior_points'].shape[0], 2):

                y = np.expand_dims(domain['interior_points'][j], -1)
                y_ev = np.expand_dims(domain['ev'], -1)
                f_gs = x1
                f_domain = np.expand_dims(x2, -1)
                output_value = x3[j]
                name = (
                    str(datetime.datetime.now().date())
                    + "_"
                    + str(datetime.datetime.now().time()).replace(":", ".")
                )
                save_file([np_to_torch(y), np_to_torch(y_ev), np_to_torch(
                    f_gs), np_to_torch(f_domain)], dir_path+'input/', name)
                save_file([np_to_torch(output_value)],
                          dir_path+'output/', name)


if __name__ == "__main__":

    polygons_files_names = extract_path_from_dir(Constants.path + "polygons/")
    # domain=torch.load(polygons_files_names[1])
    # plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1])
    # plt.show()
    # test_domains_path=[Constants.path + "polygons/rect.pt"]
    # train_domains_path=[Constants.path + "polygons/rect.pt"]
    test_domains_path=[polygons_files_names[0]]
    train_domains_path=train_domains_path = list(set(polygons_files_names) - set(test_domains_path))
    
    # test_domains_path = [Constants.path + "hints_polygons/lshape.pt"]
    # train_domains_path = list(set(polygons_files_names) - set(test_domains_path))


    # train_modes_temp=[ (1,1),(2,1)]
    # train_modes_temp = []
    # for i in range(1, 3, 1):
    #     for j in range(1, 3, 1):
    #         train_modes_temp.append((i, j))
    # num_samples = 20

    # a = np.random.rand(num_samples, len(train_modes_temp))
    # train_weights = [list(a[i]/np.sum(a[i])) for i in range(num_samples)]
    # train_modes = [train_modes_temp]*num_samples

    # test_modes_temp = train_modes_temp
    # b = np.random.rand(1, len(test_modes_temp))
    # # test_weights=[train_weights[0]]
    # test_weights = [list(b[0]/np.sum(b[0]))]
    # test_modes = [test_modes_temp]

    # train_domains = [torch.load(name) for name in train_domains_path]
    # # train_functions=[[sin_function(ind[0], ind[1], rect['a'], rect['b']) for ind in train_modes] for rect in train_domains ]
    # train_functions = [[sum_sin(ind, rect['a'], rect['b'], weights) for weights, ind in zip(
    #     train_weights, train_modes)] for rect in train_domains]

    # rect version:

    # test_domains = [torch.load(name) for name in test_domains_path]
    # test_functions = [[sum_sin(ind, rect['a'], rect['b'], weights) for weights, ind in zip(
    #     test_weights, test_modes)] for rect in test_domains]

    train_domains=[torch.load(name) for name in train_domains_path]
    train_functions=[[Test_function(domain, index=1)] for domain in train_domains ]

    test_domains=[torch.load(name) for name in test_domains_path]
    test_functions=[[Test_function(domain, index=1)] for domain in test_domains ]

    create_data(test_domains, test_functions, Constants.path+'test/')
    create_data(train_domains, train_functions, Constants.path+'train/')


#