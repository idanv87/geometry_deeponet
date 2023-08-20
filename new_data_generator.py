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
from functions.functions import sin_function, Test_function, sum_sin, gaussian


def generate_domains():
    # p=rectangle(0.3,1)
    # p.create_mesh(Constants.h)
    # p.save(Constants.path + "polygons/rect.pt")
    num_domains=2
    for j in range(num_domains):
        theta1=np.random.uniform(low=0.0, high=math.pi/2)
        theta2=np.random.uniform(low=math.pi/2, high=math.pi)
        theta3=np.random.uniform(low=math.pi, high=3*math.pi/2)
        theta4=np.random.uniform(low=3*math.pi/2, high=2*math.pi)
        theta=[theta1, theta2, theta3, theta4]
        # X=np.array([[1+np.cos(theta[i]), 1+np.sin(theta[i])] for i in range(len(theta))])
        # plt.scatter(X[:,0],X[:,1])
        # plt.show()
    
        p=Polygon(np.array([[np.cos(theta[i]), np.sin(theta[i])] for i in range(len(theta))]))
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
            x2 = np.array(list(map(f, domain['hot_points'][:,0], domain['hot_points'][:,1])))
          

            x1 = np.array([[domain['moments'][l].real, domain['moments'][l].imag]
                          for l in range(len(domain['moments']))])

            x3 = f.solve_helmholtz(domain)

            for j in range(0, domain['interior_points'].shape[0], 1):

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
    test_domains_path=[Constants.path + "polygons/1.pt"]
    train_domains_path=[Constants.path + "polygons/1.pt"]

    # train_domains_path=[polygons_files_names[2]]
    # train_domains_path=train_domains_path = list(set(polygons_files_names) - set(test_domains_path))

    # index_set=[1]
    # train_domains=[torch.load(name) for name in train_domains_path]
    # train_functions=[[Test_function(domain, index) for index in index_set] for domain in train_domains ]
    domain_temp=torch.load(Constants.path + "polygons/1.pt")
    x=domain_temp['hot_points'][:,0]
    y=domain_temp['hot_points'][:,1]
    n=70

    all_ind=set(list(range(y.shape[0])))
    ind1=set(y.argsort()[n:-n])
    ind2=set(x.argsort()[n:-n])
    good_ind=list(ind1.intersection(ind2))
    sources=list(spread_points(30,domain_temp['hot_points'][good_ind]))
    # plt.scatter(np.array(sources)[:,0], np.array(sources)[:,1])
    # plt.scatter(0, -0.3, color='r')
    # plt.show()
    train_domains=[torch.load(name) for name in train_domains_path]
    train_functions=[[gaussian(point) for point in sources[:]] for domain in train_domains ]

    test_domains=[torch.load(name) for name in test_domains_path]
    test_functions=[[gaussian(np.array([0,-0.3]))] for domain in test_domains ]
    # train_domains=test_domains
    # train_functions=test_functions
    
    create_data(test_domains, test_functions, Constants.path+'test/')
    create_data(train_domains, train_functions, Constants.path+'train/')


def view_moments():
    for d in test_domains:
        x1=d['interior_points'][:,0]
        x2=d['interior_points'][:,1]

        moments=np.array([[d['moments'][l].real, d['moments'][l].imag]
                          for l in range(len(d['moments']))])
        plt.plot(moments[:8,0]/8,'b')    
    for d in train_domains:
        x1=d['interior_points'][:,0]
        x2=d['interior_points'][:,1]

        moments=np.array([[d['moments'][l].real, d['moments'][l].imag]
                          for l in range(len(d['moments']))])
   
        plt.plot(moments[:8,1]/8,'r')

        # plt.plot(moments[:8,1]/8)
        
        # plt.scatter(x1,x2)
     
#     plt.show()
# view_moments()

pass
        

#


        # func=Test_function(d, index=1)
        # f=np.array(list(map(func, x1,x2)))
        # u=func.solve_helmholtz(d)