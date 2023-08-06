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
from functions.functions import sin_function, Test_function, sum_sin


def generate_domains():
    pass

    x_length = list(np.linspace(0.3, 1, 20))
    y_length = [1]
    for i, a in enumerate(x_length):
        for j, b in enumerate(y_length):

            rect = rectangle(a, b)
            rect.create_mesh(Constants.h)
            rect.save(Constants.path + "polygons/rect"+str(i)+str(j)+".pt")
    # rect=rectangle(1,1)
    # rect.create_mesh(Constants.h)
    # rect.save(Constants.path + "polygons/rect_train_"+str(0)+".pt")
    # p=Polygon(np.array([[0, 0], [1, 0], [1, 1 / 2], [1 / 2, 1 / 2], [1 / 2, 1], [0, 1]]))
    # p.create_mesh(Constants.h/2)
    # p.save(Constants.path + "polygons/lshape.pt")

#


# generate_domains()
# for large data
def create_data(domains, Functions, dir_path):

    for domain, funcs in zip(domains, Functions):

        for i, f in enumerate(funcs):
            x2 = np.array(list(map(f, domain['hot_points'])))

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

    # polygons_files_names=list(set(polygons_files_names)-set([Constants.path + "polygons/lshape.pt"]))
    test_domains_path = [Constants.path + "hints_polygons/trapz.pt"]
    train_domains_path = list(
        set(polygons_files_names) - set(test_domains_path))


    # train_modes_temp=[ (1,1),(2,1)]
    train_modes_temp = []
    for i in range(1, 3, 1):
        for j in range(1, 2, 1):
            train_modes_temp.append((i, j))
    num_samples = 20

    a = np.random.rand(num_samples, len(train_modes_temp))
    train_weights = [list(a[i]/np.sum(a[i])) for i in range(num_samples)]
    train_modes = [train_modes_temp]*num_samples

    test_modes_temp = train_modes_temp
    b = np.random.rand(1, len(test_modes_temp))
    # test_weights=[train_weights[0]]
    test_weights = [list(b[0]/np.sum(b[0]))]
    test_modes = [test_modes_temp]

    train_domains = [torch.load(name) for name in train_domains_path]
    # train_functions=[[sin_function(ind[0], ind[1], rect['a'], rect['b']) for ind in train_modes] for rect in train_domains ]
    train_functions = [[sum_sin(ind, rect['a'], rect['b'], weights) for weights, ind in zip(
        train_weights, train_modes)] for rect in train_domains]

    # rect version:

    # test_domains = [torch.load(name) for name in test_domains_path]
    # test_functions = [[sum_sin(ind, rect['a'], rect['b'], weights) for weights, ind in zip(
    #     test_weights, test_modes)] for rect in test_domains]


    test_domains=[torch.load(name) for name in test_domains_path]
    test_functions=[[Test_function(domain)] for domain in test_domains ]

    create_data(test_domains, test_functions, Constants.path+'test/')
    # create_data(train_domains, train_functions, Constants.path+'train/')


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
