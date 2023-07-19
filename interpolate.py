import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
import torch
import scipy

from constants import Constants
from utils import (
    solve_helmholtz,
    Test_function,
    Gauss_zeidel,
    plot3d,
    spread_points,
    np_to_torch,
)
import matplotlib.pyplot as plt
from model import model
from data_generator import create_data_points, control_polygons

best_model = torch.load(
    Constants.path + "best_model/" + "2023-07-19_13.42.46.247856.pth"
)
model.load_state_dict(best_model["model_state_dict"])


class interplation_block:
    def __init__(self, points, values):
        assert points.shape[-1] == 2

        self.tck = scipy.interpolate.bisplrep(points[:, 0], points[:, 1], values)

    def call(self, X, Y):
        return scipy.interpolate.bisplev(X, Y, self.tck)


def hint_block(err, points, model):
    # p=torch.load(pol_path)

    # data=create_data_points(control_polygons,[pol_path], 'hints',interplation_block(points, err).call)
    data = create_data_points(
        control_polygons, [pol_path], "hints", Test_function().call
    )

    solution = []
    for dat in data:

        input = dat[:-1]

        input = [input[k].unsqueeze(0) for k in range(len(input))]
        solution.append(torch.stack(model(input)))

    solution = torch.stack(solution).detach().numpy()

    return solution


def GS_block(A, b, x):
    return Gauss_zeidel(A, b, x)


def hint_iterations(A, b, n_it, points, model):

    k_it = 0
    x = 0 * b
    for i in range(n_it):
        x_0 = x
        k_it += 1

        if (k_it % 30) > 0:
            x = GS_block(A, np.squeeze(b), np.squeeze(x_0))

        else:
            # x=np.squeeze(hint_block(np.squeeze(b), points, model))
            x = np.squeeze(x_0) + abs(
                np.squeeze(
                    hint_block(
                        np.dot(A, np.squeeze(x_0)) - np.squeeze(b), points, model
                    )
                )
            )

            pass

        # if   np.allclose(np.dot(A,x),b, 1e-5):
        if np.allclose(x_0, x, 1e-5):
            return k_it, np.dot(A, x) - b

    return k_it, np.dot(A, x) - b


# add_new_polygon()
pol_path = Constants.path + "polygons/lshape.pt"
p = torch.load(pol_path)
# print(len(p['interior_points']))
f = Test_function().call
b = np.array(list(map(f, p["interior_points"][:, 0], p["interior_points"][:, 1])))
A = -p["M"][p["interior_indices"]][
    :, p["interior_indices"]
] - Constants.k * scipy.sparse.identity(len(p["interior_indices"]))
A = A.todense()

it, err = hint_iterations(A, b, 100, p["interior_points"], model)
print(np.linalg.norm(err))


# data=create_data_points(control_polygons,[pol_path], 'hints',Test_function().call)
# solution=[]
# sol_data=[]
# for dat in data:

#         input=dat

#         input=[input[k].unsqueeze(0) for k in range(len(input))]
#         solution.append(torch.stack(model(input[:-1])))
#         sol_data.append(input[-1])

# sol2=solve_helmholtz(p['M'],p['interior_indices'], np.array(list(map(f, p['X'][:,0],p['X'][:,1]))) )
# solution=np.squeeze(torch.stack(solution).detach().numpy()   )

# b=np.array(list(map(f,p['interior_points'][:,0],p['interior_points'][:,1])))
# print(solution-sol_data)

# # b=np.sin(p['interior_points'][:,0])
# A=p['M'][p['interior_indices']][:,p['interior_indices']]
# A=A.todense()
# # # print(type(err))
# it,err=hint_iterations(A,b,200, p['interior_points'], model)
# print(model)


# p=torch.load(Constants.path+'polygons/special1.pt')
# # A=torch.tensor(p['M'].todense()[p['interior_indices']][:,p['interior_indices']])
# points=spread_points(40, p['interior_points'])

# grid_x=p['interior_points'][:,0]
# grid_y=p['interior_points'][:,1]
# values=np.sin(points[:,0]*points[:,1])
# # grid=scipy.interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
# tck=scipy.interpolate.bisplrep(points[:,0], points[:,1], values)
# print(scipy.interpolate.bisplev(3.5,15,tck))
# # fig = plt.figure()
# # ax = fig.add_subplot(projection='3d')
# # ax.scatter(p['X'][:,0],p['X'][:,1],np.sin(p['X'][:,0]*p['X'][:,1]), color='black')
# # ax.scatter(grid_x,grid_y,grid, color='red' , s=20)
# # print(grid)

# # # ax.scatter(points[:,0],points[:,1],values,color='red', marker='x')


# hint_iteration(A,b,10)


# x1=torch.rand(1,2,1)
# x2=torch.rand(1,10,1)
# x3=torch.rand(1,10,2)
# x4=torch.rand(1,10,2)


# outputs = model(torch.squeeze(x3).unsqueeze(0),torch.squeeze(x4).unsqueeze(0),torch.squeeze(x1).unsqueeze(0),x2.unsqueeze(0))
#     err+=torch.nn.MSELoss()(outputs, u)
# print(err)


#


# plt.scatter(interior_points[:,0], interior_points[:,1], color='black')
# plt.scatter(boundary_points[:,0], boundary_points[:,1], color='red')
# plt.show()
