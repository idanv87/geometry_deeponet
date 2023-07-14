import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline
import torch 
import scipy

from constants import Constants
from utils import solve_helmholtz, Test_function, Gauss_zeidel, plot3d, spread_points, np_to_torch
import matplotlib.pyplot as plt
from model import model
from data_generator import create_data_points, control_polygons



class interplation_block:
    def __init__(self, points, values):
        assert points.shape[-1]==2

        self.tck=scipy.interpolate.bisplrep(points[:,0], points[:,1], values)

    def call(self, X,Y):
        return scipy.interpolate.bisplev(X,Y,self.tck)
        


def hint_block(err, points, model):
    # p=torch.load(pol_path)
    
    
    data=create_data_points(control_polygons,[pol_path], 'hints',interplation_block(points, err).call)
    solution=[]
    for dat in data:
        x1,x2,x3,x4=dat
        solution.append(model(x3.unsqueeze(0),x4.unsqueeze(0),x1.unsqueeze(0),x2.unsqueeze(0)))
    solution=torch.stack(solution).detach().numpy()   

    return solution

def GS_block(A,b, x):
    return Gauss_zeidel(A,b, x)

def hint_iteration(A,b, n_it, points, model):

    k_it=0
    x=0*b
    for i in range(n_it): 
        x_0=x
        k_it+=1
        
        if k_it % 5 >0:
            # y=hint_block(np.dot(A,np.squeeze(x_0))-np.squeeze(b), points)
            # print(y.shape)
            
            # x=x_0+hint_block(np.dot(A,x_0)-b, points)
            x=GS_block(A,np.squeeze(b),np.squeeze(x_0))
          
        else:
            
            # x=np.squeeze(x_0)+np.squeeze(hint_block(np.dot(A,np.squeeze(x_0))-np.squeeze(b), points, model))
            pass
        print(np.linalg.norm(np.dot(A,x)-b))
        if   np.allclose(np.dot(A,x),b, 1e-10):
            return k_it,  np.dot(A,x)-b
    
    return k_it, x-x_0
pol_path=Constants.path+'polygons/rect.pt'
p=torch.load(pol_path)
err=np.sin(p['interior_points'][:,0])
A=p['M'][p['interior_indices']][:,p['interior_indices']]
A=A.todense()
# print(type(err))
it,err=hint_iteration(A,err,500, p['interior_points'], model)
# print(err)


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


plt.show()

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


