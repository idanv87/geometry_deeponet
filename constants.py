import os
import torch
import numpy as np
import math
import cmath


class Constants:
       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       dtype=torch.float32
       
       
       path='/Users/idanversano/Documents/clones/deeponet_data/'
       output_path='/Users/idanversano/Documents/clones/deeponet_output/'

       num_samples=5
       num_edges=7
       var_center=0
       var_angle=0.4
       radius=3
       h=1/40
       gauss_points=5

  
       num_control_polygons=1
       batch_size=4
       num_epochs=300
       hot_spots_ratio=2
       num_moments=5

       pts_per_polygon=10
       points_on_circle=[]
       for r in list(np.linspace(0,0.99,50)):
              for theta in list(np.linspace(0,2*math.pi,50)):
                     z=r*cmath.exp(theta*1j)
                     points_on_circle.append([z.real,z.imag ])

       points_on_circle=np.array(points_on_circle)
       ev_per_polygon=2

       k=3.11
       
       dim=2
       # num_ev=4



       isExist = os.path.exists(path+'polygons')
       if not isExist:
              os.makedirs(path+'polygons')    
       
       isExist = os.path.exists(path+'hints_polygons')
       if not isExist:
              os.makedirs(path+'hints_polygons')   

       
      
              
       l=[]
       for i in range(1,5):
              for j in range(1,5):
                     l.append((i,j))



# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(Constants.path+'runs')
# x = range(100)
# for i in x:
#     writer.add_scalar('y=5x', i * 10, i)
# writer.flush()    
# writer.close()

# class father:
#        def __init__(self,a):  
#               self.a=a

# class child(father):
#        def __init__(self,b):
#               super().__init__('a')
#               self.b=b
      
# x=child('b')
# vertices=[]
# n=4
# x,y=np.meshgrid(np.linspace(0,1,n)[1:-1], np.linspace(0,2,n)[1:-1])
# for i in range(n-2):
#             for j  in range(n-2):
#                 vertices.append([x[i,j],y[i,j]])
# print(vertices)