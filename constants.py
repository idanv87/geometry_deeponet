import os
import torch
class Constants:
       num_samples=5
       num_edges=6
       var_center=0
       var_angle=0.3
       radius=3
       h=0.1
       path='/Users/idanversano/Documents/clones/deeponet_data/'
       
       batch_size=16
       num_epochs=10
       pts_per_polygon=10
       ev_per_polygon=5
       num_control_polygons=1
       dim=2
       isExist = os.path.exists(path+'train')
       if not isExist:
              os.makedirs(path+'train')
       isExist = os.path.exists(path+'main_polygons')
       if not isExist:
              os.makedirs(path+'main_polygons')      

       isExist = os.path.exists(path+'polygons')
       if not isExist:
              os.makedirs(path+'polygons')           
       isExist = os.path.exists(path+'best_model')
       if not isExist:
              os.makedirs(path+'best_model')   

       isExist = os.path.exists(path+'figures')
       if not isExist:
              os.makedirs(path+'figures')  

       isExist = os.path.exists(path+'data_sets')

       if not isExist:
              os.makedirs(path+'data_sets')                  
       polygon_train_pathes=[]    
       main_polygons_pathes=[]   
       dtype=torch.float32

# print(os.getcwd()+'/deeponet_data/')
         