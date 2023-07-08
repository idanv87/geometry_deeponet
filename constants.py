import os
import torch
class Constants:
       dtype=torch.float32
       path='/Users/idanversano/Documents/clones/deeponet_data/'
       output_path='/Users/idanversano/Documents/clones/deeponet_output/'

       num_samples=5
       num_edges=9
       var_center=0
       var_angle=0.4
       radius=3
       h=0.2

       
       batch_size=16
       num_epochs=30
       pts_per_polygon=10
       ev_per_polygon=3
       num_control_polygons=2
       dim=2
       num_ev=4
       isExist = os.path.exists(path+'train')
       if not isExist:
              os.makedirs(path+'train')


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
       

# print(os.getcwd()+'/deeponet_data/')
         