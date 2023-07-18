import os
import torch
# heelo
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
       h=0.2
       gauss_points=5

       
       num_control_polygons=5
       batch_size=32
       num_epochs=40
       pts_per_polygon=10
       ev_per_polygon=10

       k=9.12
       
       dim=2
       # num_ev=4



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
       


