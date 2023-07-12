import os
import torch
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

       
       num_control_polygons=5
       batch_size=64
       num_epochs=40
       pts_per_polygon=25
       ev_per_polygon=10

       k=2.17
       
       dim=2
       num_ev=4
       isExist = os.path.exists(path+'train')
       if not isExist:
              os.makedirs(path+'train')


       isExist = os.path.exists(path+'polygons')
       if not isExist:
              os.makedirs(path+'polygons')     
    #    isExist = os.path.exists(path+'special_polygons')
    #    if not isExist:
    #           os.makedirs(path+'special_polygons')   

           

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
         