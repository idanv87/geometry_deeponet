import os
class Constants:
       num_samples=5
       num_edges=6
       var_center=0
       var_angle=0.3
       radius=3
       h=0.2
       path='/Users/idanversano/Documents/clones/deeponet_data/'
       batch_size=8
       num_epochs=10
       pts_per_polygon=5
       ev_per_polygon=5
       dim=2
       isExist = os.path.exists(path+'train')
       if not isExist:
              os.makedirs(path+'train')
       isExist = os.path.exists(path+'main_polygons')
       if not isExist:
              os.makedirs(path+'main_polygons')       

      #
         