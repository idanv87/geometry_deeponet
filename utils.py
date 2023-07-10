import numpy as np
import math, random
import scipy
import os
from typing import List, Tuple
from shapely import geometry
import datetime

from constants import Constants
from pydec.dec import simplicial_complex 
import pickle
import sklearn
import argparse
import torch
from tqdm import tqdm
# from matplotlib.patches import Polygon

import dmsh


parser = argparse.ArgumentParser()
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
args = vars(parser.parse_args())

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles
def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

def count_trainable_params(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import numpy as np
from scipy.integrate import solve_ivp
def exponential_decay(t, y): return -0.5 * y
# sol = solve_ivp(exponential_decay, [0, 10], [0])
# plt.plot(sol.t, sol.y)
# plt.show()

def polygon_centre_area(vertices):
    x=vertices[:,0]
    y=vertices[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def polygon_centroid(vertices):
    A=polygon_centre_area(vertices)
    x=vertices[:,0]
    y=vertices[:,1]
    Cx=np.sum((x[:-1]+x[1:])*(x[:-1]*y[1:]-x[1:]*y[:-1]))/6/A
    Cy=np.sum((y[:-1]+y[1:])*(x[:-1]*y[1:]-x[1:]*y[:-1]))/6/A
    return Cx,Cy
        
                
def map_right(p1,p2,p3):
      B=np.array([[p1[0]],[p1[1]]])
      A=np.array([[p2[0]-B[0], p3[0]-B[0]],[p2[1]-B[1],p3[1]-B[1] ]])
      
      return np.squeeze(A),B

# plt.scatter(X[:,0],X[:,1])
# plt.scatter(X[:10,0],X[:10,1],color='blue')
# plt.scatter(v[:,0],v[:,1],color= 'red')
# # plt.scatter(X[:,0],X[:,1],'r')
# print(geo.__dict__['paths'][0].__dict__)

def is_between(p1, p2, point):
    crossproduct = (point[1] - p1[1]) * (p2[0] - p1[0]) - (point[0] - p1[0]) * (p2[1] - p1[1])

    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > 1e-10:
        return False

    dotproduct = (point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1])*(p2[1] - p1[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1])
    if dotproduct > squaredlengthba:
        return False

    return True

def on_boundary(point, geo):
      for i in range(len(geo.__dict__['paths'])):
            p1=geo.__dict__['paths'][i].__dict__['x0']
            p2=geo.__dict__['paths'][i].__dict__['x1']
            if is_between(p1,p2,point):
                  return True
      return False    



def create_mu():
    x=np.linspace(-1,1,5)
    y=np.linspace(-1,1,5)
    x,y=np.meshgrid(x,y)
    # A=np.zeros((x.shape[0], x.shape[1]),dtype=tuple)
    mu=[]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mu.append((x[i,j],y[i,j]))
    return mu

def extract_pickle(file_path):
     with open(file_path, 'rb') as f:
         data = pickle.load(f)
     return  data  

def create_batches(n, batch_size):
     k=math.floor(n/batch_size)
     x=list(range(n))
     random.shuffle(x)
     return [list(x[i*batch_size:(i+1)*batch_size]) for i in range(k)]

import random
from sklearn.metrics import pairwise_distances

def spread_points(n, data):
    assert data.shape[1]==2
    r=data[:,0]**2+data[:,1]**2
    starting_point_ind=np.argmin(r)
    points_from_data=np.array([data[starting_point_ind]])



    for i in range(1, n):
        pairwise_distances_to_data = pairwise_distances (data, Y=points_from_data, metric='euclidean', n_jobs=-1)
        pairwise_distances_to_data = np.array(pairwise_distances_to_data)

        min_distances_to_data = np.amin(pairwise_distances_to_data, axis=1)

        k = min_distances_to_data.argmax()

        points_from_data = np.append(points_from_data, [data[k]], axis=0)

    return points_from_data

def plot_polygon(path):
   df=extract_pickle(path)
   v=df['generator']
   coord=[v[i] for i in range(v.shape[0])]
   coord.append(coord[0]) #repeat the first point to create a 'closed loop'
   xs, ys = zip(*coord) #create lists of x and y values
   plt.figure()
   plt.plot(xs,ys) 
   plt.show()


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True        


# training function
def fit(model, train_dataloader, train_dataset, optimizer, criterion, device):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy

def np_to_torch(x):
    return torch.tensor(x, dtype=Constants.dtype)

def save_file(f, dir, name):

    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)   
    torch.save(f, dir+name+'.pt')
    return dir+name+'.pt'

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.uniq_filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')+'.pth'

        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")

            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, Constants.path+'best_model/'+self.uniq_filename)


def save_plots(train_loss, valid_loss, test_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots

    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss[1:], color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss[1:], color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Constants.path+'figures/train_val_loss.png')

    plt.figure(figsize=(10, 7))
    plt.plot(
        test_loss, color='blue', linestyle='-', 
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Constants.path+'figures/test_loss.png')

    plt.show()





# def plot_polygon(coord):
def gaussian(x,y, mu):
     return math.exp(-((x-mu[0])**2+(y-mu[1])**2)/np.sqrt(Constants.h))  
class Gaussian:
    def __init__(self, mu):
        self.mu=mu
    def call(self, x,y):
        return gaussian(x,y,self.mu)    

class Test_function:
    def __init__(self):
        pass

    def call(self, x,y):
        # (x-np.sqrt(math.pi))*(x+np.sqrt(math.pi))
        return -2*y**2+math.pi-2*x**2-x**2*y**2+0.25*math.pi*(x**2+y**2)-math.pi**2/16
      
def calc_min_angle(geo):
    seg1=[]
    for i in range(len(geo.__dict__['paths'])):
                p1=geo.__dict__['paths'][i].__dict__['x0']
                p2=geo.__dict__['paths'][i].__dict__['x1']
                seg1.append(p1)
                
    
    angle=[]
    for i in range(len(seg1)):
        p1=seg1[i % len(seg1)]
        p2=seg1[(i-1) % len(seg1)]
        p3=seg1[(i+1) % len(seg1)]  
        angle.append(np.dot(p2-p1,p3-p1)/(np.linalg.norm(p2-p1)*np.linalg.norm(p3-p1)))
    return np.arccos(angle) 

def Gauss_zeidel(A, b):
    ITERATION_LIMIT = 1000

# # initialize the matrix
#     A = np.array(
#         [
#             [10.0, -1.0, 2.0, 0.0],
#             [-1.0, 11.0, -1.0, 3.0],
#             [2.0, -1.0, 10.0, -1.0],
#             [0.0, 3.0, -1.0, 8.0],
#         ]
#     )
#     # initialize the RHS vector
#     b = np.array([6.0, 25.0, -11.0, 15.0])

    print("System of equations:")
    for i in range(A.shape[0]):
        row = [f"{A[i,j]:3g}*x{j+1}" for j in range(A.shape[1])]
        print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

    x = np.zeros_like(b, np.float_)
    for it_count in range(1, ITERATION_LIMIT):
        x_new = np.zeros_like(x, dtype=np.float_)
        # print(f"Iteration {it_count}: {x}")
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.allclose(x, x_new, rtol=1e-8):
            break
        x = x_new

    # print(f"Solution: {x}")
    # error = np.dot(A, x) - b
    return x, it_count, np.max(abs(np.dot(A, x) - b))
    # print(f"Error: {error}")  
# def generate_shape():
#     V=[[0,0],[1,0],[1,1],[0,1]]
#     for v in V:








# def plot_all_polygons(path):
# data=np.random.rand(100,2)
# data2=spread_points(20, data)
# plt.scatter(data[:,0], data[:,1], color='black')
# plt.scatter(data2[:,0], data2[:,1], color='red')
# plt.show()
  

          
# for filename in os.listdir(Constants.path+'train'):
#         f = os.path.join(Constants.path+'train', filename)
#         if f.endswith('.pkl'):
#           print('h')
#           print(f)
# boundary_indices=[i for i in range(v.shape[0])]     

# for i in range(X.shape[0]):
#       if on_boundary(X[i],geo):
#             boundary_indices.append(i)
# print(boundary_indices)            
# for index in boundary_indices:
#       plt.scatter(X[index][0],X[index][1],color='red' )           
# print(boundary_indices)      


# p1=geo.__dict__['paths'][0].__dict__['x0']
# p2=geo.__dict__['paths'][0].__dict__['x1']
# # print(is_between(p1,p2,point))
# plt.scatter(p1[0],p1[1],color= 'red')
# plt.scatter(p2[0],p2[1],color= 'red')
# point=X[7]
# plt.scatter(point[0],point[1],color= 'red')
# plt.scatter(v[1][0],v[1][1],color= 'black')
# plt.scatter(v[2][0],v[2][1],color= 'black')
# # print(on_boundary(point, geo))
# print(is_between(v[0],v[1],point))
# print(is_between(v[1],v[2],point))
# print(is_between(v[2],v[0],point))
# dmsh.show(X, cells, geo)
# plt.show()
# point=np.array([0.5,0.00000001])
# p1=np.array([0,0])
# p2=np.array([1,0])
# print(is_between(p1,p2,point))

