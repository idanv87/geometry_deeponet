import pandas as pd
from sklearn.metrics import pairwise_distances
import random
from tqdm import tqdm
import datetime
import pickle
import math
import random
from scipy.stats import gaussian_kde
import cmath
import os
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import List, Tuple

import torch


from constants import Constants



def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)
    
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))



def count_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def polygon_centre_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_centroid(vertices):
    A = polygon_centre_area(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]
    Cx = np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / 6 / A
    Cy = np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1])) / 6 / A
    return Cx, Cy


def map_right(p1, p2, p3):
    B = np.array([[p1[0]], [p1[1]]])
    A = np.array([[p2[0] - B[0], p3[0] - B[0]], [p2[1] - B[1], p3[1] - B[1]]])

    return np.squeeze(A), B


def is_between(p1, p2, point):
    crossproduct = (point[1] - p1[1]) * (p2[0] - p1[0]) - (point[0] - p1[0]) * (
        p2[1] - p1[1]
    )

    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > 1e-10:
        return False

    dotproduct = (point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1]) * (
        p2[1] - p1[1]
    )
    if dotproduct < 0:
        return False

    squaredlengthba = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (
        p2[1] - p1[1]
    )
    if dotproduct > squaredlengthba:
        return False

    return True


def on_boundary(point, geo):
    for i in range(len(geo.__dict__["paths"])):
        p1 = geo.__dict__["paths"][i].__dict__["x0"]
        p2 = geo.__dict__["paths"][i].__dict__["x1"]
        if is_between(p1, p2, point):
            return True
    return False





def spread_points(subset_num,X):
    
    x=X[:,0]
    y=X[:,1]
    total_num = x.shape[0]
    xy = np.vstack([x, y])
    dens = gaussian_kde(xy)(xy)

    # Try playing around with this weight. Compare 1/dens,  1-dens, and (1-dens)**2
    weight = 1 / dens
    weight /= weight.sum()

    # Draw a sample using np.random.choice with the specified probabilities.
    # We'll need to view things as an object array because np.random.choice
    # expects a 1D array.
    dat = xy.T.ravel().view([('x', float), ('y', float)])
    subset = np.random.choice(dat, subset_num, p=weight)
    return np.vstack((subset['x'], subset['y'])).T
    



def np_to_torch(x):
    return torch.tensor(x, dtype=Constants.dtype)


def save_file(f, dir, name):

    isExist = os.path.exists(dir)
    if not isExist:
        os.makedirs(dir)
    torch.save(f, dir + name + ".pt")
    return dir + name + ".pt"


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, log_path, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.path = log_path

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                self.path+'best_model.pth',
            )


def save_plots(train_loss, valid_loss, test_loss, metric_type: str, dir_path):

    # accuracy plots
    fig, ax = plt.subplots(1, 2)
    # plt.figure(figsize=(10, 7))
    ax[0].plot(train_loss[1:], color="orange", linestyle="-", label="train")
    ax[0].plot(valid_loss[1:], color="red", linestyle="-", label="validataion")
    ax[0].set(xlabel='Epochs', ylabel=metric_type)
    ax[0].legend(loc="upper right")

    ax[1].plot(test_loss, color="blue", linestyle="-", label="test")
    ax[1].set(xlabel='Epochs', ylabel=metric_type)
    ax[1].legend(loc="upper right")

    fig.suptitle("metric type: "+metric_type)
    isExist = os.path.exists(dir_path+'figures')
    if not isExist:
        os.makedirs(dir_path+'figures')

    plt.savefig(dir_path + "figures/" + metric_type+".png")
    plt.show(block=False)







def calc_min_angle(geo):
    seg1 = []
    for i in range(len(geo.__dict__["paths"])):
        p1 = geo.__dict__["paths"][i].__dict__["x0"]
        p2 = geo.__dict__["paths"][i].__dict__["x1"]
        seg1.append(p1)

    angle = []
    for i in range(len(seg1)):
        p1 = seg1[i % len(seg1)]
        p2 = seg1[(i - 1) % len(seg1)]
        p3 = seg1[(i + 1) % len(seg1)]
        angle.append(
            np.dot(p2 - p1, p3 - p1)
            / (np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p1))
        )
 
    return np.arccos(angle)




def solve_helmholtz(M, interior_indices, f):
    A = -M[interior_indices][:, interior_indices] - Constants.k * scipy.sparse.identity(
        len(interior_indices)
    )
    #    x,y,e=Gauss_zeidel(A,f[interior_indices])
    #    print(e)
    return scipy.sparse.linalg.spsolve(A, f[interior_indices])


# solve_helmholtz(M, interior_indices, f)


def extract_path_from_dir(dir):
    raw_names = next(os.walk(dir), (None, None, []))[2]
    return [dir + n for n in raw_names if n.endswith(".pt")]








def complex_version(v):
    assert v.size == 2
    r = np.sqrt(v[0] ** 2 + v[1] ** 2)
    theta = np.arctan2(v[1], v[0])
    return r*cmath.exp(1j*theta)




def save_figure(X, Y, titles, names, colors):

    # accuracy plots
    fig, ax = plt.subplots(1, len(X))
    for j in range(len(X)):
        ax[j].scatter(X[j],Y[j])

    plt.savefig(Constants.fig_path + "figures/" + ".eps",format='eps',bbox_inches='tight')
    plt.show(block=False)

 



def step_fourier(L,Theta):
    N=50
    x=[0]+[np.sum(L[:k+1]) for k in range(len(L))]
    a0=np.sum([l*theta for l,theta in zip(L,Theta)])
    a1=[2*np.sum([L[i]*Theta[i]*(-np.sin(2*math.pi*n*x[i+1])+np.sin(2*math.pi*n*x[i]))/(2*math.pi*n) 
                  for i in range(len(L))]) for n in range(1,N)]
    a2=[2*np.sum([L[i]*Theta[i]*(np.cos(2*math.pi*n*x[i+1])-np.cos(2*math.pi*n*x[i]))/(2*math.pi*n)
                   for i in range(len(L))]) for n in range(1,N)]
    coeff=[a0]
    for i in range(N-1):
        coeff.append(a1[i])
        coeff.append(a2[i])

    return np.array(coeff)

def save_uniqe(file, path):
    uniq_filename = (
            str(datetime.datetime.now().date())
            + "_"
            + str(datetime.datetime.now().time()).replace(":", ".")
        )
    torch.save(file, path+uniq_filename+'.pt') 