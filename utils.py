from tqdm import tqdm
import datetime
import pickle
import math, random, cmath
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import List, Tuple
import sklearn
import argparse
import torch
import dmsh
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from constants import Constants



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
    cumsum /= 2 * math.pi
    for i in range(steps):
        angles[i] /= cumsum
    return angles


def generate_polygon(
    center: Tuple[float, float],
    avg_radius: float,
    irregularity: float,
    spikiness: float,
    num_vertices: int,
) -> List[Tuple[float, float]]:
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
        point = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
        points.append(point)
        angle += angle_steps[i]

    return points


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


def create_mu():
    x = np.linspace(-1, 1, Constants.gauss_points)
    y = np.linspace(-1, 1, Constants.gauss_points)
    x, y = np.meshgrid(x, y)
    # A=np.zeros((x.shape[0], x.shape[1]),dtype=tuple)
    mu = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mu.append((x[i, j], y[i, j]))
    return mu


def extract_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_batches(n, batch_size):
    k = math.floor(n / batch_size)
    x = list(range(n))
    random.shuffle(x)
    return [list(x[i * batch_size : (i + 1) * batch_size]) for i in range(k)]


import random
from sklearn.metrics import pairwise_distances


def spread_points(n, data):
    assert data.shape[1] == 2
    r = data[:, 0] ** 2 + data[:, 1] ** 2
    starting_point_ind = np.argmin(r)
    points_from_data = np.array([data[starting_point_ind]])

    for i in range(1, n):
        pairwise_distances_to_data = pairwise_distances(
            data, Y=points_from_data, metric="euclidean", n_jobs=-1
        )
        pairwise_distances_to_data = np.array(pairwise_distances_to_data)

        min_distances_to_data = np.amin(pairwise_distances_to_data, axis=1)

        k = min_distances_to_data.argmax()

        points_from_data = np.append(points_from_data, [data[k]], axis=0)

    return points_from_data


def plot_polygon(path):
    df = extract_pickle(path)
    v = df["generator"]
    coord = [v[i] for i in range(v.shape[0])]
    coord.append(coord[0])  # repeat the first point to create a 'closed loop'
    xs, ys = zip(*coord)  # create lists of x and y values
    plt.figure()
    plt.plot(xs, ys)
    plt.show()



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
        self.path=log_path


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


def save_plots(train_loss, valid_loss, test_loss, metric_type:str, dir_path):

    # accuracy plots
    fig, ax=plt.subplots(1,2)
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
   
    plt.savefig(dir_path + "figures/"+ metric_type+".png")
    plt.show(block=False)


  


# def plot_polygon(coord):
def gaussian(x, y, mu):
    return math.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / np.sqrt(Constants.h))


class Gaussian:
    def __init__(self, mu):
        self.mu = mu

    def call(self, x, y):
        return gaussian(x, y, self.mu)



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


def Gauss_zeidel(A, b, x):
    ITERATION_LIMIT = 2
    # x = b*0
    for it_count in range(1, ITERATION_LIMIT):
        x_new = np.zeros_like(x, dtype=np.float_)
        # print(f"Iteration {it_count}: {x}")
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])

            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        # if np.allclose(x, x_new, rtol=1e-10):
        #     break
        x = x_new

    # print(f"Solution: {x}")
    # error = np.linalg.norm(abs(np.dot(A, x) - b))
    # print(error)
    #  it_count, np.max(abs(np.dot(A, x) - b))
    return x
    # return x, it_count, np.max(abs(np.dot(A, x) - b))


# pol_path=Constants.path+'polygons/rect.pt'
# p=torch.load(pol_path)
# err=np.sin(p['interior_points'][:,0])
# A=p['M'][p['interior_indices']][:,p['interior_indices']]
# A=A.todense()
# Gauss_zeidel(A,err,err*0)
# p=torch.load(Constants.path+'polygons/rect.pt')
# M=p['M']
# interior_indices=p['interior_indices']
# f=np.array(list(map(Test_function().call, p['X'][:,0],p['X'][:,1])))
# solve_helmholtz(M, interior_indices, f)
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


def plot3d(x, y, z, color="black"):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, color=color)





    def is_inside(self, x, y):
        #   point = Point(0.5, 0.5)
        point = Point(x, y)
        return self.polygon.contains(point)

    def call(self, x, y):
        if self.is_inside(x, y):
            return 1
        else:
            return 0



          
class chi_function:
    def __init__(self, vertices) -> None:
        self.polygon = Polygon(
            [(vertices[i, 0], vertices[i, 1]) for i in range(vertices.shape[0])]
        ) 

def complex_version(v):
        assert v.size==2
        r = np.sqrt(v[0] ** 2 + v[1] ** 2)
        theta = np.arctan2(v[1], v[0])
        return r*cmath.exp(1j*theta)

def stochastic_matrix(m,n):
    a=np.random.rand(m,n)
    return [a[i]/np.sum(a[i]) for i in range(m)]
       
    

# def print_layers(model):
#     for name, layer in model.named_modules():
#         if isinstance(layer, torch.nn.Linear):
#             pass
            # print(name)
            # print(layer._parameters)
            # print(layer.weight.grad)


