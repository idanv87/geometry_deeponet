import numpy as np
import math, random
import scipy
from typing import List, Tuple
from shapely import geometry

from constants import Constants
from pydec.dec import simplicial_complex 


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

def gaussian(x,y):
     return math.exp(-(x**2+y**2)/Constants.h)  
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

