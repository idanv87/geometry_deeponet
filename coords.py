
# Mean value coordinates Michael S. Floater

# dw/dz = 1/(z^m - 1)^(2/m) circle to m gon


import math
import cmath

import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from utils import *



# Python3 code to find all three angles
# of a triangle given coordinate
# of all three vertices
import math



# returns square of distance b/w two points
# def tanacos(x):
#        return math.sqrt(1-x**2)/x



def lengthSquare(X, Y):
	xDiff = X[0] - Y[0]
	yDiff = X[1] - Y[1]
	return xDiff * xDiff + yDiff * yDiff
	
def get_angle(A, B, C):
	
	# Square of lengths be a2, b2, c2
	a2 = lengthSquare(B, C)
	b2 = lengthSquare(A, C)
	c2 = lengthSquare(A, B)

	# length of sides be a, b, c
	a = math.sqrt(a2)
	b = math.sqrt(b2)
	c = math.sqrt(c2)

	# From Cosine law
    
	alpha = math.acos((b2 + c2 - a2) /
						(2 * b * c))
	betta = math.acos((a2 + c2 - b2) /
						(2 * a * c))
	gamma = math.acos((a2 + b2 - c2) /
						(2 * a * b))
    
	


	return 0.5*alpha, betta, gamma




# This code is contributed
# by ApurvaRaj


# vertices=[[0,0.],[1,0.],[1.,1],[0.,1]]
# vertices=[[0,0],[1,0],[1,1/4],[1/4,1/4],[1/4,1],[0,1]]
# point=(0.5,0.1)
def find_meanvalue_coords(vertices, point):
        
        gamma=[]
        betta=[]
        polygon=Polygon([(vertices[i][0],vertices[i][1]) for i in range(len(vertices))])
        # plt.scatter(np.array(vertices)[:,0], np.array(vertices)[:,1])
        # plt.scatter(point[0], point[1])
        # plt.show()
     

        assert polygon.contains(Point(point[0],point[1]))
        for i in range(len(vertices)):
            gamma.append(get_angle(point,vertices[i],vertices[(i+1) % len(vertices)] )[0])
            betta.append(get_angle(point,vertices[i-1],vertices[i % len(vertices)] )[0])

        w=[(math.tan(betta[i])+math.tan(gamma[i]))/(math.sqrt(lengthSquare(point, vertices[i]))) for i in range(len(gamma))]    
        return [w[i]/np.sum(w) for i in range(len(w)) ]
	    
def map_circle_to_n_gon(n,r,theta):
    z=r*cmath.exp(1j*theta)
    value1=scipy.special.hyp2f1(1/n, 2/n, 1+1/n, z**n, out=None)
    value=z*((1-z**n)**(2/n))*((z**n-1)**(-2/n))*value1
    x=value.real
    y=value.imag

    # for r in list(np.linspace(0, 0.9, 200)):   
    #     for theta in list(np.linspace(0, 2*np.pi, 100)):
    #         z=r*cmath.exp(1j*theta)
    #         value1=scipy.special.hyp2f1(1/n, 2/n, 1+1/n, z**n, out=None)
    #         value=z*((1-z**n)**(2/n))*((z**n-1)**(-2/n))*value1
    #         x.append(value.real)
    #         y.append(value.imag)
    return x, y
def map_regular_n_gon_to_polygon(generators_of_target_polygon, xi,eta):
    generators=generators_of_target_polygon
    #   generators np array of shape (n,2) 
    n=len(generators)
    reg_vertices=[cmath.exp(k*2*math.pi*1J/n) for k in range(n)]
    # reg_vertices=[reg_vertices[k]-reg_vertices[0] for k in range(n)]
    # angles=[k*2*math.pi/n/math.pi*180 for k in range(n)]
    x=[reg_vertices[i].real*2 for i in range(n)]
    y=[reg_vertices[i].imag*2 for i in range(n)]
    w=find_meanvalue_coords([[x[i],y[i]] for i in range(n)],(xi, eta))
    return np.matmul(np.array(w),generators)

def map_circle_to_polygon(vertices_target,r,theta):
      x,y=map_circle_to_n_gon(len(vertices_target),r, theta )
      return map_regular_n_gon_to_polygon(vertices_target, x,y)
      
class  Map_circle_to_polygon:
        def __init__(self, vertices_target):
            self.vertices= vertices_target
        def call(self,x,y):
              r=np.sqrt(x**2+y**2)
              theta=np.arctan2(y, x)
              return map_circle_to_polygon(self.vertices, r, theta)
              


# x=[]
# y=[]
# n=5
# for r in list(np.linspace(0.1, 0.99, 200)):   
#         for theta in list(np.linspace(0, 2*np.pi, 100)):
#             z=r*cmath.exp(1j*theta)
#             value1=scipy.special.hyp2f1(1/n, 2/n, 1+1/n, z**n, out=None)
#             value=z*((1-z**n)**(2/n))*((z**n-1)**(-2/n))*value1
#             x.append(value.real)
#             y.append(value.imag)
# reg_vertices=[cmath.exp(k*2*math.pi*1J/n) for k in range(n)]
# plt.scatter(x,y)
# x=[reg_vertices[i].real for i in range(n)]
# y=[reg_vertices[i].imag for i in range(n)]
# plt.scatter(x,y,color='red')
# plt.show()



# w=map_circle_to_polygon(np.array([[0,0],[1,0],[1,1],[0,1]]),0.5,math.pi)

# print(w)
      

# x,y=map_circle_to_n_gon(5)
# print(y)
# plt.scatter(x,y)
# plt.show()
# w=find_mv_coords(vertices, point)




# v=np.array([w[i]*np.array(vertices)[i] for i in range(len(w))])
# print(np.sum(v, axis=0))
# print(point)
# print(w)


        
