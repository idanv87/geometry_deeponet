import datetime
import time

from pylab import figure
import numpy as np
import matplotlib.pyplot as plt
import scipy
import dmsh
import meshio
import optimesh
from pathlib import Path

import cmath
from utils import *
from constants import Constants
# from coords import Map_circle_to_polygon
from pydec.dec import simplicial_complex
from functions.functions import Test_function, christofel, sin_function
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# sin_function(ind[0], ind[1], df['a'], df['b']).call
# u=(1/(ev_s[j]-Constants.k))*np.array(list(map(funcs[j], df['interior_points'][:, 0], df['interior_points'][:, 1])))

# x = value.real
# y = value.imag

# polygon = Polygon([(vertices[i][0], vertices[i][1]) for i in
#                   range(len(vertices))])


class circle:
    def __init__(self, R=0.95):
        vertices = []
        vertices_complex = []
        for i, r in enumerate(list(np.linspace(0, R, 40))):
            for j, theta in enumerate(list(np.linspace(0, 2*math.pi, 24))):
                # if i%4==0 and j%2 ==0:
                if True:
                    vertices_complex.append(r*cmath.exp(1j*theta))
                    vertices.append([r*math.cos(theta), r*math.sin(theta)])
        self.hot_points = np.array(vertices)
        self.hot_points_complex = vertices_complex

    def plot(self):
        plt.scatter(self.hot_points[:, 0], self.hot_points[:, 1], color='red')
        plt.title('unit circle')
        plt.show()


def calc_coeff(vertices):
    n = len(vertices)
    a = []
    Z = []
    for v in vertices:
        Z.append(complex_version(v))
    for i in range(len(Z)):
        exp1 = Z[(i-1) % len(Z)]-Z[i]
        exp2 = Z[(i) % len(Z)]-Z[(i+1) % len(Z)]
        a.append((exp1.conjugate()/exp1-exp2.conjugate()/exp2)*(1j/2))
    return a, Z


def calc_moment(k, a, z):
    # print([abs(z[i]**k)  for i in range(len(z))])
    return np.sum([a[i]*(z[i]**k) for i in range(len(a))])


class rectangle:

    def __init__(self, a, b):

        self.a = a
        self.b = b
        # self.a= a*np.sqrt(math.pi/a/b)
        # self.b= b*np.sqrt(math.pi/a/b)
        self.generators = np.array([[0, 0], [a, 0], [a, b], [0, b]])
        # p=Polygon(np.array([[0, 0], [a, 0], [a, b], [0,b]]))
        # p.create_mesh(Constants.h)
        self.M = []
        self.moments = [calc_moment(k, calc_coeff(self.generators)[
                                    0], calc_coeff(self.generators)[1]) for k in range(30)]
        self.area = self.a*self.b
        indices = [(1, 1), (1, 2), (2, 1), (2, 2)]
        self.ev = np.flip(
            (math.pi**2)*np.array([(ind[0]/self.a)**2+(ind[1]/self.b)**2 for ind in indices]))
        self.principal_ev = self.ev[-1]
        self.interior_points = None
        self.data = None

    def plot(self):
        plt.scatter(self.interior_points[:, 0],
                    self.interior_points[:, 1], color='black')
        plt.scatter(self.hot_points[:, 0], self.hot_points[:, 1], color='red')
        plt.show()

    def create_mesh(self, h):
        n = int(1/h)
        x, y = np.meshgrid(np.linspace(0, self.a, n)[
                           1:-1], np.linspace(0, self.b, n)[1:-1])
        vertices = []
        hot_points = []
        for i in range(n-2):
            for j in range(n-2):
                vertices.append([x[i, j], y[i, j]])
                if i % Constants.hot_spots_ratio == 0 and j % Constants.hot_spots_ratio == 0:
                    hot_points.append([x[i, j], y[i, j]])

        self.interior_points = np.array(vertices)
        self.hot_points = np.array(hot_points)
        self.data = {
            "ev": self.ev,
            "principal_ev": self.principal_ev,
            "interior_points": self.interior_points[np.lexsort(np.fliplr(self.interior_points).T)],
            "hot_points": self.hot_points[np.lexsort(np.fliplr(self.hot_points).T)],
            "generators": self.generators,
            "legit": True,
            'a': self.a,
            'b': self.b,
            'M': self.M,
            'moments': self.moments,
            'type': 'rectangle'
        }

    def save(self, path):
        torch.save(self.data, path)

    @classmethod
    def solve_helmholtz_equation(cls, f, *args):
        domain, mode = args
        b = f(domain['interior_points'][:, 0], domain['interior_points'][:, 1])

        assert abs(mode-Constants.k) > 1e-6
        return 1/(mode-Constants.k)*b


class Polygon:
    def __init__(self, generators):
        self.generators = generators
        # self.generators= (np.sqrt(math.pi) / np.sqrt(polygon_centre_area(generators))) * generators
        # self.diameter=np.max(np.linalg.norm(self.generators))
        # v[:, 0] -= np.mean(v[:, 0])
        # v[:, 1] -= np.mean(v[:, 1])
        self.geo = dmsh.Polygon(self.generators)
        self.moments = [calc_moment(k, calc_coeff(self.generators)[
                                    0], calc_coeff(self.generators)[1]) for k in range(30)]

    def create_mesh(self, h):

        if np.min(calc_min_angle(self.geo)) > (math.pi / 8):
            X, cells = dmsh.generate(self.geo, h)

            X, cells = optimesh.optimize_points_cells(
                X, cells, "CVT (full)", 1.0e-6, 120
            )
        # dmsh.show(X, cells, self.geo)

        self.vertices = X
        self.sc = simplicial_complex(X, cells)
        self.M = (
            (self.sc[0].star_inv)
            @ (-(self.sc[0].d).T)
            @ (self.sc[1].star)
            @ self.sc[0].d
        )

        self.boundary_indices = [i for i in range(self.generators.shape[0])]
        self.calc_boundary_indices()
        self.interior_indices = list(
            set(range(self.vertices.shape[0])) - set(self.boundary_indices)
        )
        self.interior_points = self.vertices[self.interior_indices]

        self.hot_points = spread_points(int(
            (int(1/Constants.h)-2)**2/(Constants.hot_spots_ratio**2)), self.interior_points)
        self.ev = self.laplacian().real

    def calc_boundary_indices(self):
        for i in range(self.generators.shape[0], (self.vertices).shape[0]):
            if on_boundary(self.vertices[i], self.geo):
                self.boundary_indices.append(i)

    def laplacian(self):
        return scipy.sparse.linalg.eigs(
            -self.M[self.interior_indices][:, self.interior_indices],
            k=5,
            return_eigenvectors=False,
            which="SM",
        )

    def is_legit(self):
        if np.min(abs(self.sc[1].star.diagonal())) > 0:
            return True
        else:
            return False

    def save(self, path):
        assert self.is_legit()
        data = {
            "ev": self.ev,
            "principal_ev": self.ev[-1],
            "interior_points": self.interior_points[np.lexsort(np.fliplr(self.interior_points).T)],
            "hot_points": self.hot_points[np.lexsort(np.fliplr(self.hot_points).T)],
            "generators": self.generators,
            "M": self.M[self.interior_indices][:, self.interior_indices],
            'moments': self.moments,
            "legit": True,
            'type': 'polygon'
        }
        torch.save(data, path)

    def plot(self):
        plt.scatter(self.interior_points[:, 0],
                    self.interior_points[:, 1], color='black')
        plt.scatter(self.hot_points[:, 0], self.hot_points[:, 1], color='red')
        plt.show()

    @classmethod
    def solve_helmholtz_equation(self, func, domain):

        solution = Test_function(domain['generators'], True)
        return np.array(list(map(solution, domain['interior_points'][:, 0], domain['interior_points'][:, 1])))
        return
        M = domain['M']
        b = func(domain['interior_points'][:, 0],
                 domain['interior_points'][:, 1])
        A = -M - Constants.k * scipy.sparse.identity(
            len(domain['interior_points'])
        )
        solution = scipy.sparse.linalg.spsolve(A, b)

        return solution


def analyze_momnets(path, col):
    domain = torch.load(path)
    n = domain['generators'].shape[0]
    x1 = np.array([[domain['moments'][l].real, domain['moments'][l].imag]
                  for l in range(8)])
    m = 6
    x = x1[m, 0]
    y = x1[m, 1]

    plt.scatter(x, y, color=col)

if __name__ == '__main__':
    pass

    polygons_files_names = extract_path_from_dir(Constants.path + "polygons/")
    test_domains_path = [Constants.path + "hints_polygons/trapz.pt"]
    train_domains_path = polygons_files_names
    # for name in train_domains_path:
    #     analyze_momnets(name,'r')
    # for name in test_domains_path:
    #     analyze_momnets(name,'b')

    # plt.show()


def eval_on_domain(path):
    from functions.functions import sin_function
    domain = torch.load(path)
    f = sin_function(4, 4, domain['a'], domain['b']).call
    x = list(domain['hot_points'][:, 0])
    y = list(domain['hot_points'][:, 1])
    solution = list(map(f, x, y))
    plt.scatter(x, y, c=solution)
    plt.show()


def print_mommets(domain, col):
    # x1 = np.array([[domain.moments[l].real, domain.moments[l].imag]
    #               for l in range(8)])
    x1 = np.array([[domain['moments'][l].real, domain['moments'][l].imag]
                  for l in range(8)])
    m=3
    x = x1[2:, 0]
    y = x1[2:, 1]
    # plt.plot(x,col)
    # plt.scatter(x,y,color=col)
    plt.plot(range(x.shape[0]), x/8, color=col)

# domain=torch.load(polygons_files_names[2])
# plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1])
# plt.show()

# test_domains_path=[polygons_files_names[0]]
# train_domains_path=polygons_files_names[:3]

# for name in train_domains_path:
#     domain=torch.load(name)
#     print_mommets(domain,'r')    
# for name in test_domains_path:
#     domain=torch.load(name)
#     print_mommets(domain,'b')    
# plt.show()    
#

 