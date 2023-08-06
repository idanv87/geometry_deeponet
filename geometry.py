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
    def __init__(self,R=0.95):
        vertices=[]
        vertices_complex=[]
        for i,r in enumerate(list(np.linspace(0,R,40))):
            for j,theta in enumerate(list(np.linspace(0,2*math.pi,24))):
                # if i%4==0 and j%2 ==0:
                if True:
                    vertices_complex.append(r*cmath.exp(1j*theta))
                    vertices.append([r*math.cos(theta), r*math.sin(theta)])
        self.hot_points=np.array(vertices)
        self.hot_points_complex=vertices_complex
       

    def plot(self):
        plt.scatter(self.hot_points[:,0], self.hot_points[:,1],color='red')
        plt.title('unit circle')
        plt.show()    


def calc_coeff(vertices):
    n=len(vertices)   
    a=[]
    Z=[]
    for v in vertices:
        Z.append(complex_version(v))
    for i  in range(len(Z)):
        exp1=Z[(i-1) % len(Z)]-Z[i]
        exp2=Z[(i) % len(Z)]-Z[(i+1)% len(Z)]
        a.append((exp1.conjugate()/exp1-exp2.conjugate()/exp2)*(1j/2))
    return a,Z

def calc_moment(k,a,z):
    return np.sum([a[i]*z[i]**k for i in range(len(a))])

class rectangle:
    
    def __init__(self, a,b): 
        self.a=a
        self.b=b
        # self.a= a*np.sqrt(math.pi/a/b)
        # self.b= b*np.sqrt(math.pi/a/b)
        self.generators=np.array([[0,0],[self.a,0],[self.a,self.b],[0,self.b]])
        # p=Polygon(np.array([[0, 0], [a, 0], [a, b], [0,b]]))
        # p.create_mesh(Constants.h)
        self.M=[]
        self.moments=[calc_moment(k,calc_coeff(self.generators)[0],calc_coeff(self.generators)[1]) for k in range(20)]
        self.area=self.a*self.b
        indices=[(1,1), (1,2),(2,1),(2,2)]
        self.ev=np.flip((math.pi**2)*np.array([(ind[0]/self.a)**2+(ind[1]/self.b)**2 for ind in indices]))
        self.principal_ev=self.ev[-1]
        self.interior_points=None
        self.data=None
       

    def plot(self):
        plt.scatter(self.interior_points[:,0], self.interior_points[:,1],color='black')
        plt.scatter(self.hot_points[:,0], self.hot_points[:,1],color='red')
        plt.show()

    def create_mesh(self,h):
        n=int(1/h)
        x,y=np.meshgrid(np.linspace(0,self.a,n)[1:-1], np.linspace(0,self.b,n)[1:-1])
        vertices=[]
        hot_points=[]
        for i in range(n-2):
            for j  in range(n-2):
                vertices.append([x[i,j],y[i,j]])
                if i %Constants.hot_spots_ratio ==0 and j%Constants.hot_spots_ratio==0:
                    hot_points.append([x[i,j],y[i,j]])
                
        self.interior_points=np.array(vertices)   
        self.hot_points =np.array(hot_points)
        self.data={
                "ev": self.ev,
                "principal_ev":self.principal_ev,
                "interior_points": self.interior_points[np.lexsort(np.fliplr(self.interior_points).T)],
               "hot_points": self.hot_points[np.lexsort(np.fliplr(self.hot_points).T)],
                 "generators": self.generators,
                "legit": True,
                'a':self.a,
                'b':self.b,
                'M':self.M,
                'moments':self.moments,
                'type':'rectangle'
             }
        


    
    def save(self, path):
        torch.save(self.data, path)

    @classmethod
    def solve_helmholtz_equation(cls,f,*args):
        domain,mode= args
        b=f(domain['interior_points'][:,0], domain['interior_points'][:,1])

        assert abs(mode-Constants.k)>1e-6
        return 1/(mode-Constants.k)*b







class Polygon:
    def __init__(self, generators ):
        self.generators=generators
        # self.generators= (np.sqrt(math.pi) / np.sqrt(polygon_centre_area(generators))) * generators
        # self.diameter=np.max(np.linalg.norm(self.generators))
        # v[:, 0] -= np.mean(v[:, 0])
        # v[:, 1] -= np.mean(v[:, 1])
        self.geo = dmsh.Polygon(self.generators)
        self.moments=[calc_moment(k,calc_coeff(self.generators)[0],calc_coeff(self.generators)[1]) for k in range(30)]


    def create_mesh(self, h):
      
        if np.min(calc_min_angle(self.geo)) > (math.pi / 8):
            X, cells = dmsh.generate(self.geo, h )

            X, cells = optimesh.optimize_points_cells(
                X, cells, "CVT (full)", 1.0e-6, 120
            )
        # dmsh.show(X, cells, self.geo)
            
        self.vertices=X
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
        self.interior_points=self.vertices[self.interior_indices]

        self.hot_points=spread_points(int((int(1/Constants.h)-2)**2/(Constants.hot_spots_ratio**2)), self.interior_points)
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
            data={
                "ev": self.ev,
                "principal_ev":self.ev[-1],
                "interior_points": self.interior_points[np.lexsort(np.fliplr(self.interior_points).T)],
                "hot_points": self.hot_points[np.lexsort(np.fliplr(self.hot_points).T)],
                "generators": self.generators,
                "M":self.M[self.interior_indices][:, self.interior_indices],
                'moments':self.moments,
                "legit": True,
                'type':'polygon'
             }
            torch.save(data, path)

    def plot(self):
        plt.scatter(self.interior_points[:,0], self.interior_points[:,1],color='black')
        plt.scatter(self.hot_points[:,0], self.hot_points[:,1],color='red')
        plt.show()

    @classmethod
    def solve_helmholtz_equation(self,func,domain):

        solution=Test_function(domain['generators'], True)
        return np.array(list(map(solution,domain['interior_points'][:,0], domain['interior_points'][:,1])))
        return 
        M=domain['M']
        b=func(domain['interior_points'][:,0], domain['interior_points'][:,1])
        A = -M - Constants.k * scipy.sparse.identity(
        len(domain['interior_points'])
    )
        solution=scipy.sparse.linalg.spsolve(A, b)

        return solution
def analyze_momnets(path, col):         
    domain=torch.load(path)
    n=domain['generators'].shape[0]
    x1=np.array([[domain['moments'][l].real,domain['moments'][l].imag] for l in range(12)])
    x=x1[:,0]
    y=x1[:,1]
    # plt.plot(x,col)
    plt.scatter(range(y.shape[0]),y/6,color=col)

 
    # vertices=[domain['generators'][i] for i in range(n)]
    # a,Z=calc_coeff(vertices)
    # n_moments=2*n
    # tau= [calc_moment(k,a,Z) for k in range(n_moments)]
    # V=np.zeros((int(n_moments/2),int(n_moments/2)), dtype=np.complex_)
    # for i in range(int(n_moments/2)):
    #     for j in range(int(n_moments/2)):
    #         V[i,j]=Z[j]**i
    # A=np.diag(a)
    # H=V@A@V.T  
    # b=np.array(tau[-int(n_moments/2):])
    # p=list(np.linalg.solve(H,-b))
    # p.reverse()
    # print(np.roots([1]+p))
if __name__=='__main__':
    pass

    polygons_files_names = extract_path_from_dir(Constants.path + "polygons/")
    test_domains_path=[polygons_files_names[10] ]
    train_domains_path=polygons_files_names
    # domain=torch.load(train_domains_path[0])
    # print(domain['moments'][3])
    # # print(domain['interior_points'].shape)
    # a=domain['a']
    # b=domain['b']
    # func=sin_function(1,1,a,b).call
    # x1=np.array(list(map(func, domain['hot_points'][:,0], domain['hot_points'][:,1])))
    # domain=torch.load(test_domains_path[0])
    # print(domain['moments'][3])
    # a=domain['a']
    # b=domain['b']
    # func=sin_function(1,1,a,b).call
    # x2=np.array(list(map(func, domain['hot_points'][:,0], domain['hot_points'][:,1])))



    # plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1],c=list(map(func, domain['hot_points'][:,0], domain['hot_points'][:,1])))
    # plt.colorbar()
    # plt.show()
    # print(Constants.k-math.pi**2*(2**2/a**2+1**2/b**2))

    # for name in train_domains_path:
    #     analyze_momnets(name,'r')
    # for name in test_domains_path:
    #     analyze_momnets(name,'b')

    # plt.show()
    

def eval_on_domain(path):
    from functions.functions import sin_function
    domain=torch.load(path)
    f=sin_function(4,4,domain['a'], domain['b']).call
    x=list(domain['hot_points'][:,0])
    y=list(domain['hot_points'][:,1])
    solution=list(map(f,x,y))
    plt.scatter(x,y,c=solution)
    plt.show()

# eval_on_domain(Constants.path + "polygons/rect00.pt")        

# print(vertices)
    # Z.reverse()
    # p=np.array(Z.copy())
    
    
# With square kernels and equal stride
# m = torch.nn.Conv2d(16, 33, (3,3), padding_mode='zeros')
# non-square kernels and unequal stride and with padding
# m = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# # non-square kernels and unequal stride and with padding and dilation
# m = torch.nn.Conv1d(50, 50, 3, padding=1)
# input = torch.randn(20, 50,50)
# output = m(input)
# print(output.shape)


  

   
    





# domain=torch.load(Constants.path + "polygons/rect00.pt")
# domain=torch.load(Constants.path + "polygons/lshape.pt")
# print(len(domain['hot_points']))
# pass
# # domain=torch.load(Constants.path + "polygons/lshape.pt")
# plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1])
# plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1],color='red')
# plt.show()
# print(Constants.h)
# print(len(domain['hot_points']))
# p=Polygon(np.array([[0, 0], [1, 0], [1, 1 / 4], [1 / 4, 1 / 4], [1 / 4, 1], [0, 1]]))

# p=rectangle(1,1)
# p.create_mesh(Constants.h)
# x=[]
# y=[]
# for m in p.moments:
#     x.append(m.real)
#     y.append(m.imag)
# plt.plot(x[:15])
# plt.show()    

# p.plot()
# pass
# rect=rectangle(1,2)
# rect.create_mesh(Constants.h)
# rect.plot()
# print(rect.generators)
# rect.save(Constants.path + "polygons/rect_train.pt")
# p=Polygon(np.array([[0, 0], [1, 0], [1, 1 / 4], [1 / 4, 1 / 4], [1 / 4, 1], [0, 1]]))

# a=0.5
# b=1/a
# a= a*np.sqrt(math.pi/a/b)
# b= b*np.sqrt(math.pi/a/b)
# p=Polygon(np.array([[0, 0], [a, 0], [a, b], [0,b]]))
# p.create_mesh(Constants.h)
# print(p.generators)

# p.save(Constants.path + "polygons/lshape.pt")
# domain=torch.load(Constants.path + "polygons/rect_train_1.pt")

# plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1], color='red')
# plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1],color='black')
# plt.show()
# print(len(domain['hot_points']))
# print(int((int(1/Constants.h)-2)**2/(Constants.hot_spots_ratio**2)))
# domain=torch.load(Constants.path + "polygons/rect_train.pt")
# print(len(domain['hot_points']))

# u=Polygon.solve_helmholtz_equation(np.sin,domain)
# plt.scatter(domain['interior_points'][:,0], domain['interior_points'][:,1])
# # plt.scatter(domain['hot_points'][:,0], domain['hot_points'][:,1],color='red')

# plt.show()



def print_mommets(domain,col):
    x1=np.array([[domain.moments[l].real,domain.moments[l].imag] for l in range(5)])
    x=x1[:,0]
    y=x1[:,1]
    # plt.plot(x,col)
    # plt.scatter(range(y.shape[0]),y,color=col)
    plt.scatter(range(x.shape[0]),x,color=col)
    
# p1=Polygon(0.5*np.array([[2, 0], [3, 1], [3, 2], [1, 2], [1 , 3], [0, 2], [1,1],[2,1]]))
# p2=Polygon(0.5*np.array([[2, 0], [2, 1], [3, 1], [2, 2], [1 , 2], [1, 3],[0,3] ,[0,2]]))
# p1.create_mesh(0.05)
# p2.create_mesh(0.05)
# print(p1.ev)
# print(p2.ev)
# print_mommets(p1, 'r')
# print_mommets(p2, 'b')
# plt.show()

# p=Polygon(np.array([[0, 0], [1, 0], [0.8, 1 ], [0.5,1]]))
# p.create_mesh(Constants.h)
# p.save(Constants.path + "hints_polygons/trapz.pt")


# domain=torch.load(Constants.path + "hints_polygons/trapz.pt")
# # domain=torch.load(Constants.path + "polygons/rect00.pt")
# print(domain['hot_points'].shape)


# # f=Test_function(domain)
# # # print(f"{f.v[0]}  {f.v[1]}")
# plt.scatter(domain['hot_points'][:, 0], domain['hot_points'][:, 1])
# plt.show()