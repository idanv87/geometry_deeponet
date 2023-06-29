__all__ = ['is_wellcentered', 'circumcenter', 'circumcenter_barycentric']

from numpy import bmat, hstack, vstack, dot, sqrt, ones, zeros, sum, \
        asarray
from numpy.linalg import solve,norm
import numpy as np

def orient(a,b,c):
      return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
     

def is_wellcentered(pts, tol=1e-8):
    """Determines whether a set of points defines a well-centered simplex.
    """
    barycentric_coordinates = circumcenter_barycentric(pts)    
    return min(barycentric_coordinates) > tol
#######################################################################
#######################################################################
#######################################################################
#######################################################################


def unsigned_volume(pts):
    """Unsigned volume of a simplex    
    
    Computes the unsigned volume of an M-simplex embedded in N-dimensional 
    space. The points are stored row-wise in an array with shape (M+1,N).
    
    Parameters
    ----------
    pts : array
        Array with shape (M+1,N) containing the coordinates
        of the (M+1) vertices of the M-simplex.

    Returns
    -------
    volume : scalar
        Unsigned volume of the simplex

    Notes
    -----
    Zero-dimensional simplices (points) are assigned unit volumes.
        

    Examples
    --------
    >>> # 0-simplex point 
    >>> unsigned_volume( [[0,0]] )
    1.0
    >>> # 1-simplex line segment
    >>> unsigned_volume( [[0,0],[1,0]] )             
    1.0
    >>> # 2-simplex triangle 
    >>> unsigned_volume( [[0,0,0],[0,1,0],[1,0,0]] ) 
    0.5


    References
    ----------
    [1] http://www.math.niu.edu/~rusin/known-math/97/volumes.polyh

    """       
    
    pts = asarray(pts)
    
    M,N = pts.shape
    M -= 1

    if M < 0 or M > N:
        raise ValueError('array has invalid shape')
    
    if M == 0:
        return 1.0 
        
    A = pts[1:] - pts[0]
    return sqrt(abs(np.linalg.det(np.inner(A,A))))/np.math.factorial(M)

if 0:    
        def circumcenter_barycentric(pts):

            rows,cols = pts.shape
            assert(rows <= cols + 1) 

            if rows==1:
                bary_coords=1
            if rows==2:
                p1=pts[0,:]
                p2=pts[1,:]

                center=circumcenter(pts)[0]
                rhs=center-p2
                lhs=p1-p2
                if abs(lhs[0])<1e-10:
                    l1=rhs[1]/lhs[1]
                else:
                    l1=rhs[0]/lhs[0]
            
                bary_coords=np.hstack((l1,1-l1))

            if rows==3:
                center=circumcenter(pts)[0]
                
                A=np.vstack((pts.T,np.ones((1,rows))))
                # A = bmat( [[ pts.T], [ones((1,rows))] ]
                    #   )
            
                b = np.hstack((center,np.array([1])) )
                

                x = np.linalg.solve(A,b)

                bary_coords = x 
                
            return bary_coords

            



            
            
        def circumcenter(pts):
            
            # pts = asarray(pts)
                
            # bary_coords = circumcenter_barycentric(pts)
            # center = dot(bary_fcoords,pts)
            center=weighted_circ(pts)



            radius = np.linalg.norm(pts[0,:] - center)
            return (center,radius)
            
            
            
        def weighted_circ(pts):

            s=pts[0]
            if len(pts)==2:
                    n=(pts[1]-pts[0])/unsigned_volume(pts)
                    s=s+(1/(2*unsigned_volume(pts)))* (
                    unsigned_volume(
                        np.vstack((pts[0],pts[1]))
                        )**2
                    )*n
            if len(pts)==3:
                
                
                if orient(pts[0],pts[1],pts[2])>0:
                  A=np.array([[0,-1],[1,0]])
                else:  
                  A=np.array([[0,-1],[1,0]]).T   
                
                n1=np.matmul(A,pts[0]-pts[2])
                n2=np.matmul(A,pts[1]-pts[0])
                n=[n1,n2]
               
            
                k=2
                for i in range(k):
                
                    s=s+(1/(2*np.math.factorial(k)*unsigned_volume(pts)))* (
                    unsigned_volume(
                        np.vstack((pts[0],pts[i+1]))
                        )**2
                    )*n[i]   
                
            return s
else:
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################

        def circumcenter_barycentric(pts):
            """Barycentric coordinates of the circumcenter of a set of points.
            
            Parameters
            ----------
            pts : array-like
                An N-by-K array of points which define an (N-1)-simplex in K dimensional space.
                N and K must satisfy 1 <= N <= K + 1 and K >= 1.

            Returns
            -------
            coords : ndarray
                Barycentric coordinates of the circumcenter of the simplex defined by pts.
                Stored in an array with shape (K,)
                
            Examples
            --------
            >>> from pydec.math.circumcenter import *
            >>> circumcenter_barycentric([[0],[4]])           # edge in 1D
            array([ 0.5,  0.5])
            >>> circumcenter_barycentric([[0,0],[4,0]])       # edge in 2D
            array([ 0.5,  0.5])
            >>> circumcenter_barycentric([[0,0],[4,0],[0,4]]) # triangle in 2D
            array([ 0. ,  0.5,  0.5])

            See Also
            --------
            circumcenter_barycentric
        
            References
            ----------
            Uses an extension of the method described here:
            http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html

            """    

            pts = asarray(pts)

            rows,cols = pts.shape

            assert(rows <= cols + 1)    

            A = bmat( [[ 2*dot(pts,pts.T), ones((rows,1)) ],
                       [  ones((1,rows)) ,  zeros((1,1))  ]] )

            b = hstack((sum(pts * pts, axis=1),ones((1))))
            x = solve(A,b)
            bary_coords = x[:-1]  

            return bary_coords
            
        def circumcenter(pts):
            """Circumcenter and circumradius of a set of points.
            
            Parameters
            ----------
            pts : array-like
                An N-by-K array of points which define an (N-1)-simplex in K dimensional space.
                N and K must satisfy 1 <= N <= K + 1 and K >= 1.

            Returns
            -------
            center : ndarray
                Circumcenter of the simplex defined by pts.  Stored in an array with shape (K,)
            radius : float
                Circumradius of the circumsphere that circumscribes the points defined by pts.
                
            Examples
            --------
            >>> circumcenter([[0],[1]])             # edge in 1D
            (array([ 0.5]), 0.5)
            >>> circumcenter([[0,0],[1,0]])         # edge in 2D
            (array([ 0.5,  0. ]), 0.5)
            >>> circumcenter([[0,0],[1,0],[0,1]])   # triangle in 2D
            (array([ 0.5,  0.5]), 0.70710678118654757)

            See Also
            --------
            circumcenter_barycentric
        
            References
            ----------
            Uses an extension of the method described here:
            http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html

            """
            pts = asarray(pts)      
            bary_coords = circumcenter_barycentric(pts)
            center = dot(bary_coords,pts)
            radius = norm(pts[0,:] - center)
            return (center,radius)
    
    
    
