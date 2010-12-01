import numpy
import numpy.linalg
from numpy import *

def rigit(body, world):
    assert shape(world)[0] == 3 and shape(body)[0] == 3, 'Only accepts points in R^3. Maybe try transposing the data matrices?'
    assert shape(world)[1] == shape(body)[1], 'Number of world points and body points does not match'
    npoints = shape(world)[1]

    centroid_body = mean(body, 1)
    centroid_world = mean(world, 1)

    body_nc = body - tile(centroid_body, (1,npoints))
    world_nc = world - tile(centroid_world, (1,npoints))

    M = mat(zeros((4,4)))

    for i in range(npoints):
        a = concatenate((mat(0.), body_nc[:,i]))
        b = concatenate((mat(0.), world_nc[:,i]))

        Ma = mat([[a[0,0], -a[1,0], -a[2,0], -a[3,0]],
                  [a[1,0], a[0,0], a[3,0], -a[2,0]],
                  [a[2,0], -a[3,0], a[0,0], a[1,0]],
                  [a[3,0], a[2,0], -a[1,0], a[0,0]]])
        
        Mb = mat([[b[0,0], -b[1,0], -b[2,0], -b[3,0]],
                  [b[1,0], b[0,0], -b[3,0], b[2,0]],
                  [b[2,0], b[3,0], b[0,0], -b[1,0]],
                  [b[3,0], -b[2,0], b[1,0], b[0,0]]])

        M = M + Ma.T*Mb

    E, D = numpy.linalg.eig(M)

    max_idx = E.argmax()

    e = D[:,max_idx]

    M1 = mat([[e[0,0], -e[1,0], -e[2,0], -e[3,0]],
              [e[1,0], e[0,0], e[3,0], -e[2,0]],
              [e[2,0], -e[3,0], e[0,0], e[1,0]],
              [e[3,0], e[2,0], -e[1,0], e[0,0]]])
        
    M2 = mat([[e[0,0], -e[1,0], -e[2,0], -e[3,0]],
              [e[1,0], e[0,0], -e[3,0], e[2,0]],
              [e[2,0], e[3,0], e[0,0], -e[1,0]],
              [e[3,0], -e[2,0], e[1,0], e[0,0]]])

    R4 = M1.T * M2

    R = R4[1:,1:]

    T = centroid_world - R*centroid_body

    return R, T

if __name__ == '__main__':

    body = mat(numpy.random.rand(3,10))
    
    R = mat([[2./3, 2./3, -1./3], [-1./3, 2./3, 2./3], [2./3, -1./3, 2./3]])
    T = mat([[1.], [2.], [3.]])

    world = R * body + T 

    R_est, T_est = rigit(body, world)

    print R_est
    print T_est


