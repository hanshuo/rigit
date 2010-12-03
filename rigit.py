import numpy
import numpy.linalg
from numpy import *

import time
import itertools

def rigit(body, world):
    assert shape(world)[0] == 3 and shape(body)[0] == 3, 'Only accepts points in R^3. Maybe try transposing the data matrices?'
    assert shape(world)[1] == shape(body)[1], 'Number of world points and body points does not match'
    npoints = shape(world)[1]

    centroid_body = mean(body, 1)
    centroid_world = mean(world, 1)

    body_nc = body - centroid_body[:,newaxis]
    world_nc = world - centroid_world[:,newaxis]

    M = zeros((4,4))

    for i in range(npoints):

        a = array([0., body_nc[0,i], body_nc[1,i], body_nc[2,i]])
        b = array([0., world_nc[0,i], world_nc[1,i], world_nc[2,i]])

        Ma = array([[a[0], -a[1], -a[2], -a[3]],
                  [a[1], a[0], a[3], -a[2]],
                  [a[2], -a[3], a[0], a[1]],
                  [a[3], a[2], -a[1], a[0]]])
        
        Mb = array([[b[0], -b[1], -b[2], -b[3]],
                  [b[1], b[0], -b[3], b[2]],
                  [b[2], b[3], b[0], -b[1]],
                  [b[3], -b[2], b[1], b[0]]])

        M = M + dot(Ma.T,Mb)

    E, D = numpy.linalg.eig(M)

    max_idx = E.argmax()

    e = D[:,max_idx]

    M1 = array([[e[0], -e[1], -e[2], -e[3]],
              [e[1], e[0], e[3], -e[2]],
              [e[2], -e[3], e[0], e[1]],
              [e[3], e[2], -e[1], e[0]]])

    M2 = array([[e[0], -e[1], -e[2], -e[3]],
              [e[1], e[0], -e[3], e[2]],
              [e[2], e[3], e[0], -e[1]],
              [e[3], -e[2], e[1], e[0]]])

    R4 = dot(M1.T, M2)

    R = R4[1:,1:]

    T = centroid_world - dot(R,centroid_body)

    # The root sum of squared error
    err = sqrt(sum((dot(R,body) + T[:,newaxis] - world)**2))

    return R, T, err

def best_matching(body, world):
    p = shape(body)[1]; q = shape(world)[1]

    err2 = 0
    idx_corresp = zeros(q)

    for i in range(q):
        dist_list = ((body - world[:,i][:,newaxis])**2).sum(0)
        err2_cur = dist_list.min()
        idx_corresp[i] = dist_list.argmin()
        err2 = err2 + err2_cur
    
    err = sqrt(err2)

    return err, idx_corresp
        
def rigit_ransac(body, world, max_iters, tol):
    assert hasattr(itertools, 'permutations'), 'Needs itertools.permutations(). Please make sure the system runs Python 2.6 or higher.'
    assert shape(world)[0] == 3 and shape(body)[0] == 3, 'Only accepts points in R^3. Maybe try transposing the data matrices?'

    p = shape(body)[1]; q = shape(world)[1]

    iter = 0
    err = 1e40                  # some large number
    idx_corresp = [];

    while iter < max_iters and err > tol:
        world_rand_idx = numpy.random.permutation(q)
        world_rand_pts = world[:, world_rand_idx[:4]]
    
        for body_rand_idx in itertools.permutations(range(p), 4):
            body_rand_pts = body[:, body_rand_idx]

            R, T, err_part = rigit(body_rand_pts, world_rand_pts)

            if err_part < tol:
                world_pts_tr = dot(R.T, world - T[:,newaxis])
                err, idx_corresp = best_matching(body, world_pts_tr)
        
            iter = iter + 1

            if iter > max_iters or err < tol:
                break

    if err < tol:
        is_successful = True
    else:
        is_successful = False
        
    return R, T, err, idx_corresp, is_successful, iter

if __name__ == '__main__':

    body = numpy.random.rand(3,10)
    
    R = array([[2./3, 2./3, -1./3], [-1./3, 2./3, 2./3], [2./3, -1./3, 2./3]])
    T = array([1., 2., 3.])

    world = dot(R, body) + T[:,newaxis]

    #  print body
    # print world

    # time_cur = time.time()
    # for i in range(1):

    #     R_est, T_est, err = rigit(body, world)

    # print 'Time elapsed = %g (s)' % (time.time() - time_cur)

    # print R_est
    # print T_est
    # print err

    time_start = time.time()
    R_ransac, T_ransac, err, idx_corresp, is_successful, iter = rigit_ransac(body, world[:,:8], 50000, 1e-8)
    time_elapsed = time.time() - time_start
    print 'total iterations = %d' % (iter)
    print 'Time elapsed = %g (s)' % (time_elapsed)
    print 'Iterations per second = %g' % (iter/time_elapsed)
    print R_ransac
    print T_ransac
    print err
    print idx_corresp


