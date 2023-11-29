import numpy as np
from numpy import random

def corners(bound_min, bound_max):
    delta = 0.01
    corners = np.array([[bound_min[0] - delta, bound_min[1] - delta, bound_min[2] - delta],
                        [bound_max[0] + delta, bound_min[1] - delta, bound_min[2] - delta],
                        [bound_min[0] - delta, bound_max[1] + delta, bound_min[2] - delta],
                        [bound_min[0] - delta, bound_min[1] - delta, bound_max[2] + delta],
                        [bound_max[0] + delta, bound_max[1] + delta, bound_min[2] - delta],
                        [bound_min[0] - delta, bound_max[1] + delta, bound_max[2] + delta],
                        [bound_max[0] + delta, bound_min[1] - delta, bound_max[2] + delta],
                        [bound_max[0] + delta, bound_max[1] + delta, bound_max[2] + delta]], dtype = np.float32)
    return corners

def sample_sphere(bound_min, bound_max, N, R, perturb_f = 0.0):
    X = np.linspace(bound_min[0], bound_max[0], N)
    Y = np.linspace(bound_min[1], bound_max[1], N)
    Z = np.linspace(bound_min[2], bound_max[2], N)
    pts = np.zeros([N, N, N, 3], dtype=np.float32)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                pts[xi, yi, zi, :] = [xs, ys, zs]
    pts = pts.reshape(-1, 3)
    pts_norm = np.linalg.norm(pts, ord=2, axis=-1, keepdims=True)

    samples = pts[pts_norm[:,0] < R, :]
    if perturb_f > 0.0:
        samples = samples + perturb_f*random.rand(samples.shape[0], 3)

    # add 8 points at the corners
    
    """rand_vals = 0.0*random.rand(8, 3)
    corners = np.array([[bound_min[0] - rand_vals[0,0], bound_min[1] - rand_vals[0,1], bound_min[2] - rand_vals[0,2]],
                        [bound_max[0] + rand_vals[1,0], bound_min[1] - rand_vals[1,1], bound_min[2] - rand_vals[1,2]],
                        [bound_min[0] - rand_vals[2,0], bound_max[1] + rand_vals[2,1], bound_min[2] - rand_vals[2,2]],
                        [bound_min[0] - rand_vals[3,0], bound_min[1] - rand_vals[3,1], bound_max[2] + rand_vals[3,2]],
                        [bound_max[0] + rand_vals[4,0], bound_max[1] + rand_vals[4,1], bound_min[2] - rand_vals[4,2]],
                        [bound_min[0] - rand_vals[5,0], bound_max[1] + rand_vals[5,1], bound_max[2] + rand_vals[5,2]],
                        [bound_max[0] + rand_vals[6,0], bound_min[1] - rand_vals[6,1], bound_max[2] + rand_vals[6,2]],
                        [bound_max[0] + rand_vals[7,0], bound_max[1] + rand_vals[7,1], bound_max[2] + rand_vals[7,2]],])"""
    
    delta = 0.01
    corners = np.array([[bound_min[0] - delta, bound_min[1] - delta, bound_min[2] - delta],
                        [bound_max[0] + delta, bound_min[1] - delta, bound_min[2] - delta],
                        [bound_min[0] - delta, bound_max[1] + delta, bound_min[2] - delta],
                        [bound_min[0] - delta, bound_min[1] - delta, bound_max[2] + delta],
                        [bound_max[0] + delta, bound_max[1] + delta, bound_min[2] - delta],
                        [bound_min[0] - delta, bound_max[1] + delta, bound_max[2] + delta],
                        [bound_max[0] + delta, bound_min[1] - delta, bound_max[2] + delta],
                        [bound_max[0] + delta, bound_max[1] + delta, bound_max[2] + delta],])

    #print(corners)
    #samples = np.concatenate((samples, corners))
    samples = samples.astype(np.float32)

    #print(samples.shape, " points sampled")
    return samples

def sample_Bbox(bound_min, bound_max, resolution, perturb_f = 0.0):
    res_x = (bound_max[0] - bound_min[0])/resolution
    res_y = (bound_max[1] - bound_min[1])/resolution
    res_z = (bound_max[2] - bound_min[2])/resolution
    X = np.linspace(bound_min[0], bound_max[0], int((bound_max[0]-bound_min[0])/res_x ))
    Y = np.linspace(bound_min[1], bound_max[1], int((bound_max[1]-bound_min[1])/res_y ))
    Z = np.linspace(bound_min[2], bound_max[2], int((bound_max[2]-bound_min[2])/res_z ))
    xX, yY, zZ = np.meshgrid(X, Y, Z)
    
    pts = np.stack((xX, yY, zZ), axis = -1).astype(np.float32)
    samples = pts.reshape(-1, 3)
    if perturb_f > 0.0:
        samples = samples + perturb_f*random.rand(samples.shape[0], 3)
    print(samples.shape[0], " points sampled")
    samples = samples.astype(np.float32)
    return samples
