from copy import copy
import numpy as np
from numpy.random import default_rng
import sys
import open3d as o3d

rng = default_rng()
NUM_ITERATIONS = 100000
NUM_SAMPLES = 1000
THRESHOLD = 0.05
NUM_FIT = 100

class PARAMS:
    def __init__(self, rot, trans, s):
        self.scale = s
        self.rotation = rot
        self.translation = trans

class RESULTS:
    def __init__(self):
        self.best_params = None
        self.best_error = np.inf


results = RESULTS()

def square_error_loss(x, y):
    return (x - y) ** 2

def mean_square_error(x, y):
    return np.sum(square_error_loss(x, y)) / x.shape[0]

def transform(x, params): #mozna nejsou potreba jednicky
    r, _ = x.shape
    transformed = (params.scale * params.rotation @ x.T + np.tile(params.translation.reshape((3,1)),(1,r))).T
    return transformed 

def fit(x, y):
    centroid1 = np.mean(x, axis = 0)
    centroid2 = np.mean(y, axis = 0)

    x_centr = x - centroid1
    y_centr = y - centroid2

    s1 = np.sum(np.sqrt(np.sum(x_centr * x_centr)))
    s2 = np.sum(np.sqrt(np.sum(y_centr * y_centr)))

    K = np.zeros((3,3))
    for i in range(x.shape[0]):
        K = K + y[i].T @ x[i]
    [U, _, V] = np.linalg.svd(K)
    S = np.diag([1, 1, np.linalg.det(U) * np.linalg.det(V)])

    r = U @ S @ V
    s = s2/s1
    t = (centroid2 - s * r @ centroid1.T).T

    transformation = PARAMS(r,t,s)
    return transformation

def ransac_fit(pc1, pc2):
    x = np.asarray(pc1.points)
    y = np.asarray(pc2.points)

    

    for i in range(NUM_ITERATIONS):
        idx_permutation = rng.permutation(x.shape[0])
        possible_inliers = idx_permutation[0:NUM_SAMPLES]
        #print(possible_inliers.shape)
        possible_params = fit(x[possible_inliers], y[possible_inliers])

        filtered = square_error_loss(y[possible_inliers], transform(x[possible_inliers], possible_params)) < THRESHOLD

        inlier_idx = idx_permutation[np.flatnonzero(filtered[:,0]).flatten()] 
        #print(inlier_idx)
        #print(inlier_idx.shape)
        if inlier_idx.shape[0] > NUM_FIT:
            print("Found")
            error = mean_square_error(y[inlier_idx], transform(x[inlier_idx], possible_params))

            if error < results.best_error:
                results.best_error = error
                results.best_params = possible_params


if __name__ == "__main__":

    pc1 = o3d.io.read_point_cloud("source.ply")
    pc2 = o3d.io.read_point_cloud("target2.ply")

    ransac_fit(pc1, pc2)

    out_pc = o3d.geometry.PointCloud()
    #print(regressor.best_fit.params)
    transformed = transform(np.asarray(pc1.points), results.best_params)
    #print(type(transformed))
    #print(transformed.shape)
    out_pc.points = o3d.utility.Vector3dVector(transformed)
    o3d.io.write_point_cloud("aligned.ply", out_pc)
    