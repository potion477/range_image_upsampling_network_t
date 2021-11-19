import numpy as np
import torch

def load_from_bin(bin_path):
    """ Load KITTI point cloud data from '.bin' file. """

    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


def filter(points, x, y, z, dist_xy, v_fov, h_fov):
    """ Filtering points absed on h_fov & v_fov """
    
    h_points = np.logical_and(np.arctan2(y,x) > np.deg2rad(h_fov[0]-0.5), 
                              np.arctan2(y,x) < np.deg2rad(h_fov[1]+0.5))
    v_points = np.logical_and(np.arctan2(z,dist_xy) > np.deg2rad(v_fov[0]-0.5),
                              np.arctan2(z,dist_xy) < np.deg2rad(v_fov[1]+0.5))
    
    return points[np.logical_and(h_points, v_points)]


def convert_ptcloud2rangeimage(points, v_res, h_res, v_fov=(-24.9, 2.0), h_fov=(-180, 180)):
    
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    reflectivity = points[:, 3]
    dist = np.sqrt(x**2 + y**2 + z**2)
    dist_xy = np.sqrt(x**2 + y**2)

    """ Project point cloud in cylinderical map """
    x_img = np.arctan2(y, x) / np.deg2rad(h_res)
    y_img = -(np.arctan2(z, dist_xy)) / np.deg2rad(v_res)
    
    """ Filtering points based on h_fov & v_fov """
    x_img = filter(x_img, x, y, z, dist_xy, v_fov, h_fov)
    y_img = filter(y_img, x, y, z, dist_xy, v_fov, h_fov)
    dist = filter(dist, x, y, z, dist_xy, v_fov, h_fov)
    reflectivity = filter(reflectivity, x, y, z, dist_xy, v_fov, h_fov) 

    """ Shift negative points to positive poitns (shfit minimum value to 0) """
    x_offset = h_fov[0] / h_res
    x_img = np.round(x_img-x_offset, 0).astype(np.int32)

    y_offset = v_fov[1] / v_res
    y_img = np.round(y_img + y_offset, 0).astype(np.int32)
   
    """ Define Range Image Matrix """
    x_size = int(np.ceil((h_fov[1]-h_fov[0]) / h_res)) + 1
    y_size = int(np.ceil((v_fov[1]-v_fov[0]) / v_res)) + 1
    range_img = np.zeros((y_size, x_size, 2), dtype=np.float32)
    range_img[y_img, x_img, 0] = dist
    range_img[y_img, x_img, 1] = reflectivity

    return range_img


if __name__ == '__main__':
    points = np.random.randn(10000, 4).astype(np.float32)
    
    img_size = (64, 2048)
    v_fov = (-24.9, 2.0)
    h_fov = (-180, 180)

    """ Calculate resolution of angle """
    v_res = (v_fov[1]-v_fov[0]) / (img_size[0]-1)
    h_res = (h_fov[1]-h_fov[0]) / (img_size[1]-1)


    range_img = convert_ptcloud2rangeimage(points, v_res, h_res)
