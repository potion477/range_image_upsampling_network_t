import numpy as np
import torch
from torchvision import transforms
def rav2xyz(rav_ndarray):
    """ Convert rav(range/azimuth/vertical) coordinate to xyz coordinate """

    r = rav_ndarray[:, 0]
    a = np.deg2rad(rav_ndarray[:,1])
    v = np.deg2rad(rav_ndarray[:,2])

    x = r * np.cos(v) * np.cos(a)
    y = r * np.cos(v) * np.sin(a)
    z = r * np.sin(v)

    return np.stack((x, y, z), axis=1)


def xyz2rav(xyz_ndarray):
    """ Convert xyz coordinate to rav(range/azimuth/vertical) coordinate """

    x = xyz_ndarray[:,0]
    y = xyz_ndarray[:,1]
    z = xyz_ndarray[:,2]

    dist_xy = np.sqrt(x**2 + y**2)

    a = np.rad2deg(np.arctan2(y, x))
    v = np.rad2deg(np.arctan2(z, dist_xy))
    r = np.sqrt(x**2 + y**2 + z**2)
    
    return np.stack((r, a, v), axis=1)


def convert_ndarry2tensor(x, batch_size=1):
    """ Convert ndarray (H x W x C) to Tensor (B x C x H x W) """

    row, col, channel = map(int, x.shape)

    transform = transforms.Compose([transforms.ToTensor()])
    y = transform(x)

    if batch_size==None:
        return y
    else:
        return torch.reshape(y, (batch_size, channel, row, col))


def convert_tensor2ndarray(x):
    batch_size, channel, row, col = map(int, x.shape)

    x = x.reshape(channel, row, col)
    y = x.detach().numpy()

    # Convert from (C, H, W) to (H, W, C)
    y = y.transpose((1, 2, 0))
    
    return y


def load_from_bin(bin_path):
    """ Load KITTI point cloud data from '.bin' file. """

    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj


def filter(points, x, y, z, dist_xy, v_fov, h_fov):
    """ Filtering points absed on h_fov & v_fov """
    
    h_points = np.logical_and(np.arctan2(y,x) >= np.deg2rad(h_fov[0]-0.01), 
                              np.arctan2(y,x) <= np.deg2rad(h_fov[1]+0.01))
    v_points = np.logical_and(np.arctan2(z,dist_xy) >= np.deg2rad(v_fov[0]-0.01),
                              np.arctan2(z,dist_xy) <= np.deg2rad(v_fov[1]+0.01))
    
    return points[np.logical_and(h_points, v_points)]


def convert_ptcloud2rangeimage(points, v_res, h_res, v_fov=(-24.9, 2.0), h_fov=(-180, 180), rixyz=False):
    """ Convert to range image from point cloud """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    reflectivity = points[:, 3]
    dist = np.sqrt(x**2 + y**2 + z**2)
    dist_xy = np.sqrt(x**2 + y**2)

    """ Project point cloud in cylinderical map """
    x_img = np.rad2deg(np.arctan2(y, x)) / h_res
    y_img = -np.rad2deg(np.arctan2(z, dist_xy)) / v_res
    
    """ Filtering points based on h_fov & v_fov """
    x_img = filter(x_img, x, y, z, dist_xy, v_fov, h_fov)
    y_img = filter(y_img, x, y, z, dist_xy, v_fov, h_fov)
    dist = filter(dist, x, y, z, dist_xy, v_fov, h_fov)
    reflectivity = filter(reflectivity, x, y, z, dist_xy, v_fov, h_fov)

    """ Shift negative points to positive poitns (shfit minimum value to 0) """
    x_offset = h_fov[0] / h_res
    x_img = np.round(x_img - x_offset, 0).astype(np.int32)
    
    y_offset = v_fov[1] / v_res
    y_img = np.round(y_img + y_offset, 0).astype(np.int32)
   
    """ Define Range Image Matrix """
    x_size = int(np.ceil((h_fov[1]-h_fov[0]) / h_res)) + 1
    y_size = int(np.ceil((v_fov[1]-v_fov[0]) / v_res)) + 1

    if rixyz == False:
        range_img = np.zeros((y_size, x_size, 2), dtype=np.float32)
        range_img[y_img, x_img, 0] = dist
        range_img[y_img, x_img, 1] = reflectivity
    else:
        """ Filtering points based on h_fov & v_fov """
        x_filter = filter(x, x, y, z, dist_xy, v_fov, h_fov)
        y_filter = filter(y, x, y, z, dist_xy, v_fov, h_fov)
        z_filter = filter(z, x, y, z, dist_xy, v_fov, h_fov)

        range_img = np.zeros((y_size, x_size, 5), dtype=np.float32)
        range_img[y_img, x_img, 0] = dist
        range_img[y_img, x_img, 1] = reflectivity
        range_img[y_img, x_img, 2] = x_filter
        range_img[y_img, x_img, 3] = y_filter
        range_img[y_img, x_img, 4] = z_filter
        
    return range_img


def convert_rangeimage2ptcloud(range_image, v_res, h_res, v_fov=(-24.9, 2.0), h_fov=(-180, 180)):
    """ Convert to point cloud from range image """
    
    elev_arr = np.arange(v_fov[1], v_fov[0]-v_res, -v_res)
    azim_arr = np.arange(h_fov[0], h_fov[1]+h_res, h_res)

    num_point = np.count_nonzero(range_image[:,:,0])
    
    rav = np.zeros((num_point, 3), dtype=np.float32)
    reflectivity = np.zeros((num_point, 1), dtype=np.float32)

    """ Convert range image into point cloud format """
    k = 0
    for i in range(len(azim_arr)):
        for j in range(len(elev_arr)):
            if range_image[j][i][0] != 0:
                rav[k] = [range_image[j][i][0], azim_arr[i], elev_arr[j]]
                reflectivity[k] = [range_image[j][i][1]]
                k = k + 1

    ptcloud = np.concatenate((rav2xyz(rav), reflectivity), axis=1)    
    
    return ptcloud

