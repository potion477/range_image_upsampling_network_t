import os
import argparse
import numpy as np

from src.range_image_utils import convert_rangeimage2ptcloud

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save point cloud given range image (interpolation-based method)')
    parser.add_argument('--data_dir', default='../../Data/kitti_range_image', help='data directory')
    parser.add_argument('--save_dir', default='../../Data/kitti_point_cloud', help='save directory')
    parser.add_argument('--purpose', default='training', help='choose sub directory(training / validation)')

    args = parser.parse_args()

    read_dir = os.path.join(args.data_dir, 'kitti_64', args.purpose)

    save_dirs = [os.path.join(args.save_dir, 'kitti_64', args.purpose), 
                 os.path.join(args.save_dir, 'kitti_32', args.purpose), 
                 os.path.join(args.save_dir, 'kitti_16', args.purpose)]

    img_size = (64, 2048)
    v_fov = (-24.9, 2.0)
    h_fov = (-180, 180)

    """ Calculate resolution of angle """
    v_res = (v_fov[1]-v_fov[0]) / (img_size[0]-1)
    h_res = (h_fov[1]-h_fov[0]) / (img_size[1]-1)

    for dir in save_dirs:
        os.makedirs(dir, exist_ok=True)

    file_list = sorted([_ for _ in os.listdir(read_dir) if _.endswith('.npy')])

    for file_idx, file_name in enumerate(file_list):
        if file_idx % 100 == 0:
            print(f'{file_idx}th data is written.')

        range_image_64 = np.load(os.path.join(read_dir, file_name))
        range_image_32 = range_image_64.copy()
        range_image_16 = range_image_64.copy()

        for i in range(64):
            if i // 2 != 1:
                range_image_32[i,:,:] = 0
            if i // 4 != 1:
                range_image_16[i,:,:] = 0

        """ Convert into pointcloud """
        ptcloud_64 = convert_rangeimage2ptcloud(range_image_64, v_res, h_res)
        ptcloud_32 = convert_rangeimage2ptcloud(range_image_32, v_res, h_res)
        ptcloud_16 = convert_rangeimage2ptcloud(range_image_16, v_res, h_res)

        f, _ = os.path.splitext(file_name)
        ptcloud_64.tofile(os.path.join(save_dirs[0], f+'.bin'))
        ptcloud_32.tofile(os.path.join(save_dirs[1], f+'.bin'))
        ptcloud_16.tofile(os.path.join(save_dirs[2], f+'.bin'))