import os
import argparse
import numpy as np

from src.range_image_utils import convert_rangeimage2ptcloud

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save point cloud given range image (64,32,16, interpolation-based method)')
    parser.add_argument('--data_dir', default='../../Data/kitti_range_image', help='data directory')
    parser.add_argument('--save_dir', default='../../Data/kitti_point_cloud', help='save directory')
    parser.add_argument('--purpose', default='training', help='choose sub directory(training / validation)')

    args = parser.parse_args()

    read_dirs = [os.path.join(args.data_dir, 'kitti_64', args.purpose), 
                 os.path.join(args.data_dir, 'kitti_32', args.purpose), 
                 os.path.join(args.data_dir, 'kitti_16', args.purpose), 
                 os.path.join(args.data_dir, 'kitti_64_nn', args.purpose), 
                 os.path.join(args.data_dir, 'kitti_64_bilinear', args.purpose), 
                 os.path.join(args.data_dir, 'kitti_64_bicubic', args.purpose), 
                 os.path.join(args.data_dir, 'kitti_64_lanczos', args.purpose)]

    save_dirs = [os.path.join(args.save_dir, 'kitti_64', args.purpose, 'velodyne'), 
                 os.path.join(args.save_dir, 'kitti_32', args.purpose, 'velodyne'), 
                 os.path.join(args.save_dir, 'kitti_16', args.purpose, 'velodyne'), 
                 os.path.join(args.save_dir, 'kitti_64_nn', args.purpose, 'velodyne'), 
                 os.path.join(args.save_dir, 'kitti_64_bilinear', args.purpose, 'velodyne'), 
                 os.path.join(args.save_dir, 'kitti_64_bicubic', args.purpose, 'velodyne'), 
                 os.path.join(args.save_dir, 'kitti_64_lanczos', args.purpose, 'velodyne')]

    img_size = (64, 2048)
    v_fov = (-24.9, 2.0)
    h_fov = (-180, 180)

    """ Calculate resolution of angle """
    v_res = (v_fov[1]-v_fov[0]) / (img_size[0]-1)
    h_res = (h_fov[1]-h_fov[0]) / (img_size[1]-1)

    for dir in save_dirs:
        os.makedirs(dir, exist_ok=True)

    for i, (read_dir,save_dir) in enumerate(zip(read_dirs, save_dirs)):
        file_list = sorted([_ for _ in os.listdir(read_dir) if _.endswith('.npy')])
        for file_idx, file_name in enumerate(file_list):
            if file_idx % 100 == 0:
                print(f'{read_dir} / {file_idx}th data is written.')

            range_image = np.load(os.path.join(read_dir, file_name))

            """ Convert into pointcloud """
            ptcloud = convert_rangeimage2ptcloud(range_image, v_res, h_res)

            f, _ = os.path.splitext(file_name)
            ptcloud.tofile(os.path.join(save_dir, f+'.bin'))