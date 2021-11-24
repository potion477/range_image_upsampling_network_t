import os
import numpy as np
import argparse
import cv2

from src.range_image_utils import convert_ptcloud2rangeimage, load_from_bin

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save 5 channel(x,y,z,r,i) range image given point cloud directory (64,32,16)')
    parser.add_argument('--data_dir', default='../../Data/kitti', help='data directory')
    parser.add_argument('--save_dir', default='../../Data/kitti_range_image', help='save directory')
    parser.add_argument('--purpose', default='training', help='choose sub directory(training / validation)')
     
    args = parser.parse_args()

    read_dir = os.path.join(args.data_dir, args.purpose, 'velodyne')
    save_64_dir = os.path.join(args.save_dir, 'kitti_64_rixyz', args.purpose)
    save_32_dir = os.path.join(args.save_dir, 'kitti_32_rixyz', args.purpose)
    save_16_dir = os.path.join(args.save_dir, 'kitti_16_rixyz', args.purpose)


    save_dirs = [save_64_dir, save_32_dir, save_16_dir]

    for dir in save_dirs:
        os.makedirs(dir, exist_ok=True)
    
    data_list = sorted([os.path.join(read_dir,_) 
                            for _ in os.listdir(read_dir) if _.endswith('.bin')])
    save_list = sorted([os.path.splitext(_)[0]
                            for _ in os.listdir(read_dir) if _.endswith('.bin')])
    
    """ Image Size and FOV Setting (Recommend to do not change) """
    img_size = (64, 2048)
    v_fov = (-24.9, 2.0)
    h_fov = (-180, 180)
    v_res = (v_fov[1]-v_fov[0]) / (img_size[0]-1)
    h_res = (h_fov[1]-h_fov[0]) / (img_size[1]-1)


    """ Save Range Image """
    for idx, (data_dir,save_name) in enumerate(zip(data_list, save_list)):
        if idx % 100 == 0:
            print(f'{idx}th data is written.')
        ptcloud = load_from_bin(data_dir)

        range_image_64 = convert_ptcloud2rangeimage(ptcloud, v_res, h_res, rixyz=True)
        range_image_32 = range_image_64[1::2, :, :]
        range_image_16 = range_image_64[1::4, :, :]

        print(range_image_64.shape)
        print(range_image_32.shape)
        print(range_image_16.shape)


        np.save(os.path.join(save_64_dir,save_name), range_image_64)
        np.save(os.path.join(save_32_dir,save_name), range_image_32)
        np.save(os.path.join(save_16_dir,save_name), range_image_16)
        
