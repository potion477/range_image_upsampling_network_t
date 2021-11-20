import os
import numpy as np
import argparse
import cv2

from src.range_image_utils import convert_ptcloud2rangeimage, load_from_bin

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save range image given point cloud directory (64,32,16, interpolation-based method)')
    parser.add_argument('--data_dir', default='../../Data/kitti/training/velodyne', help='data directory')
    parser.add_argument('--purpose', default='/training', help='choose sub directory(/training  /validation)')
    
    args = parser.parse_args()

    save_64_dir = '../../Data/kitti_range_image/64' + args.purpose
    save_32_dir = '../../Data/kitti_range_image/32' + args.purpose
    save_16_dir = '../../Data/kitti_range_image/16' + args.purpose
    save_64_nn_dir = '../../Data/kitti_range_image/64_nn' + args.purpose
    save_64_bilinear_dir = '../../Data/kitti_range_image/64_bilinear' + args.purpose
    save_64_bicubic_dir = '../../Data/kitti_range_image/64_bicubic' + args.purpose
    save_64_lanczos_dir = '../../Data/kitti_range_image/64_lanczos' + args.purpose

    save_dirs = [save_64_dir, save_32_dir, save_16_dir, save_64_nn_dir, 
                 save_64_bilinear_dir, save_64_bicubic_dir, save_64_lanczos_dir]

    for save_dir in save_dirs:
        os.makedirs(save_dir, exist_ok=True)
    
    data_list = sorted([os.path.join(args.data_dir,_) 
                            for _ in os.listdir(args.data_dir) if _.endswith('.bin')])
    save_list = sorted([os.path.splitext(_)[0]
                            for _ in os.listdir(args.data_dir) if _.endswith('.bin')])
    
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

        range_image_64 = convert_ptcloud2rangeimage(ptcloud, v_res, h_res)
        range_image_32 = range_image_64[1::2, :, :]
        range_image_16 = range_image_64[1::4, :, :]

        """ Upsampled Range Image using Interpolation (in OpenCV) """
        range_image_64_nn = cv2.resize(range_image_16, None, fx=1, fy=4, interpolation=cv2.INTER_NEAREST)
        range_image_64_bilinear = cv2.resize(range_image_16, None, fx=1, fy=4, interpolation=cv2.INTER_LINEAR)
        range_image_64_bicubic = cv2.resize(range_image_16, None, fx=1, fy=4, interpolation=cv2.INTER_CUBIC)
        range_image_64_lanczos = cv2.resize(range_image_16, None, fx=1, fy=4, interpolation=cv2.INTER_LANCZOS4)

        np.save(os.path.join(save_64_dir,save_name), range_image_64)
        np.save(os.path.join(save_32_dir,save_name), range_image_32)
        np.save(os.path.join(save_16_dir,save_name), range_image_16)
        
        np.save(os.path.join(save_64_nn_dir,save_name), range_image_64_nn)
        np.save(os.path.join(save_64_bilinear_dir,save_name), range_image_64_bilinear)
        np.save(os.path.join(save_64_bicubic_dir,save_name), range_image_64_bicubic)
        np.save(os.path.join(save_64_lanczos_dir,save_name), range_image_64_lanczos)