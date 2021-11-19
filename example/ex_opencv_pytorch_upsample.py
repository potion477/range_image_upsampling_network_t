import sys
import argparse

import cv2
import torch
import torch.nn.functional as F

def opencv_upsample(args):
    """ Load the Image """
    filename = args.image
    src = cv2.imread(filename)

    """ Check if image is loaded fine """
    if src is None:
        print('[Error] File does not exist!!!')
        return -1

    dst = src.copy()

    while 1:
        row, col, channel = map(int, dst.shape)
        cv2.imshow('Raw Image', src)
        cv2.imshow('Resize Demo', dst)
        k = cv2.waitKey(0)

        """ Resize the image """
        if k == 27:
            break    
        elif chr(k) == 'q':
            dst = cv2.pyrUp(dst, dstsize=(2*col, 2*row))
            print('** Zoom in[pyrUp]: Image x 2')
        elif chr(k) == 'w':
            dst = cv2.pyrDown(dst, dstsize=(col//2, row//2))
            print('** Zoom out[pyrDown]: Image / 2 ')
        elif chr(k) == 'e':
            dst = cv2.resize(dst, None, fx=1, fy=2, interpolation=cv2.INTER_LINEAR)
            print('** Zoom in[BILINEAR]: Image x 2')
        elif chr(k) == 'r':
            dst = cv2.resize(dst, None, fx=1, fy=0.5, interpolation=cv2.INTER_LINEAR)
            print('** Zoom out[BILINEAR]: Image / 2')
        elif chr(k) == 't':
            dst = cv2.resize(dst, None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
            print('** Zoom in[BICUBIC]: Image x 2')
        elif chr(k) == 'y':
            dst = cv2.resize(dst, None, fx=1, fy=0.5, interpolation=cv2.INTER_CUBIC)
            print('** Zoom out[BICUBIC]: Image / 2')

    cv2.destroyAllWindows()
    return 0


def pytorch_upsample(args):
    filename = args.image
    src = cv2.imread(filename)

    """ Check if image is loaded fine """
    if src is None:
        print('[Error] File does not exist!!!')
        return -1

    src_tensor = torch.from_numpy(src).permute(2,0,1).unsqueeze(0)
    print(src_tensor.shape)

    dst_tensor = torch.tensor(src_tensor)

    while 1:
        batch, channel, row, col = map(int, dst_tensor.shape)
        cv2.imshow('Raw Image', src_tensor.squeeze().permute(1,2,0).numpy())
        cv2.imshow('Resize Demo', dst_tensor.squeeze().permute(1,2,0).numpy())      
        k = cv2.waitKey(0)

        """ Resize the image """
        if k == 27:
            break    
        elif chr(k) == 'q':
            dst_tensor = F.interpolate(dst_tensor, scale_factor=(2,1), mode='nearest')
            print('** Zoom in[NN]: Image x 2')
        elif chr(k) == 'w':
            dst_tensor = F.interpolate(dst_tensor, scale_factor=(0.5,1), mode='nearest')
            print('** Zoom out[NN]: Image / 2 ')
        elif chr(k) == 'e':
            dst_tensor = F.interpolate(dst_tensor, scale_factor=(2,1), mode='bilinear')
            print('** Zoom in[BILINEAR]: Image x 2')
        elif chr(k) == 'r':
            dst_tensor = F.interpolate(dst_tensor, scale_factor=(0.5,1), mode='bilinear')
            print('** Zoom out[BILINEAR]: Image / 2')
        elif chr(k) == 't':
            dst_tensor = F.interpolate(dst_tensor, scale_factor=(2,1), mode='bicubic')
            print('** Zoom in[BICUBIC]: Image x 2')
        elif chr(k) == 'y':
            dst_tensor = F.interpolate(dst_tensor, scale_factor=(0.5,1), mode='bicubic')
            print('** Zoom out[BICUBIC]: Image / 2')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upsample the selected image.')
    parser.add_argument('--image', default='example/test_img.jpg', help='image directory to upsample')
    parser.add_argument('--env', default='pytorch', help='Which environment to process. (opencv / pytorch)')

    args = parser.parse_args()

    if args.env == 'opencv':
        opencv_upsample(args)
    elif args.env == 'pytorch':
        pytorch_upsample(args)
        