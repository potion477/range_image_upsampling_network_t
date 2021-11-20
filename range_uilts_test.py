from src.range_image_utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Range Image & PointCloud Conversion Test')
    parser.add_argument('--data', default='../../Data/kitti/training/velodyne/000002.bin', help='image directory to upsample')

    args = parser.parse_args()

    points = load_from_bin(args.data)
    # points = np.random.randn(10000,4).astype(np.float32)
    img_size = (64, 2048)
    v_fov = (-24.9, 2.0)
    h_fov = (-180, 180)

    """ Calculate resolution of angle """
    v_res = (v_fov[1]-v_fov[0]) / (img_size[0]-1)
    h_res = (h_fov[1]-h_fov[0]) / (img_size[1]-1)

    range_img = convert_ptcloud2rangeimage(points, v_res, h_res)
    ptcloud_restored = convert_rangeimage2ptcloud(range_img, v_res, h_res)
    
    r2 = convert_ptcloud2rangeimage(ptcloud_restored, v_res, h_res)
    p2 = convert_rangeimage2ptcloud(r2, v_res, h_res)

    r3 = convert_ptcloud2rangeimage(p2, v_res, h_res)
    p3 = convert_rangeimage2ptcloud(r3, v_res, h_res)

    r4 = convert_ptcloud2rangeimage(p3, v_res, h_res)
    p4 = convert_rangeimage2ptcloud(r4, v_res, h_res)

    print(points.shape)
    print(ptcloud_restored.shape)
    print(p2.shape)
    print(p3.shape)
    print(p4.shape)

    import matplotlib.pyplot as plt
    
    """ display result image """
    plt.subplots(1,1, figsize = (26,6))
    plt.title("Result of Vertical FOV ({} , {}) & Horizontal FOV ({} , {})".format(v_fov[0],v_fov[1],h_fov[0],h_fov[1]))
    plt.imshow(range_img[:,:,0])
    plt.axis('off')
    # plt.savefig("output.png")

    plt.subplots(1,1, figsize = (26,6))    
    plt.title("Result of Vertical FOV ({} , {}) & Horizontal FOV ({} , {})".format(v_fov[0],v_fov[1],h_fov[0],h_fov[1]))
    plt.imshow(r2[:,:,0])
    plt.axis('off')
    # plt.savefig("output2.png")