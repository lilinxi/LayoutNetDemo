import numpy as np
from scipy.ndimage import map_coordinates

import open3d
from PIL import Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import functools
from multiprocessing import Pool

from utils_eval import np_coor2xy, np_coory2v

"""
open3d 是右手坐标系，z 轴向外
"""


def xyz_2_coorxy(xs, ys, zs, H, W):
    # 3D-box xyz to coorxy
    us = np.arctan2(xs, -ys)  # (-pi, pi) 负号反转左右
    vs = -np.arctan(zs / np.sqrt(xs ** 2 + ys ** 2))  # (-pi/2, pi/2) 负号反转前后
    coorx = (us / (2 * np.pi) + 0.5) * W  # (-pi, pi) -> (-0.5, 0.5) -> (0, 1) -> (0, W)
    coory = (vs / np.pi + 0.5) * H  # (-pi/2, pi/2) -> (-0.5, 0.5) -> (0, 1) -> (0, H)
    return coorx, coory


def pt_in_poly(poly, pt):
    return poly.contains(Point(pt))


def warp_walls(xy, floor_z, ceil_z, H, W, ppm, alpha):
    all_rgba = []
    all_xyz = []
    for i in range(len(xy)):
        print("----------", i, "----------")
        next_i = (i + 1) % len(xy)
        xy_a = xy[i]
        xy_b = xy[next_i]
        xy_w = np.sqrt(((xy_a - xy_b) ** 2).sum())  # 底边长

        # TODO print
        print("xy_a, xy_b:", xy_a, xy_b)

        t_h = int(round((ceil_z - floor_z) * ppm))  # 像素高
        t_w = int(round(xy_w * ppm))  # 像素宽

        # TODO print
        print("t_h, t_w:", t_h, t_w)

        xs = np.linspace(xy_a[0], xy_b[0], t_w)[None].repeat(t_h, 0)  # (t_w,) -> (1, t_w) -> (t_h, t_w)
        ys = np.linspace(xy_a[1], xy_b[1], t_w)[None].repeat(t_h, 0)  # (t_w,) -> (1, t_w) -> (t_h, t_w)
        zs = np.linspace(floor_z, ceil_z, t_h)[:, None].repeat(t_w, 1)  # (t_h,) -> (t_h, 1) -> (t_h, t_w)

        # TODO print
        print("----------xs,ys,zs----------")
        print("xs.shape:", xs.shape)
        print("ys.shape:", ys.shape)
        print("zs.shape:", zs.shape)

        coorx, coory = xyz_2_coorxy(xs, ys, zs, H, W)

        # TODO print
        print("equirect_texture.shape:", equirect_texture.shape)  # (512, 1024, 3)
        print("equirect_texture[..., 0].shape:", equirect_texture[..., 0].shape)  # (512, 1024)

        plane_texture = np.stack([
            map_coordinates(equirect_texture[..., 0], [coory, coorx], order=1, mode='wrap'),  # r
            map_coordinates(equirect_texture[..., 1], [coory, coorx], order=1, mode='wrap'),  # g
            map_coordinates(equirect_texture[..., 2], [coory, coorx], order=1, mode='wrap'),  # b
            np.zeros([t_h, t_w]) + alpha,
        ], -1)
        plane_xyz = np.stack([xs, ys, zs], axis=-1)  # (t_h, t_w, 3)

        all_rgba.extend(plane_texture.reshape(-1, 4))
        all_xyz.extend(plane_xyz.reshape(-1, 3))

    return all_rgba, all_xyz


def warp_floor_ceiling(xy, z_floor, z_ceiling, H, W, ppm, alpha, n_thread):
    # 底层和顶层的轴对齐包围盒
    min_x = xy[:, 0].min()
    max_x = xy[:, 0].max()
    min_y = xy[:, 1].min()
    max_y = xy[:, 1].max()
    # 底层和顶层的轴对齐包围盒的像素 高 X 宽
    t_h = int(round((max_y - min_y) * ppm))
    t_w = int(round((max_x - min_x) * ppm))
    # 在轴对齐包围盒内采样
    xs = np.linspace(min_x, max_x, t_w)[None].repeat(t_h, 0)  # (t_w,) -> (1, t_w) -> (t_h, t_w)
    ys = np.linspace(min_y, max_y, t_h)[:, None].repeat(t_w, 1)  # (t_h,) -> (t_h, 1) -> (t_h, t_w)
    # 底层和顶层的 z 轴保持一致
    zs_floor = np.zeros_like(xs) + z_floor
    zs_ceil = np.zeros_like(xs) + z_ceiling
    # 转化成全景图的坐标
    coorx_floor, coory_floor = xyz_2_coorxy(xs, ys, zs_floor, H, W)
    coorx_ceil, coory_ceil = xyz_2_coorxy(xs, ys, zs_ceil, H, W)
    # map 到全景图上，获取 rgb 贴图
    floor_texture = np.stack([
        map_coordinates(equirect_texture[..., 0], [coory_floor, coorx_floor], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 1], [coory_floor, coorx_floor], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 2], [coory_floor, coorx_floor], order=1, mode='wrap'),
        np.zeros([t_h, t_w]) + alpha,
    ], -1).reshape(-1, 4)
    # stack 对叠 xyz 坐标
    floor_xyz = np.stack([xs, ys, zs_floor], axis=-1).reshape(-1, 3)  # (526, 530, 3) -> (278780, 3)

    # TODO print
    print("np.stack([xs, ys, zs_floor], axis=-1).shape:",
          np.stack([xs, ys, zs_floor], axis=-1).shape)  # (526, 530, 3)
    print("np.stack([xs, ys, zs_floor], axis=-1).reshape(-1, 3).shape:",
          np.stack([xs, ys, zs_floor], axis=-1).reshape(-1, 3).shape)  # (278780, 3)

    ceil_texture = np.stack([
        map_coordinates(equirect_texture[..., 0], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 1], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 2], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        np.zeros([t_h, t_w]) + alpha,
    ], -1).reshape(-1, 4)
    ceil_xyz = np.stack([xs, ys, zs_ceil], axis=-1).reshape(-1, 3)

    # 在轴对齐包围盒内的采用点可能不在地面或顶面的四边形内，还需要继续判断
    xy_poly = Polygon(xy)  # 多线程（map） + 偏函数：判断底面和顶面是否在这个四边形内
    with Pool(n_thread) as p:
        sel = p.map(functools.partial(pt_in_poly, xy_poly), floor_xyz[:, :2])

    # TODO print
    print(type(sel))  # <class 'list'>
    print(len(sel))  # 278780

    return floor_texture[sel], floor_xyz[sel], ceil_texture[sel], ceil_xyz[sel]


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', default='assert/output_preprocess/demo_aligned_rgb.png',
                        help='Image texture in equirectangular format')
    parser.add_argument('--layout', default='assert/output/demo_aligned_rgb_cor_id.txt',
                        help='Txt file containing layout corners (cor_id)')
    parser.add_argument('--camera_height', default=1.6, type=float,
                        help='Camera height in meter (not the viewer camera)')
    parser.add_argument('--ppm', default=120, type=int,
                        help='Points per meter')
    parser.add_argument('--point_size', default=0.0025, type=int,
                        help='Point size')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Opacity of the texture')
    parser.add_argument('--threads', default=10, type=int,
                        help='Number of threads to use')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling', action='store_true',
                        help='Skip rendering ceiling')
    args = parser.parse_args()

    # Reading source (texture img, cor_id txt)
    equirect_texture = np.array(Image.open(args.img)) / 255.0

    # TODO print
    print("equirect_texture.shape:", equirect_texture.shape)  # (512, 1024, 3)

    with open(args.layout) as f:
        cor_id = np.array([line.split() for line in f], np.float32)
        # TODO print
        print("cor_id:\n", cor_id)
        '''
[[110. 188.]
 [110. 331.]
 [384. 180.]
 [384. 355.]
 [647. 182.]
 [647. 340.]
 [896. 191.]
 [896. 327.]]
        '''

    # Convert cor_id to 3d xyz
    N = len(cor_id) // 2  # 3
    H, W = equirect_texture.shape[:2]  # (512, 1024)
    floor_z = -args.camera_height  # -1.6
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H)  # 获取四个底层的角点，转换成底层的四个 3D 点

    # TODO print
    print("cor_id[1::2]:\n", cor_id[1::2])  # 获取四个底层的角点
    '''
[[110. 331.]
 [384. 355.]
 [647. 340.]
 [896. 327.]]
    '''
    print("floor_xy:\n", floor_xy)
    '''
 [[-2.0091412  2.494616 ]
 [-1.6119697 -1.6218909]
 [ 2.0720608 -1.8896177]
 [ 2.4037855  2.4185803]]
    '''

    # TODO unknown
    c = np.sqrt((floor_xy ** 2).sum(1))  # 计算四个底层的角点的 x^2+y^2
    v = np_coory2v(cor_id[0::2, 1], H)  # 获取四个顶层的角点的 v
    ceil_z = (c * np.tan(v)).mean()  # 顶层的高为四个角点计算出来的均值

    # TODO print
    print("(floor_xy ** 2).shape:", (floor_xy ** 2).shape)  # (4, 2)
    print("(floor_xy ** 2).sum(1).shape:", (floor_xy ** 2).sum(1).shape)  # (4, 2) -> (4,)
    print("cor_id[0::2, 1]:\n", cor_id[0::2, 1])  # [188. 180. 182. 191.] -> 获取四个顶层的 y 坐标
    print("ceil_z:", ceil_z)  # 1.3332773

    # Warp each wall
    all_rgba, all_xyz = warp_walls(floor_xy, floor_z, ceil_z, H, W, args.ppm, args.alpha)

    # Warp floor and ceiling
    if not args.ignore_floor or not args.ignore_ceiling:
        fi, fp, ci, cp = warp_floor_ceiling(floor_xy, floor_z, ceil_z, H, W,
                                            ppm=args.ppm,
                                            alpha=args.alpha,
                                            n_thread=args.threads)

        if not args.ignore_floor:
            all_rgba.extend(fi)
            all_xyz.extend(fp)

        if not args.ignore_ceiling:
            all_rgba.extend(ci)
            all_xyz.extend(cp)

    # Launch point cloud viewer
    print('# of points:', len(all_rgba))
    all_xyz = np.array(all_xyz)
    all_rgb = np.array(all_rgba)[:, :3]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(all_xyz)
    pcd.colors = open3d.utility.Vector3dVector(all_rgb)
    open3d.visualization.draw_geometries([pcd])
