#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : data_aug.py
# Purpose :
# Creation Date : 21-12-2017
# Last Modified : Fri 19 Jan 2018 01:06:35 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import cv2
import os
import multiprocessing as mp
import argparse
import glob

from utils.utils import *
from utils.preprocess import *

object_dir = './data/object'


def aug_data(tag, object_dir):
    np.random.seed()
    rgb = cv2.resize(cv2.imread(os.path.join(object_dir,
                                             'image_2', tag + '.png')), (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    #rgb = cv2.imread( os.path.join(object_dir,'image_2', tag + '.png')  )
    lidar = np.fromfile(os.path.join(object_dir,
                                     'velodyne', tag + '.bin'), dtype=np.float32).reshape(-1, 4)
    label = np.array([line for line in open(os.path.join(
        object_dir, 'label_2', tag + '.txt'), 'r').readlines()])  # (N')
    cls = np.array([line.split()[0] for line in label])  # (N')
    gt_box3d = label_to_gt_box3d(np.array(label)[np.newaxis, :], cls='', coordinate='camera')[0]  # (N', 7) x, y, z, h, w, l, r
    choice = np.random.randint(0, 8)
    if choice >= 7:
        # disable this augmention here. current implementation will decrease the performances
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_corner_gt_box3d = center_to_corner_box3d(
            lidar_center_gt_box3d, coordinate='lidar')
        for idx in range(len(lidar_corner_gt_box3d)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal(loc=0.0, scale=1)
                t_y = np.random.normal(loc=0.0, scale=1)
                t_z = np.random.normal(loc=0.0, scale=1)
                box_corner = lidar_corner_gt_box3d[idx]
                rotation = lidar_center_gt_box3d[idx,-1]
                vx = lidar_center_gt_box3d[idx,0]
                vy = lidar_center_gt_box3d[idx,1]
                vz = lidar_center_gt_box3d[idx,2]
                h  = lidar_center_gt_box3d[idx,3] 
                w  = lidar_center_gt_box3d[idx,4]
                l  = lidar_center_gt_box3d[idx,5]
                
                # check collision
                tmp = box_transform(
                    lidar_center_gt_box3d[[idx]], -vx, -vy, -vz, t_rz, 'lidar')
                tmp = box_transform(
                    tmp, vx+t_x, vy+t_y, vz+t_z, 0, 'lidar')
                is_collision = False
                for idy in range(idx):
                    x1, y1, w1, l1, r1 = tmp[0][[0, 1, 4, 5, 6]]
                    x2, y2, w2, l2, r2 = lidar_center_gt_box3d[idy][[
                        0, 1, 4, 5, 6]]
                    iou = cal_iou2d(np.array([x1, y1, w1, l1, r1], dtype=np.float32),
                                    np.array([x2, y2, w2, l2, r2], dtype=np.float32))
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = point_transform(box_corner, 0,0,0, rz=-rotation)
                lidar[:,0:3] = point_transform(lidar[:,0:3], 0,0,0, rz=-rotation)
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[:,0:3] = point_transform(lidar[:,0:3], 0,0,0, rz=rotation)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], -vx, -vy, -vz, rz=t_rz)
                # lidar[bound_box, 0:3] = point_transform(
                #     lidar[bound_box, 0:3], 0,0,0, rz=-rotation)                    
                # lidar[bound_box, 0:3] = point_transform(
                #     lidar[bound_box, 0:3], 0,0,0, rz=rotation+t_rz)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], vx+t_x, vy+t_y, vz+t_z, rz=0)        

                # lidar_center_gt_box3d[idx] = box_transform(
                #     lidar_center_gt_box3d[[idx]], -vx, -vy, -vz, t_rz, 'lidar')
                # # lidar_center_gt_box3d[idx] = box_transform(
                # #     lidar_center_gt_box3d[[idx]], 0,0,0, -rotation, 'lidar')                    
                # # lidar_center_gt_box3d[idx] = box_transform(
                # #     lidar_center_gt_box3d[[idx]], 0, 0, 0, rotation+t_rz, 'lidar')                    
                # lidar_center_gt_box3d[idx] = box_transform(
                #     lidar_center_gt_box3d[[idx]], vx+t_x, vy+t_y, vz+t_z, 0, 'lidar')
                lidar_center_gt_box3d[idx] = tmp
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_1_{}'.format(
            tag, np.random.randint(1, 1024))
    elif choice < 7 and choice >= 4:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, 0, 0, 0, r=angle, coordinate='lidar')
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_2_{:.4f}'.format(tag, angle).replace('.', '_')
    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_') 
    label = box3d_to_label(gt_box3d[np.newaxis, ...], cls[np.newaxis, ...], coordinate='camera')[0]  # (N')
    voxel_dict = process_pointcloud(lidar)
    return newtag, rgb, lidar, voxel_dict, label 


def worker(tag):
    new_tag, rgb, lidar, voxel_dict, label = aug_data(tag)
    output_path = os.path.join(object_dir, 'training_aug')

    cv2.imwrite(os.path.join(output_path, 'image_2', newtag + '.png'), rgb)
    lidar.reshape(-1).tofile(os.path.join(output_path,
                                          'velodyne', newtag + '.bin'))
    np.savez_compressed(os.path.join(
        output_path, 'voxel' if cfg.DETECT_OBJ == 'Car' else 'voxel_ped', newtag), **voxel_dict)
    with open(os.path.join(output_path, 'label_2', newtag + '.txt'), 'w+') as f:
        for line in label:
            f.write(line)
    print(newtag)


def main():
    fl = glob.glob(os.path.join(object_dir, 'training', 'calib', '*.txt'))
    candidate = [f.split('/')[-1].split('.')[0] for f in fl]
    tags = []
    for _ in range(args.aug_amount):
        tags.append(candidate[np.random.randint(0, len(candidate))])
    print('generate {} tags'.format(len(tags)))
    pool = mp.Pool(args.num_workers)
    pool.map(worker, tags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--aug-amount', type=int, nargs='?', default=1000)
    parser.add_argument('-n', '--num-workers', type=int, nargs='?', default=10)
    args = parser.parse_args()

    main()
