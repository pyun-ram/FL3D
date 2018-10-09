import glob
import numpy as np
from parse_xml import parseXML
from config import cfg
import random
import math
import cv2
def load_pc_from_pcd(pcd_path):
    """Load PointCloud data from pcd file."""
    # p = pcl.load(pcd_path)
    p = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    return p

def load_pc_from_bin(bin_path):
    """Load PointCloud data from pcd file."""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def read_calib_file(calib_path):
    """Read a calibration file."""
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]
    #
    P = np.array(lines[CAM]).reshape(3,4)
    P = np.concatenate( (  P, np.array( [[0,0,0,0]] )  ), 0  )
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
    Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3,:3] = np.array(lines[4][:9]).reshape(3,3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect

def proj_to_velo(calib_data):
    """Projection matrix to 3D axis for 3D Label"""
    rect = calib_data["R0_rect"].reshape(3, 3)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    inv_rect = np.linalg.inv(rect)
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)

def read_labels(label_path, label_type, calib_path=None, is_velo_cam=False, proj_velo=None):
    """Read labels from xml or txt file.
    Original Label value is shifted about 0.27m from object center.
    So need to revise the position of objects.
    """
    if label_type == "txt": #TODO
        places, size, rotates = read_label_from_txt(label_path)
        if places is None:
            return None, None, None
        rotates = np.pi / 2 - rotates
        dummy = np.zeros_like(places)
        dummy = places.copy()
        if calib_path:
            places = np.dot(dummy, proj_velo.transpose())[:, :3]
        else:
            places = dummy
        if is_velo_cam:
            places[:, 0] += 0.27

    elif label_type == "xml":
        bounding_boxes, size = read_label_from_xml(label_path)
        places = bounding_boxes[30]["place"]
        rotates = bounding_boxes[30]["rotate"][:, 2]
        size = bounding_boxes[30]["size"]

    return places, rotates, size

def get_boxcorners(places, rotates, size):
    """Create 8 corners of bounding box from bottom center."""
    corners = []
    for place, rotate, sz in zip(places, rotates, size):
        x, y, z = place
        h, w, l = sz
        if l > 10:
            continue

        corner = np.array([
            [x - l / 2., y - w / 2., z],
            [x + l / 2., y - w / 2., z],
            [x - l / 2., y + w / 2., z],
            [x - l / 2., y - w / 2., z + h],
            [x - l / 2., y + w / 2., z + h],
            [x + l / 2., y + w / 2., z],
            [x + l / 2., y - w / 2., z + h],
            [x + l / 2., y + w / 2., z + h],
        ])

        corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), -np.sin(rotate), 0],
            [np.sin(rotate), np.cos(rotate), 0],
            [0, 0, 1]
        ])

        a = np.dot(corner, rotate_matrix.transpose())
        a += np.array([x, y, z])
        corners.append(a)
    return np.array(corners)

def filter_camera_angle(places):
    """Filter camera angles for KiTTI Datasets"""
    bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    # bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[bool_in]

def raw_to_voxel(pc, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert PointCloud2 to Voxel"""
    logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
    logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
    logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])
    pc = pc[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]
    pc =((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
    voxel = np.zeros((int((x[1] - x[0]) / resolution), int((y[1] - y[0]) / resolution), int(round((z[1]-z[0]) / resolution))))
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    return voxel

def create_label(places, size, corners, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4, min_value=np.array([0., -50., -4.5])):
    """Create training Labels which satisfy the range of experiment"""
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] + size[:, 0]/2. < z[1]), (places[:, 2] + size[:, 0]/2. >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))

    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2. # Move bottom to center
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)

    train_corners = corners[xyz_logical].copy()
    anchor_center = sphere_to_center(sphere_center, resolution=resolution, scale=scale, min_value=min_value) #sphere to center
    for index, (corner, center) in enumerate(zip(corners[xyz_logical], anchor_center)):
        train_corners[index] = corner - center
    return sphere_center, train_corners

def create_objectness_label(sphere_center, resolution=0.5, x=90, y=100, z=10, scale=4):
    """Create Objectness label"""
    obj_maps = np.zeros((int(x / (resolution * scale)), int(y / (resolution * scale)), int(round(z / (resolution * scale)))))
    obj_maps[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = 1
    return obj_maps

def corner_to_voxel(voxel_shape, corners, sphere_center, scale=4):
    """Create final regression label from corner"""
    # print("YUN: {}".format((int(voxel_shape[0] / scale), int(voxel_shape[1] / scale), int(voxel_shape[2] / scale), int(24))))
    corner_voxel = np.zeros((int(voxel_shape[0] / scale), int(voxel_shape[1] / scale), int(voxel_shape[2] / scale), int(24)))
    corner_voxel[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = corners
    return corner_voxel

def read_label_from_txt(label_path):
    """Read label from txt file."""
    text = np.fromfile(label_path)
    bounding_box = []
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            if not label:
                continue
            label = label.split(" ")
            if (label[0] == "DontCare"):
                continue

            if label[0] == ("Car" or "Van"): #  or "Truck"
                bounding_box.append(label[8:15])

    if bounding_box:
        data = np.array(bounding_box, dtype=np.float32)
        return data[:, 3:6], data[:, :3], data[:, 6]
    else:
        return None, None, None

def sphere_to_center(p_sphere, resolution=0.5, scale=4, min_value=np.array([0., -50., -4.5])):
    """from sphere center to label center"""
    center = p_sphere * (resolution*scale) + min_value
    return center

def read_label_from_xml(label_path):
    """Read label from xml file.

    # Returns:
        label_dic (dictionary): labels for one sequence.
        size (list): Bounding Box Size. [l, w. h]?
    """
    labels = parseXML(label_path)
    label_dic = {}
    for label in labels:
        first_frame = label.firstFrame
        nframes = label.nFrames
        size = label.size
        obj_type = label.objectType
        for index, place, rotate in zip(range(first_frame, first_frame+nframes), label.trans, label.rots):
            if index in label_dic.keys():
                label_dic[index]["place"] = np.vstack((label_dic[index]["place"], place))
                label_dic[index]["size"] = np.vstack((label_dic[index]["size"], np.array(size)))
                label_dic[index]["rotate"] = np.vstack((label_dic[index]["rotate"], rotate))
            else:
                label_dic[index] = {}
                label_dic[index]["place"] = place
                label_dic[index]["rotate"] = rotate
                label_dic[index]["size"] = np.array(size)
    return label_dic, size

def lidar_to_camera_point(points, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 3) -> (N, 3)
    N = points.shape[0] # BUG
    points = np.hstack([points, np.ones((N, 1))]).T
    
    
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.MATRIX_R_RECT_0)

    points = np.matmul(T_VELO_2_CAM, points)
    points = np.matmul(R_RECT_0, points).T
    points = points[:, 0:3]
    return points.reshape(-1, 3)

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle

def camera_to_lidar(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.MATRIX_R_RECT_0)

    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(R_RECT_0), p)
    p = np.matmul(np.linalg.inv(T_VELO_2_CAM), p)
    p = p[0:3]
    return tuple(p)

def camera_to_lidar_box(boxes, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -ry - np.pi / 2
        rz = angle_in_limit(rz)
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 7)

def corner_to_center_box3d(boxes_corner, coordinate='camera', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry/z
    boxes_corner_cp = np.copy(boxes_corner)
    if coordinate == 'lidar':
        for idx in range(len(boxes_corner_cp)):
            boxes_corner_cp[idx] = lidar_to_camera_point(boxes_corner_cp[idx], T_VELO_2_CAM, R_RECT_0)
    ret = []
    # 0[- l / 2.,- w / 2., z],
    # 1[+ l / 2.,- w / 2., z],
    # 2[- l / 2.,+ w / 2., z],
    # 3[- l / 2.,- w / 2., z + h],
    # 4[- l / 2.,+ w / 2., z + h],
    # 5[+ l / 2.,+ w / 2., z],
    # 6[+ l / 2.,- w / 2., z + h],
    # 7[+ l / 2.,+ w / 2., z + h],
    #    0        1      2      3        4      5      6      7
    # [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
    # [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
    # [0    , 0     , 0     , 0    , h    , h     , h     , h    ]])
    for roi in boxes_corner_cp:
        if cfg.CORNER2CENTER_AVG:  # average version
            roi = np.array(roi)
            tmp = np.zeros_like(roi)
            tmp[0,:] = roi[2,:]
            tmp[1,:] = roi[0,:]
            tmp[2,:] = roi[1,:]
            tmp[3,:] = roi[5,:]
            tmp[4,:] = roi[4,:]
            tmp[5,:] = roi[3,:]
            tmp[6,:] = roi[6,:]
            tmp[7,:] = roi[7,:]
            roi = tmp
            h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
            w = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            ) / 4
            l = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            ) / 4
            x = np.sum(roi[:, 0], axis=0)/ 8.0
            y = np.sum(roi[0:4, 1], axis=0)/ 4.0
            z = np.sum(roi[:, 2], axis=0)/ 8.0
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8.0 # BUG HERE
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        else:  # max version
            h = max(abs(roi[:4, 1] - roi[4:, 1]))
            w = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            )
            l = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            )
            x = np.sum(roi[:, 0], axis=0)/ 8
            y = np.sum(roi[0:4, 1], axis=0)/ 4
            z = np.sum(roi[:, 2], axis=0)/ 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        ret.append([x, y, z, h, w, l, ry])
    if coordinate == 'lidar':
        ret = camera_to_lidar_box(np.array(ret), T_VELO_2_CAM, R_RECT_0)

    return np.array(ret)

def center_to_corner_box3d(boxes_center, coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center, T_VELO_2_CAM, R_RECT_0)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    if coordinate == 'camera':
        for idx in range(len(ret)):
            ret[idx] = lidar_to_camera_point(ret[idx], T_VELO_2_CAM, R_RECT_0)

    return ret

def lidar_box3d_to_camera_box(boxes3d, cal_projection=False, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)

    lidar_boxes3d_corner = center_to_corner_box3d(boxes3d, coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    if type(P2) == type(None):
        P2 = np.array(cfg.MATRIX_P2)

    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d, T_VELO_2_CAM, R_RECT_0)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = int(np.min(points[:, 0]))
        maxx = int(np.max(points[:, 0]))
        miny = int(np.min(points[:, 1]))
        maxy = int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d

def lidar_to_camera(x, y, z, T_VELO_2_CAM=None, R_RECT_0=None):
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(cfg.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(cfg.MATRIX_R_RECT_0)

    p = np.array([x, y, z, 1])
    p = np.matmul(T_VELO_2_CAM, p)
    p = np.matmul(R_RECT_0, p)
    p = p[0:3]
    return tuple(p)

def lidar_to_camera_box(boxes, T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -rz - np.pi / 2
        ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret).reshape(-1, 7)

def box3d_to_label(batch_box3d, batch_cls, batch_score=[], coordinate='camera', P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   (N, N', 7) x y z h w l r
    #   (N, N')
    #   cls: (N, N') 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate(input): 'camera' or 'lidar'
    # Output:
    #   label: (N, N') N batches and N lines
    batch_label = []
    if batch_score:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(15)]) + '\n'
        for boxes, scores, clses in zip(batch_box3d, batch_score, batch_cls):
            label = []
            for box, score, cls in zip(boxes, scores, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0), cal_projection=False, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection=False, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(
                    cls, 0, 0, 0, *box2d, *box3d, float(score)))
            batch_label.append(label)
    else:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(14)]) + '\n'
        for boxes, clses in zip(batch_box3d, batch_cls):
            label = []
            for box, cls in zip(boxes, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0), cal_projection=False,  P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection=False, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(cls, 0, 0, 0, *box2d, *box3d))
            batch_label.append(label)

    return np.array(batch_label)

def lidar_to_bird_view_img(lidar, factor=1):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros(
        (cfg.VOXEL_SHAPE[1]*factor, cfg.VOXEL_SHAPE[0]*factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.X[0] < x < cfg.X[1] and cfg.Y[0] < y < cfg.Y[1]:
            x, y = int((x - cfg.X[0]) / cfg.RESOLUTION * factor), int((y - cfg.Y[0]) / cfg.RESOLUTION * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) * 5 * factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)
    return birdview

def lidar_to_bird_view(x, y, factor=1):
    # using the cfg.INPUT_XXX
    a = (x - cfg.X[0]) / cfg.RESOLUTION * factor
    b = (y - cfg.Y[0]) / cfg.RESOLUTION * factor
    a = np.clip(a, a_max=(cfg.X[1] - cfg.X[0]) / cfg.RESOLUTION * factor, a_min=0)
    b = np.clip(b, a_max=(cfg.Y[1] - cfg.Y[0]) / cfg.RESOLUTION * factor, a_min=0)
    return a, b

def draw_lidar_box3d_on_birdview(birdview, boxes3d, gt_boxes3d=np.array([]),
                                 color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1, factor=1, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    corner_boxes3d = center_to_corner_box3d(boxes3d, coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate='lidar', T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)

    # draw detections
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

def label_to_gt_box3d(labels, cls='Car', coordinate='camera', T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    boxes3d = []
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    elif cls == 'Cyclist':
        acc_cls = ['Cyclist']
    else: # all
        acc_cls = []

    for label in labels:
        boxes3d_a_label = []
        for line in label:
            ret = line.split()
            if ret[0] in acc_cls or acc_cls == []:
                h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]
                box3d = np.array([x, y, z, h, w, l, r])
                boxes3d_a_label.append(box3d)
        if coordinate == 'lidar':
            boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label), T_VELO_2_CAM, R_RECT_0)

        boxes3d.append(np.array(boxes3d_a_label).reshape(-1, 7))
    return boxes3d

def center_to_corner_box2d(boxes_center, coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes3d_corner = center_to_corner_box3d(
        boxes3d_center, coordinate=coordinate, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)

    return boxes3d_corner[:, 0:4, 0:2]


def center_to_corner_box3d(boxes_center, coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center, T_VELO_2_CAM, R_RECT_0)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    if coordinate == 'camera':
        for idx in range(len(ret)):
            ret[idx] = lidar_to_camera_point(ret[idx], T_VELO_2_CAM, R_RECT_0)

    return ret

def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

    return standup_boxes2d

def get_pts(place, rotate, size, pc):
    x, y, z = place
    h, w, l = size
    min_max_anchors = np.array([
                                [x - l/2, y - w/2, z],
                                [x + l/2, y + w/2, z + h]
                               ])

    # min_max_anchors -= np.array([x, y, z])
    rotate_matrix = np.array([
        [np.cos(-rotate), -np.sin(-rotate), 0],
        [np.sin(-rotate), np.cos(-rotate), 0],
        [0, 0, 1]
    ])
    # min_max_anchors = np.dot(min_max_anchors, rotate_matrix.transpose())
    # min_max_anchors += np.array([x, y, z])
    
    tmp_pc = pc[:,:3].copy()
    tmp_pc -= np.array([x, y, z])
    tmp_pc = np.dot(tmp_pc, rotate_matrix.transpose())
    tmp_pc += np.array([x, y, z])

    bool_x = np.logical_and(tmp_pc[:,0]>= min_max_anchors[0,0], tmp_pc[:,0]<= min_max_anchors[1,0])
    bool_y = np.logical_and(tmp_pc[:,1]>= min_max_anchors[0,1], tmp_pc[:,1]<= min_max_anchors[1,1])
    bool_z = np.logical_and(tmp_pc[:,2]>= min_max_anchors[0,2], tmp_pc[:,2]<= min_max_anchors[1,2])
    bool_xy = np.logical_and(bool_x, bool_y)
    bool_xyz = np.logical_and(bool_xy, bool_z)
    return pc[bool_xyz]

def get_pts_standup(place, rotate, size, pc):
    x, y, z = place
    h, w, l = size
    min_h = np.min(pc[:,2])
    max_h = np.max(pc[:,2])
    min_max_anchors = np.array([
                                [x - l/2, y - w/2, min_h],
                                [x + l/2, y + w/2, max_h]
                               ])

    # min_max_anchors -= np.array([x, y, z])
    rotate_matrix = np.array([
        [np.cos(-rotate), -np.sin(-rotate), 0],
        [np.sin(-rotate), np.cos(-rotate), 0],
        [0, 0, 1]
    ])
    # min_max_anchors = np.dot(min_max_anchors, rotate_matrix.transpose())
    # min_max_anchors += np.array([x, y, z])
    
    tmp_pc = pc[:,:3].copy()
    tmp_pc -= np.array([x, y, z])
    tmp_pc = np.dot(tmp_pc, rotate_matrix.transpose())
    tmp_pc += np.array([x, y, z])

    bool_x = np.logical_and(tmp_pc[:,0]>= min_max_anchors[0,0], tmp_pc[:,0]<= min_max_anchors[1,0])
    bool_y = np.logical_and(tmp_pc[:,1]>= min_max_anchors[0,1], tmp_pc[:,1]<= min_max_anchors[1,1])
    bool_xy = np.logical_and(bool_x, bool_y)
    return pc[bool_xy]

def remove_empty_bbox(places, rotates, sizes, pc, thresh=10):
    if type(places) is type(None):
        return None, None, None
    out_places = []
    out_rotates = []
    out_sizes = []
    out_pts = []
    pc_test = []
    for place, rotate, size in zip(places, rotates, sizes):
        pts = get_pts(place, rotate, size, pc)
        out_pts.append(pts.shape[0])
        if pts.shape[0] >= thresh:
            out_places.append(place)
            out_sizes.append(size)
            out_rotates.append(rotate)
    if out_places:
        out_places = np.concatenate(out_places).reshape(-1, 3)
        out_sizes = np.concatenate(out_sizes).reshape(-1, 3)
        out_rotates = np.asarray(out_rotates)
        return out_places, out_sizes, out_rotates
    else:
        return None, None, None

def draw_lidar_box3d_on_image(img, boxes3d, gt_boxes3d=np.array([]),
                              color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1, P2 = None, T_VELO_2_CAM=None, R_RECT_0=None):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=True, P2=P2, T_VELO_2_CAM=T_VELO_2_CAM, R_RECT_0=R_RECT_0)

    # draw projections
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)
    # draw gt projections
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

def lidar_generator(batch_num, velodyne_path, calib_path, label_path=None, resolution=0.2, dataformat="pcd", label_type="txt", is_velo_cam=False, shuffle=False, \
                        scale=4, x=(0, 80), y=(-40, 40), z=(-2.5, 1.5)):                    
    velodynes_path = glob.glob(velodyne_path)
    labels_path = glob.glob(label_path)
    calibs_path = [path[:-18] + "calib" + path[-11:] for path in labels_path]
    velodynes_path.sort()
    labels_path.sort()
    calibs_path.sort()
    if shuffle:
        merge_path = list(zip(velodynes_path, labels_path, calibs_path))
        random.shuffle(merge_path)
        velodynes_path, labels_path, calibs_path = zip(*merge_path)

    iter_num = len(velodynes_path) // batch_num

    for itn in range(iter_num):
        batch_voxel = []
        batch_g_map = []
        batch_g_cord = []

        for velodynes, labels, calibs in zip(velodynes_path[itn*batch_num:(itn+1)*batch_num], \
            labels_path[itn*batch_num:(itn+1)*batch_num], calibs_path[itn*batch_num:(itn+1)*batch_num]):
            p = []
            pc = None
            bounding_boxes = None
            places = None
            rotates = None
            size = None
            proj_velo = None

            if dataformat == "bin":
                pc = load_pc_from_bin(velodynes)
            elif dataformat == "pcd":
                pc = load_pc_from_pcd(velodynes)

            if calib_path:
                calib = read_calib_file(calibs)
                proj_velo = proj_to_velo(calib)[:, :3]

            if label_path:
                # places, rotates, size = read_labels(labels, label_type, calib_path=calib_path, is_velo_cam=is_velo_cam, proj_velo=proj_velo)
                gt_box3d = label_to_gt_box3d([ [line for line in open(labels, 'r').readlines()]], cls='Car', coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None)[0]
                print("lidar_generator: gt_box3d.shape is {}".format(gt_box3d.shape))
                if gt_box3d.size != 0:
                    places = gt_box3d[:,:3]
                    size = gt_box3d[:,3:6]
                    rotates = gt_box3d[:, -1]
                else:
                    places = None
                    size = None
                    rotates = None
            pc = filter_camera_angle(pc)
            places, size, rotates = remove_empty_bbox(places, rotates, size, pc, thresh=10)
            if places is None:
                voxel =  raw_to_voxel(pc, resolution=resolution, x=x, y=y, z=z)
                g_map = np.zeros((int(voxel.shape[0] / scale), int(voxel.shape[1] / scale), int(voxel.shape[2] / scale)))
                g_cord = np.zeros((int(voxel.shape[0] / scale), int(voxel.shape[1] / scale), int(voxel.shape[2] / scale), int(24)))
            else:                 
                corners = get_boxcorners(places, rotates, size)
                voxel =  raw_to_voxel(pc, resolution=resolution, x=x, y=y, z=z)
                center_sphere, corner_label = create_label(places, size, corners, resolution=resolution, x=x, y=y, z=z, \
                    scale=scale, min_value=np.array([x[0], y[0], z[0]]))
                if not center_sphere.shape[0]:
                    continue
                g_map = create_objectness_label(center_sphere, resolution=resolution, x=(x[1] - x[0]), y=(y[1] - y[0]), z=(z[1] - z[0]), scale=scale)
                g_cord = corner_label.reshape(corner_label.shape[0], -1)
                g_cord = corner_to_voxel(voxel.shape, g_cord, center_sphere, scale=scale)

            batch_voxel.append(voxel)
            batch_g_map.append(g_map)
            batch_g_cord.append(g_cord)
        # print("lidar_generator: batch_voxel shape is {}".format( np.array(batch_voxel, dtype=np.float32).shape))
        # print("lidar_generator: batch_g_map shape is {}".format(np.array(batch_g_map, dtype=np.float32).shape))
        # print("lidar_generator: batch_g_cord shape is {}".format( np.array(batch_g_cord, dtype=np.float32).shape))
        if np.array(batch_voxel, dtype=np.float32).shape[0] == 0:
            continue
        yield np.array(batch_voxel, dtype=np.float32)[:, :, :, :, np.newaxis], np.array(batch_g_map, dtype=np.float32), np.array(batch_g_cord, dtype=np.float32)