'''
 Peng YUN Created on Wed Jun 21 2018

 Copyright (c) 2018 RAM-LAB
'''
import logging
import os
import tensorflow as tf
import cv2
from shutil import copyfile
from config import cfg
from utils import *
from model.model import FCN3D, DenseNet
tf.set_random_seed(123)

def gpu_config():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
        visible_device_list=cfg.GPU_AVAILABLE,
        allow_growth=True)
    config = tf.ConfigProto(
        gpu_options=gpu_options,
        device_count={
            "GPU": cfg.GPU_USE_COUNT,
        },
        allow_soft_placement=True,
    )
    return config

def test_data_loader(velodyne_regpath, img_regpath, label_regpath):
    velodyne_paths = glob.glob(velodyne_regpath)
    img_paths = glob.glob(img_regpath)
    label_paths = glob.glob(label_regpath)
    velodyne_paths.sort()
    img_paths.sort()
    label_paths.sort()
    for velodyne_path, img_path, label_path in zip(velodyne_paths, img_paths, label_paths):
        tag = velodyne_path.split('/')[-1]
        tag = tag.split('.')[0]
        yield tag, load_pc_from_bin(velodyne_path), cv2.imread(img_path), [line for line in open(label_path, 'r').readlines()]

def main():
    # config
    eval_dir = os.path.join(cfg.DATA_DIR, "validation")
    log_dir = os.path.join("./log/" , cfg.TAG)
    pred_dir = os.path.join("./predicts/" , cfg.TAG)
    save_model_dir = os.path.join("./save_model/" , cfg.TAG)

    eval_velodyne_path = os.path.join(eval_dir, "velodyne/*.bin")
    eval_img_path = os.path.join(eval_dir, "image_2/*.png")
    eval_label_path = os.path.join(eval_dir, "label_2/*.txt")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(os.path.join(pred_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(pred_dir, "vis"), exist_ok=True)
    logging.basicConfig(filename='./log/'+cfg.TAG+'/test.log', level=logging.ERROR)    # config logging
    
    print("tag: {}".format(cfg.TAG))
    print("eval_threshold: {}".format(cfg.EVAL_THRESHOLD))
    logging.critical("tag: {}".format(cfg.TAG))
    logging.critical("eval_threshold: {}".format(cfg.EVAL_THRESHOLD))
    

    config = gpu_config()   # GPU configuration
    with tf.Session(config=config) as sess:
        # load Model
        if cfg.MODEL=="FCN":
            model = FCN3D(sess, scale=cfg.SCALE, voxel_shape=cfg.VOXEL_SHAPE, is_training=False, alpha=cfg.ALPHA, beta=cfg.BETA, eta=cfg.ETA, gamma=cfg.GAMMA)
        elif cfg.MODEL=="DENSENET":
            model = DenseNet(sess, scale=cfg.SCALE, voxel_shape=cfg.VOXEL_SHAPE, is_training=False, alpha=cfg.ALPHA, beta=cfg.BETA, eta=cfg.ETA)
        saver = tf.train.Saver()

        # param init/restore
        print("Reading model parameters from %s" % save_model_dir)
        logging.critical("Reading model parameters from %s" % save_model_dir)
        # if cfg.TAG == '3dfcn-baseline':
        #     saver.restore(sess, os.path.join(save_model_dir, "velodyne_025_deconv_norm_valid40.ckpt"))    
        # else:
        #     saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
        if cfg.LOAD_CHECKPT == None:
            saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
        else :
            saver.restore(sess, os.path.join(save_model_dir, cfg.LOAD_CHECKPT))
        print("checkpoint: {}".format(cfg.LOAD_CHECKPT))
        logging.critical("checkpoint: {}".format(cfg.LOAD_CHECKPT))
        model.set_nms(max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)
        data_loader_eval = test_data_loader(eval_velodyne_path, eval_img_path, eval_label_path)
        for tag, pc, img, label in data_loader_eval:
            print(tag)
            gt_box3d = label_to_gt_box3d([label], cls='Car', coordinate='lidar', T_VELO_2_CAM=None, R_RECT_0=None)[0]
            pc = filter_camera_angle(pc)
            voxel =  raw_to_voxel(pc, resolution=cfg.RESOLUTION, x=cfg.X, y=cfg.Y, z=cfg.Z)
            voxel_x = voxel.reshape(1, voxel.shape[0], voxel.shape[1], voxel.shape[2], 1)   
                  
            objectness, cordinate, y_pred = model.evaluate(sess, voxel_x)
            
            index = np.where(y_pred >= cfg.EVAL_THRESHOLD)
            centers = np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()
            centers = sphere_to_center(centers, resolution=cfg.RESOLUTION, \
                scale=cfg.SCALE, min_value=np.array([cfg.X[0], cfg.Y[0], cfg.Z[0]]))
            corners = cordinate[index].reshape(-1, 8, 3) + centers[:, np.newaxis]
            corners += np.array([0.1,0,0])
            scores = y_pred[index].reshape(-1, 1)

            # NMS   
            box3d = corner_to_center_box3d(corners, coordinate='lidar')
            if cfg.ENABLE_NMS:
                box2d = box3d[:, [0, 1, 4, 5, 6]]
                box2d_standup = corner_to_standup_box2d(
                    center_to_corner_box2d(box2d, coordinate='lidar'))
                ind = sess.run(model.box2d_ind_after_nms, {
                    model.boxes2d: box2d_standup,
                    model.boxes2d_scores: scores.reshape(-1)
                })                
                box3d = box3d[ind]
                scores = scores[ind]
            result = np.concatenate([np.tile(cfg.CLS, len(box3d))[:, np.newaxis], box3d, scores], axis=-1)
            
            of_path = os.path.join(pred_dir, 'data', tag + '.txt')
            with open(of_path, 'w+') as f:
                labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                for line in labels:
                    f.write(line)
                print('write out {} objects to {}'.format(len(labels), tag))
                logging.critical('write out {} objects to {}'.format(len(labels), tag))


            bird_view = lidar_to_bird_view_img(pc, factor=cfg.BV_LOG_FACTOR)
            front_view_image = draw_lidar_box3d_on_image(img, box3d, gt_box3d)
            bird_view = draw_lidar_box3d_on_birdview(bird_view, box3d, gt_box3d, factor=cfg.BV_LOG_FACTOR, thickness=int(cfg.BV_LOG_FACTOR/2))
            front_view_path = os.path.join(pred_dir, 'vis', tag + '_fv.jpg')
            bird_view_path = os.path.join(pred_dir, 'vis', tag + '_bv.jpg')
            cv2.imwrite( front_view_path, front_view_image )
            cv2.imwrite( bird_view_path, bird_view )
        print('{} Testing Done! Starting Evaluation!'.format(cfg.TAG))
        logging.critical('{} Testing Done!  Starting Evaluation!'.format(cfg.TAG))
        # ./kitti_eval/evaluate_object_3d_offline /usr/app/TuneDataKitti/validation/label_2/ ./predicts/$TAG/ > ./predicts/$TAG/cmd.log
        cmd = "./kitti_eval/evaluate_object_3d_offline" + " " \
            + os.path.join(eval_dir, "label_2/") + " " \
            + pred_dir + " " \
            + ">" + " " \
            + os.path.join(pred_dir, "cmd.log")
        os.system(cmd)
        print('{} Evaluation Done!'.format(cfg.TAG))
        logging.critical('{} Evaluation Done!'.format(cfg.TAG))

            
if __name__ == '__main__':
    main()
