'''
 Peng YUN Created on Sat Aug 04 2018

 Copyright (c) 2018 RAM-LAB
'''

import os
import logging
import cv2
from shutil import copyfile
import tensorflow as tf
from config import cfg
from utils.kitti_loader import iterate_data
from utils import *
from model import RPN3D

tf.set_random_seed(123)

def gpu_config():
    """
    config gpu settings.
    """
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

def main():
    # load config
    eval_dataset_dir = os.path.join(cfg.DATA_DIR, "validation")
    save_model_dir = os.path.join("./save_model", cfg.TAG)
    log_dir = os.path.join("./log", cfg.TAG)
    pred_dir = os.path.join("./predicts/" , cfg.TAG)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(os.path.join(pred_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(pred_dir, "vis"), exist_ok=True)
    config = gpu_config()


    # config logging
    logging.basicConfig(filename='./log/'+cfg.TAG+'/test.log', level=logging.ERROR)   
    
    print("tag: {}".format(cfg.TAG))
    logging.critical("tag: {}".format(cfg.TAG))
    with tf.Session(config=config) as sess:
        # load model
        model = RPN3D(cls=cfg.DETECT_OBJ, single_batch_size=cfg.SINGLE_BATCH_SIZE, is_training=False, 
                learning_rate=cfg.LR, max_gradient_norm=5.0, alpha=cfg.ALPHA,
                beta=cfg.BETA, gamma=cfg.GAMMA, avail_gpus=cfg.GPU_AVAILABLE.split(','))
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10, 
                    pad_step_number=True, keep_checkpoint_every_n_hours=1.0)   
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # param init/restore
        if not cfg.LOAD_CHECKPT == None :
            print("Reading model parameters from {}, {}".format(save_model_dir, cfg.LOAD_CHECKPT))
            logging.critical("Reading model parameters from {}, {}".format(save_model_dir, cfg.LOAD_CHECKPT))
            saver.restore(sess, os.path.join(save_model_dir, cfg.LOAD_CHECKPT))
        elif tf.train.get_checkpoint_state(save_model_dir):
            print("Reading model parameters from %s" % save_model_dir)
            logging.critical("Reading model parameters from %s" % save_model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
        else:
            print("No saved weight exist for %s" % cfg.TAG)
            logging.critical("No saved weight exist for %s" % cfg.TAG)
            exit(-1)
        
        # load data
        data_generator_test = iterate_data(eval_dataset_dir, shuffle=False, aug=False, is_testset=False, 
            batch_size=cfg.SINGLE_BATCH_SIZE * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT)
        for batch in data_generator_test:
            tags, results, front_images, bird_views, heatmaps = model.predict_step(sess, batch, summary=False, vis=True)
            # ret: A, B
            # A: (N) tag
            # B: (N, N') (class, x, y, z, h, w, l, rz, score)
            for tag, result in zip(tags, results):
                of_path = os.path.join(pred_dir, 'data', tag + '.txt')
                with open(of_path, 'w+') as f:
                    labels = box3d_to_label([result[:, 1:8]], [result[:, 0]], [result[:, -1]], coordinate='lidar')[0]
                    for line in labels:
                        f.write(line)
                    print('write out {} objects to {}'.format(len(labels), tag))
                    logging.critical('write out {} objects to {}'.format(len(labels), tag))
                for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps):
                    front_img_path = os.path.join( pred_dir, 'vis', tag + '_front.jpg'  )
                    bird_view_path = os.path.join( pred_dir, 'vis', tag + '_bv.jpg'  )
                    heatmap_path = os.path.join( pred_dir, 'vis', tag + '_heatmap.jpg'  )
                    cv2.imwrite( front_img_path, front_image )
                    cv2.imwrite( bird_view_path, bird_view )
                    cv2.imwrite( heatmap_path, heatmap )
        print('{} Testing Done! Starting Evaluation'.format(cfg.TAG))
        logging.critical('{} Testing Done! Starting Evaluation'.format(cfg.TAG))            
        # ./kitti_eval/evaluate_object_3d_offline /usr/app/TuneDataKitti/validation/label_2/ ./predicts/$TAG/ > ./predicts/$TAG/cmd.log
        cmd = "./kitti_eval/evaluate_object_3d_offline" + " " \
            + os.path.join(eval_dataset_dir, "label_2/") + " " \
            + pred_dir + " " \
            + ">" + " " \
            + os.path.join(pred_dir, "cmd.log")
        os.system(cmd)
        print('{} Evaluation Done!'.format(cfg.TAG))
        logging.critical('{} Evaluation Done!'.format(cfg.TAG))  

if __name__ == '__main__':
    main()