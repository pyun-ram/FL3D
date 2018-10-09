'''
 Peng YUN Created on Wed Jun 20 2018

 Copyright (c) 2018 RAM-LAB
'''
import logging
import os
import tensorflow as tf

from shutil import copyfile
from config import cfg
from utils import *
from model.model import *
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

def data_loader(velodyne_path, label_path, calib_path, is_velo_cam, shuffle):
    """
    load data
    """
    return lidar_generator(cfg.BATCH_NUM, velodyne_path=velodyne_path, label_path=label_path, calib_path=calib_path, \
                resolution=cfg.RESOLUTION, dataformat=cfg.DATA_FORMAT, label_type=cfg.LABEL_TYPE, is_velo_cam=is_velo_cam, shuffle=shuffle, \
                scale=cfg.SCALE, x=cfg.X, y=cfg.Y, z=cfg.Z)

def output_log(epoch, step, pos_cls_loss, neg_cls_loss, cls_loss, reg_loss, loss, phase="train", logger=None):
    """
    standard output and logging
    """
    print("Epoch: {}, Step: {}, pos cls loss ({}): {:.3f}".format(epoch, step, phase, pos_cls_loss))
    print("Epoch: {}, Step: {}, neg cls loss ({}): {:.3f}".format(epoch, step, phase, neg_cls_loss))
    print("Epoch: {}, Step: {},     cls loss ({}): {:.3f}".format(epoch, step, phase, cls_loss))                   
    print("Epoch: {}, Step: {},     reg loss ({}): {:.3f}".format(epoch, step, phase, reg_loss))   
    print("Epoch: {}, Step: {},         loss ({}): {:.3f}".format(epoch, step, phase, loss)) 
    if logger is not None:
        logger.critical("Epoch: {}, Step: {}, pos cls loss ({}): {:.3f}".format(epoch, step, phase, pos_cls_loss))
        logger.critical("Epoch: {}, Step: {}, neg cls loss ({}): {:.3f}".format(epoch, step, phase, neg_cls_loss))
        logger.critical("Epoch: {}, Step: {},     cls loss ({}): {:.3f}".format(epoch, step, phase, cls_loss))                   
        logger.critical("Epoch: {}, Step: {},     reg loss ({}): {:.3f}".format(epoch, step, phase, reg_loss))   
        logger.critical("Epoch: {}, Step: {},         loss ({}): {:.3f}".format(epoch, step, phase, loss)) 
        
def main():
    # setting path
    train_dir = os.path.join(cfg.DATA_DIR, "training")
    val_dir = os.path.join(cfg.DATA_DIR, "validation")
    log_dir = os.path.join("./log/" , cfg.TAG)
    save_model_dir = os.path.join("./save_model/" , cfg.TAG)
    train_velodyne_path = os.path.join(train_dir, "velodyne/*.bin")
    train_label_path = os.path.join(train_dir, "label_2/*.txt")
    train_calib_path = os.path.join(train_dir, "calib/*.txt")
    val_velodyne_path = os.path.join(val_dir, "velodyne/*.bin")
    val_label_path = os.path.join(val_dir, "label_2/*.txt")
    val_calib_path = os.path.join(val_dir, "calib/*.txt")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    # config logging
    logging.basicConfig(filename='./log/'+cfg.TAG+'/train.log', level=logging.ERROR)   
    copyfile("model/model.py", os.path.join(log_dir, "model.py"))                       # copy model.py into log/$TAG/
    copyfile("model/layers.py", os.path.join(log_dir, "layers.py"))                     # copy layers.py into log/$TAG/
    copyfile("config.py", os.path.join(log_dir, "config.py"))                           # copy config.py into log/$TAG/
    
    print("tag: {}".format(cfg.TAG))
    logging.critical("tag: {}".format(cfg.TAG))
    max_epoch = cfg.MAX_EPOCH
    config = gpu_config()  
    with tf.Session(config=config) as sess:
        # load Model
        if cfg.MODEL == "FCN":
            model = FCN3D(sess, scale=cfg.SCALE, voxel_shape=cfg.VOXEL_SHAPE, is_training=True, alpha=cfg.ALPHA, beta=cfg.BETA, eta=cfg.ETA, gamma=cfg.GAMMA)
        elif cfg.MODEL == "DENSENET":
            model = DenseNet(sess, scale=cfg.SCALE, voxel_shape=cfg.VOXEL_SHAPE, is_training=True, alpha=cfg.ALPHA, beta=cfg.BETA, eta=cfg.ETA)
        optimizer = model.set_optimizer(lr=cfg.LR)
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
                            max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # param init/restore
        if not cfg.LOAD_CHECKPT == None :
            print("Reading model parameters from {}, {}".format(save_model_dir, cfg.LOAD_CHECKPT))
            logging.critical("Reading model parameters from {}, {}".format(save_model_dir, cfg.LOAD_CHECKPT))
            saver.restore(sess, os.path.join(save_model_dir, cfg.LOAD_CHECKPT))
            start_epoch = model.epoch.eval() + 1
        elif tf.train.get_checkpoint_state(save_model_dir):
            print("Reading model parameters from %s" % save_model_dir)
            logging.critical("Reading model parameters from %s" % save_model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
            start_epoch = model.epoch.eval() + 1
        else:
            print("Created model with fresh parameters.")
            logging.critical("Created model with fresh parameters.")
            tf.global_variables_initializer().run()    
            start_epoch = 0
        end_epoch = max_epoch
        # training
        for epoch in range(start_epoch, end_epoch):
            data_generator_train = data_loader(velodyne_path=train_velodyne_path, \
                    label_path=train_label_path, calib_path = train_calib_path, is_velo_cam=True, shuffle=True)
            data_generator_val = data_loader(velodyne_path=val_velodyne_path, \
                label_path=val_label_path, calib_path = train_calib_path, is_velo_cam=True, shuffle=True)
            for (train_batch_x, train_batch_g_map, train_batch_g_cord) in data_generator_train:
                print("sum of train_batch_g_map is {}".format(np.sum(train_batch_g_map)))
                train_pos_cls_loss, train_neg_cls_loss, train_cls_loss, train_reg_loss, train_loss, train_summary = model.train(sess, train_batch_x, train_batch_g_map, train_batch_g_cord)
                output_log(epoch, model.global_step.eval(), train_pos_cls_loss, train_neg_cls_loss, train_cls_loss, train_reg_loss, train_loss, phase="training", logger=logging)
                summary_writer.add_summary(train_summary, model.global_step.eval())
                # validate
                if model.global_step.eval() % cfg.VALIDATE_INTERVAL == 0:
                    val_batch_x, val_batch_g_map, val_batch_g_cord =  data_generator_val.__next__()
                    val_pos_cls_loss, val_neg_cls_loss, val_cls_loss, val_reg_loss, val_loss, val_summary = model.validate(sess, val_batch_x, val_batch_g_map, val_batch_g_cord)
                    output_log(epoch, model.global_step.eval(), val_pos_cls_loss, val_neg_cls_loss, val_cls_loss, val_reg_loss, val_loss, phase="validation", logger=logging)                  
                    summary_writer.add_summary(val_summary, model.global_step.eval())
            # save weights
            saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step = model.global_step)        
            sess.run(model.epoch_add_op)
        print('{} Training Done!'.format(cfg.TAG))
        logging.critical('{} Training Done!'.format(cfg.TAG))

if __name__ == '__main__':
    main()
