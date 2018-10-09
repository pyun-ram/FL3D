'''
 Peng YUN Created on Sat Aug 04 2018

 Copyright (c) 2018 RAM-LAB
'''

import os
import logging
from shutil import copyfile
import tensorflow as tf
from config import cfg
from utils.kitti_loader import iterate_data, sample_test_data
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
    # load config
    train_dataset_dir = os.path.join(cfg.DATA_DIR, "training")
    val_dataset_dir = os.path.join(cfg.DATA_DIR, "validation")
    eval_dataset_dir = os.path.join(cfg.DATA_DIR, "validation")
    save_model_dir = os.path.join("./save_model", cfg.TAG)
    log_dir = os.path.join("./log", cfg.TAG)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    config = gpu_config()
    max_epoch = cfg.MAX_EPOCH

    # config logging
    logging.basicConfig(filename='./log/'+cfg.TAG+'/train.log', level=logging.ERROR)   
    copyfile("model/model.py", os.path.join(log_dir, "model.py"))                                             # copyu model.py into log/$TAG/
    copyfile("model/group_pointcloud.py", os.path.join(log_dir, "group_pointcloud.py"))                       # copyu model.py into log/$TAG/
    copyfile("model/rpn.py", os.path.join(log_dir, "rpn.py"))                                                 # copy rpn.py into log/$TAG/
    copyfile("config.py", os.path.join(log_dir, "config.py"))                                                 # copy config.py into log/$TAG/

    print("tag: {}".format(cfg.TAG))
    logging.critical("tag: {}".format(cfg.TAG))
    with tf.Session(config=config) as sess:
        # load model
        model = RPN3D(cls=cfg.DETECT_OBJ, single_batch_size=cfg.SINGLE_BATCH_SIZE, is_training=True, 
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
        
        # train
        for epoch in range(start_epoch, max_epoch):
            # load data
            data_generator_train = iterate_data(train_dataset_dir, shuffle=True, aug=False, is_testset=False, 
                batch_size=cfg.SINGLE_BATCH_SIZE * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT)
            data_generator_val = iterate_data(val_dataset_dir, shuffle=True, aug=False, is_testset=False, 
                batch_size=cfg.SINGLE_BATCH_SIZE * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT) 
            for batch in data_generator_train:
                # train
                ret = model.train_step( sess, batch, train=True, summary = True )
                output_log(epoch, model.global_step.eval(), 
                    pos_cls_loss=ret[3], neg_cls_loss=ret[4], cls_loss=ret[2], 
                    reg_loss=ret[1], loss=ret[0], phase="train", logger=logging)
                summary_writer.add_summary(ret[-1], model.global_step.eval())
                if model.global_step.eval() % cfg.VALIDATE_INTERVAL == 0:
                    # val
                    val_batch = data_generator_val.__next__()
                    ret = model.validate_step(sess, val_batch, summary=True)
                    output_log(epoch, model.global_step.eval(), 
                        pos_cls_loss=ret[3], neg_cls_loss=ret[4], cls_loss=ret[2], 
                        reg_loss=ret[1], loss=ret[0], phase="validation", logger=logging)  
                    summary_writer.add_summary(ret[-1], model.global_step.eval())
                    # eval
                    eval_batch =sample_test_data(eval_dataset_dir, cfg.SINGLE_BATCH_SIZE * cfg.GPU_USE_COUNT, multi_gpu_sum=cfg.GPU_USE_COUNT)
                    try:
                        ret = model.predict_step(sess, eval_batch, summary=True)
                        summary_writer.add_summary(ret[-1], model.global_step.eval())
                    except:
                        print("prediction skipped due to error")                    

            sess.run(model.epoch_add_op)
            model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)
        print('{} Training Done!'.format(cfg.TAG))
        logging.critical('{} Training Done!'.format(cfg.TAG))

if __name__ == '__main__':
    main()