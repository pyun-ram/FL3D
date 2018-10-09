#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np

from config import cfg
from model.layers import batch_norm

small_addon_for_BCE = 1e-6


class MiddleAndRPN:
    def __init__(self, input, training, alpha=1.5, beta=1, sigma=3, gamma=2, name=''):
        # scale = [batchsize, 10, 400/200, 352/240, 128] should be the output of feature learning network
        self.input = input
        self.training = training
        # groundtruth(target) - each anchor box, represent as △x, △y, △z, △l, △w, △h, rotation
        self.targets = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14], name="targets")
        # postive anchors equal to one and others equal to zero(2 anchors in 1 position)
        self.pos_equal_one = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2], name="pos_equal_one")
        self.pos_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1], name="pos_equal_one_sum")
        self.pos_equal_one_for_reg = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14], name="pos_equal_one_for_reg")
        # negative anchors equal to one and others equal to zero
        self.neg_equal_one = tf.placeholder(
            tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2], name="neg_equal_one")
        self.neg_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1], name="neg_equal_one_sum")

        with tf.variable_scope('MiddleAndRPN_' + name):
            # convolutinal middle layers
            temp_conv = ConvMD(3, 128, 64, 3, (2, 1, 1), 
                               (1, 1, 1), self.input, training=self.training, name='conv1')
            temp_conv = ConvMD(3, 64, 64, 3, (1, 1, 1),
                               (0, 1, 1), temp_conv,  training=self.training, name='conv2')
            temp_conv = ConvMD(3, 64, 64, 3, (2, 1, 1),
                               (1, 1, 1), temp_conv, training=self.training, name='conv3')
            temp_conv = tf.transpose(temp_conv, perm=[0, 2, 3, 4, 1])
            temp_conv = tf.reshape(
                temp_conv, [-1, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])

            # rpn
            # block1:
            temp_conv = ConvMD(2, 128, 128, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv4')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv5')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv6')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv7')
            deconv1 = Deconv2D(128, 256, 3, (1, 1), (0, 0),
                               temp_conv, training=self.training, name='deconv1')

            # block2:
            temp_conv = ConvMD(2, 128, 128, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv8')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv9')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv10')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv11')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv12')
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv13')
            deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0),
                               temp_conv, training=self.training, name='deconv2')

            # block3:
            temp_conv = ConvMD(2, 128, 256, 3, (2, 2), (1, 1),
                               temp_conv, training=self.training, name='conv14')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv15')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv16')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv17')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv18')
            temp_conv = ConvMD(2, 256, 256, 3, (1, 1), (1, 1),
                               temp_conv, training=self.training, name='conv19')
            deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0),
                               temp_conv, training=self.training, name='deconv3')

            # final:
            temp_conv = tf.concat([deconv3, deconv2, deconv1], -1)
            # Probability score map, scale = [None, 200/100, 176/120, 2]
            p_map = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), temp_conv,
                           training=self.training, activation=False, bn=False, name='conv20')
            # Regression(residual) map, scale = [None, 200/100, 176/120, 14]
            r_map = ConvMD(2, 768, 14, 1, (1, 1), (0, 0),
                           temp_conv, training=self.training, activation=False, bn=False, name='conv21')
            # softmax output for positive anchor and negative anchor, scale = [None, 200/100, 176/120, 1]
            self.p_pos = tf.sigmoid(p_map)
            #self.p_pos = tf.nn.softmax(p_map, dim=3)
            self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
            with tf.variable_scope("losses"):
                if gamma == 0:
                    print("Normal Loss.")
                elif gamma > 0:
                    print("Focal Loss: Gamma:{}".format(gamma))
                else:
                    raise NotImplementedError                
                self.cls_pos_loss = (-self.pos_equal_one * (1 - self.p_pos + small_addon_for_BCE) ** gamma * tf.log(    self.p_pos + small_addon_for_BCE)) / self.pos_equal_one_sum * alpha
                self.cls_neg_loss = (-self.neg_equal_one * (    self.p_pos + small_addon_for_BCE) ** gamma * tf.log(1 - self.p_pos + small_addon_for_BCE)) / self.neg_equal_one_sum * beta                    
                self.cls_loss = tf.reduce_sum( self.cls_pos_loss + self.cls_neg_loss ) * cfg.ETA
                self.cls_pos_loss_rec = tf.reduce_sum( self.cls_pos_loss )
                self.cls_neg_loss_rec = tf.reduce_sum( self.cls_neg_loss )

                self.reg_loss = smooth_l1(r_map * self.pos_equal_one_for_reg, self.targets *
                                        self.pos_equal_one_for_reg, sigma) / self.pos_equal_one_sum
                self.reg_loss = tf.reduce_sum(self.reg_loss)
                self.loss = tf.reduce_sum(self.cls_loss + self.reg_loss)

            self.delta_output = r_map
            self.prob_output = self.p_pos


def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


def ConvMD(M, Cin, Cout, k, s, p, input, training, activation=True, bn=True, name='conv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    with tf.variable_scope(name) as scope:
        if(M == 2):
            paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
            pad = tf.pad(input, paddings, "CONSTANT")
            temp_conv = tf.layers.conv2d(
                pad, Cout, k, strides=s, padding="valid", reuse=tf.AUTO_REUSE, name=scope,
                kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                bias_initializer=tf.zeros_initializer())
        if(M == 3):
            paddings = (np.array(temp_p)).repeat(2).reshape(5, 2)
            pad = tf.pad(input, paddings, "CONSTANT")
            temp_conv = tf.layers.conv3d(
                pad, Cout, k, strides=s, padding="valid", reuse=tf.AUTO_REUSE, name=scope,
                kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                bias_initializer=tf.zeros_initializer())
        if bn:
            # temp_conv = tf.layers.batch_normalization(
            #     temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
            temp_conv = batch_norm(temp_conv, phase_train=training, name=name)
        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv

def Deconv2D(Cin, Cout, k, s, p, input, training, bn=True, name='deconv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values=(0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    pad = tf.pad(input, paddings, "CONSTANT")
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d_transpose(
            pad, Cout, k, strides=s, padding="SAME", reuse=tf.AUTO_REUSE, name=scope,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            bias_initializer=tf.zeros_initializer())
        if bn:
            # temp_conv = tf.layers.batch_normalization(
            #     temp_conv, axis=-1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
            temp_conv = batch_norm(temp_conv, phase_train=training, name=name)
        return tf.nn.relu(temp_conv)

if(__name__ == "__main__"):
    m = MiddleAndRPN(tf.placeholder(
        tf.float32, [None, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))