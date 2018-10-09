#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

from config import cfg
from model.layers import batch_norm

class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        self.name = name
        self.dense = tf.layers.Dense(
            self.units, None, name='dense', _reuse=tf.AUTO_REUSE, 
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            bias_initializer=tf.zeros_initializer())

    def apply(self, inputs, mask, training):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as scope:
            # [K, T, 7] tensordot [7, units] = [K, T, units]
            pointwise = tf.nn.relu(batch_norm(self.dense.apply(inputs), phase_train=training, name=self.name))
            #n [K, 1, units]
            aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)
            # [K, T, units]
            repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])
            # [K, T, 2 * units]
            concatenated = tf.concat([pointwise, repeated], axis=2)
            mask = tf.tile(mask, [1, 1, 2 * self.units])
            concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))
        return concatenated


class FeatureNet(object):

    def __init__(self, training, batch_size, name=''):
        super(FeatureNet, self).__init__()
        self.training = training
        self.name = name
        # scalar
        self.batch_size = batch_size
        # [ΣK, 35/45, 7]
        self.feature = tf.placeholder(
            tf.float32, [None, cfg.VOXEL_POINT_COUNT, 7], name='feature')
        # [ΣK]
        self.number = tf.placeholder(tf.int64, [None], name='number')
        # [ΣK, 4], each row stores (batch, d, h, w)
        self.coordinate = tf.placeholder(
            tf.int64, [None, 4], name='coordinate')

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.vfe1 = VFELayer(32, 'VFE-1')
            self.vfe2 = VFELayer(128, 'VFE-2')
            # YUN: Add this line to coincide with original paper and official code
            self.dense = tf.layers.Dense(
                128, None, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope,
                kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                bias_initializer=tf.zeros_initializer())
            # YUN: Add this line to coincide with original paper and official code
            # self.batch_norm = tf.layers.BatchNormalization(
            #     name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

            # boolean mask [K, T, 2 * units]
            mask = tf.not_equal(tf.reduce_max(
                self.feature, axis=2, keep_dims=True), 0)
            x = self.vfe1.apply(self.feature, mask, self.training)
            x = self.vfe2.apply(x, mask, self.training)
            # YUN: Add this line to coincide with original paper and official code
            #FCN
            x = self.dense.apply(x)
            # YUN: Add this line to coincide with original paper and official code
            # x = self.batch_norm.apply(x, self.training)
            x = batch_norm(x, phase_train=self.training, name=name)
            x = tf.nn.relu(x)
            # [ΣK, 128]
            voxelwise = tf.reduce_max(x, axis=1)

            # car: [N * 10 * 400 * 352 * 128]
            # pedestrian/cyclist: [N * 10 * 200 * 240 * 128]
            self.outputs = tf.scatter_nd(
                self.coordinate, voxelwise, [self.batch_size, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])


