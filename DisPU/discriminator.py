# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:49 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com

import tensorflow as tf
from Common import ops as ops
from Common.tf_util2 import pointnet_sa_module_msg,mlp_conv,pointnet_sa_module_msg2,pointnet_sa_module_msg3,mlp_conv2d
class Discriminator(object):
    def __init__(self, opts,is_training, name="Discriminator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.bn = False
        self.start_number = 32
        #print('start_number:',self.start_number)

    def __call__(self, pred, gt=None):
        with tf.variable_scope(self.name, reuse=self.reuse):
            divide_ratio = 2
            l0_xyz = pred
            l0_points = None
            num_point = pred.get_shape()[1].value
            sample_list1 =  [32 // divide_ratio, 32 // divide_ratio, 64 // divide_ratio]
            sample_list2 =  [64 // divide_ratio, 64 // divide_ratio, 128 // divide_ratio]
            sample_list3 =  [64 // divide_ratio, 96 // divide_ratio, 128 // divide_ratio]

            knn = True
            neigh_sample = [8, 16, 24] if knn else [16, 32, 64]

            l1_xyz, l1_points = pointnet_sa_module_msg3(gt, pred, int(num_point / 8),
                                                        [0.1, 0.2, 0.4], neigh_sample,
                                                        [sample_list1, sample_list2, sample_list3],
                                                        scope='layer1', knn=knn)
            patch_values = mlp_conv2d(l1_points, [1], bn=None, bn_params=None, name='patch')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

        return patch_values