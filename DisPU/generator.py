# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:49 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import tensorflow as tf
from Common import ops
import numpy as np
import os,sys
import math
#ICLR 18

sys.path.append(os.path.join(os.getcwd(),"tf_ops/sampling"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/nn_distance"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/approxmatch"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/grouping"))

from tf_sampling import gather_point, farthest_point_sample
import tf_sampling
from tf_grouping import query_ball_point, group_point
class Generator(object):
    def __init__(self, opts,is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.out_num_point = int(self.num_point*self.up_ratio)

    def __call__(self, inputs):
        B, N, _ = inputs.get_shape()
        use_bn = False
        n_layer = 6
        K = 16
        filter = 24
        use_noise = False
        dense_block = 4
        bn_decay = 0.95
        use_sm = True
        step_ratio = self.up_ratio
        fine_extracotr = False
        is_off = True
        refine = True
        with tf.variable_scope(self.name+"/generator", reuse=self.reuse):
            coarse_feat = ops.feature_extraction_GCN(inputs, scope='feature_extraction_coarse', dense_block=dense_block,
                                                     growth_rate=filter, is_training=self.is_training, bn_decay=bn_decay, use_bn=use_bn) ## [B.N.C]


            patch_num = self.num_point
            for i in range(round(math.pow(self.opts.up_ratio, 1 / step_ratio))):
                coarse_feat = ops.duplicate_up(inputs, coarse_feat, is_training=self.is_training, up_ratio=step_ratio,
                                               scope="upshuffle_%d"%i, atten=False, edge=False)
                patch_num = int(step_ratio*patch_num)

            coarse = ops.coordinate_regressor(coarse_feat, is_training=self.is_training, layer=512,
                                              use_bn=use_bn, scope="coarse_coordinate_regressor")


        with tf.variable_scope(self.name+"/refine", reuse=self.reuse):
            if fine_extracotr:
                fine_feat =ops.feature_extraction_GCN(coarse, scope='feature_extraction_fine', dense_block=2,
                                                      growth_rate=filter, is_training=self.is_training, bn_decay=bn_decay, use_bn=use_bn) # [B.rN.C]

                fine_feat = tf.concat([fine_feat,coarse_feat],axis=-1)

            else:
                fine_feat = coarse_feat

            if refine:
                new_coarse, fine_feat = ops.PointShuffle2(coarse, fine_feat, nsample=16, mlp=[128, 128, 256],
                                                          is_training=self.is_training, bn_decay=bn_decay,
                                                          scope='PointShuffle', use_bn=use_bn,
                                                          use_knn=True, NL=True, Local=True, refine_point=False)



                fine = ops.coordinate_regressor(fine_feat, is_training=self.is_training, layer=512,
                                                use_bn=use_bn, scope="fine_coordinate_regressor", is_off=is_off)
                if is_off:
                    fine = new_coarse + fine
            else:
                fine = coarse


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return coarse,fine