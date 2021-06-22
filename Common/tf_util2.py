import os,sys
import numpy as np
import tensorflow as tf
import random
import math
from Common.Const import GPU
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/sampling"%GPU))
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/nn_distance"%GPU))
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/approxmatch"%GPU))
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/interpolation"%GPU))
sys.path.append(os.path.join(os.getcwd(),"tf_ops_%s/grouping"%GPU))

from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
import tf_nndistance

def mlp(features, layer_dims, bn=None, bn_params=None,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, num_outputs in enumerate(layer_dims[:-1]):
            features = tf.contrib.layers.fully_connected(
                features, num_outputs,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='fc_%d' % i)
        outputs = tf.contrib.layers.fully_connected(
            features, layer_dims[-1],
            activation_fn=None,
            scope='fc_%d' % (len(layer_dims) - 1))
        return outputs

def mlp_conv(inputs, layer_dims, bn=None, bn_params=None,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf.contrib.layers.conv1d(
                inputs, num_out_channel,
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_%d' % i)
        outputs = tf.contrib.layers.conv1d(
            inputs, layer_dims[-1],
            kernel_size=1,
            activation_fn=None,
            scope='conv_%d' % (len(layer_dims) - 1))
        return outputs

def mlp_conv2d(inputs, layer_dims, bn=None, bn_params=None,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf.contrib.layers.conv2d(
                inputs, num_out_channel,
                kernel_size=[1,1],
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_%d' % i)
        outputs = tf.contrib.layers.conv2d(
            inputs, layer_dims[-1],
            kernel_size=[1,1],
            activation_fn=None,
            scope='conv_%d' % (len(layer_dims) - 1))
        return outputs

def chamfer_distance(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    return dist1 , dist2

def gen_grid_up(up_ratio):
    sqrted = int(math.sqrt(up_ratio))+1
    for i in range(1,sqrted+1).__reversed__():
        if (up_ratio%i) == 0:
            num_x = i
            num_y = up_ratio//i
            break
    grid_x = tf.linspace(-0.2, 0.2, num_x)
    grid_y = tf.linspace(-0.2, 0.2, num_y)

    x, y = tf.meshgrid(grid_x, grid_y)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid

def gen_grid(num_grid_point):
  """
  output [num_grid_point, 2]
  """
  x = tf.linspace(-0.05, 0.05, num_grid_point)
  x, y = tf.meshgrid(x, x)
  grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
  return grid



def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse) as sc:
        if weight_decay>0:
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer=None
        outputs = tf.contrib.layers.conv2d(inputs, num_output_channels, kernel_size, stride, padding,
                                            activation_fn=activation_fn,weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           weights_regularizer=regularizer,
                                           biases_regularizer=regularizer)
        return outputs

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, scope, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        p1_idx = farthest_point_sample(npoint, xyz)
        new_xyz = gather_point(xyz, p1_idx)
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx) #b*n*k*3
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = conv2d(grouped_points, num_out_channel, [1,1],weight_decay=0,
                                        padding='VALID', stride=[1,1], scope='conv%d_%d'%(i,j),activation_fn=tf.nn.leaky_relu)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2]) #b*n*c
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


def pointnet_sa_module_msg2(gt, pred, npoint, radius_list, nsample_list, mlp_list, scope, use_xyz=True, use_nchw=False,knn=True):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        p1_idx = farthest_point_sample(npoint, gt)
        source_gt = gather_point(gt, p1_idx)
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            if knn:
                _, idx_gt = knn_point(nsample, gt, source_gt)
            else:
                idx_gt, _ = query_ball_point(radius, nsample, gt, source_gt)
            grouped_gt = group_point(gt, idx_gt) #b*n*k*3
            grouped_gt -= tf.tile(tf.expand_dims(source_gt, 2), [1,1,nsample,1])

            if knn:
                _, idx_pred = knn_point(nsample, pred, source_gt)
            else:
                idx_pred, _ = query_ball_point(radius, nsample, pred, source_gt)
            grouped_pred = group_point(pred, idx_pred)  # b*n*k*3
            grouped_pred -= tf.tile(tf.expand_dims(source_gt, 2), [1, 1, nsample, 1])

            grouped_points = tf.concat([grouped_gt,grouped_pred],axis=2)

            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = conv2d(grouped_points, num_out_channel, [1,1],weight_decay=0,
                                        padding='VALID', stride=[1,1], scope='conv%d_%d'%(i,j),activation_fn=tf.nn.leaky_relu)

            #new_points = tf.reduce_max(grouped_points, axis=[2]) #b*n*c
            feat_max = tf.layers.max_pooling2d(grouped_points, [1, nsample], strides=[1, nsample], padding='VALID', name='maxpool_%d'%i)
            new_points = conv2d(feat_max, mlp_list[i][-1], [1, 2], weight_decay=0,
                   padding='VALID', stride=[1,1], scope='conv_fusion_%d'%i, activation_fn=tf.nn.leaky_relu)

            new_points = tf.squeeze(new_points,axis=2)
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return source_gt, new_points_concat


def pointnet_sa_module_msg3(gt, pred, npoint, radius_list, nsample_list, mlp_list, scope, use_xyz=True, use_nchw=False,knn=True):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        p1_idx = farthest_point_sample(npoint, gt)
        source_gt = gather_point(gt, p1_idx)
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            if knn:
                _, idx_gt = knn_point(nsample, gt, source_gt)
            else:
                idx_gt, _ = query_ball_point(radius, nsample, gt, source_gt)
            grouped_gt = group_point(gt, idx_gt) #b*n*k*3
            grouped_gt -= tf.tile(tf.expand_dims(source_gt, 2), [1,1,nsample,1])

            if knn:
                _, idx_pred = knn_point(nsample, pred, source_gt)
            else:
                idx_pred, _ = query_ball_point(radius, nsample, pred, source_gt)
            grouped_pred = group_point(pred, idx_pred)  # b*n*k*3
            grouped_pred -= tf.tile(tf.expand_dims(source_gt, 2), [1, 1, nsample, 1])

            grouped_points = tf.concat([grouped_gt,grouped_pred],axis=2)

            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = conv2d(grouped_points, num_out_channel, [1,1],weight_decay=0,
                                        padding='VALID', stride=[1,1], scope='conv%d_%d'%(i,j),activation_fn=tf.nn.leaky_relu)

            #new_points = tf.reduce_max(grouped_points, axis=[2]) #b*n*c
            new_points = tf.layers.max_pooling2d(grouped_points, [1, nsample], strides=[1, nsample], padding='VALID', name='maxpool_%d'%i)

            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return source_gt, new_points_concat

