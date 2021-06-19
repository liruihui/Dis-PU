# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
import numpy as np
import os
import sys


sys.path.append(os.path.join(os.getcwd(),"tf_ops/sampling"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/nn_distance"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/approxmatch"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/interpolation"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/grouping"))
from tf_interpolate import three_nn, three_interpolate
import tf_grouping
import tf_sampling
import libs.nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors
from tf_grouping import query_ball_point, group_point, knn_point,knn_point_2
from Common import tf_util
from Common.tf_util import conv1d,conv2d
from functools import partial, update_wrapper
sys.path.append(os.path.dirname(os.getcwd()))
def mlp(features, layer_dims, bn=None, bn_params=None,name='mlp'):
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


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None,name='mlp_conv'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf.contrib.layers.conv2d(
                inputs, num_out_channel,
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_%d' % i)
        outputs = tf.contrib.layers.conv2d(
            inputs, layer_dims[-1],
            kernel_size=1,
            activation_fn=None,
            scope='conv_%d' % (len(layer_dims) - 1))
        return outputs

##################################################################################
# Back projection Blocks
##################################################################################



def gen_grid(up_ratio):
    import math
    """
    output [num_grid_point, 2]
    """
    sqrted = int(math.sqrt(up_ratio))+1
    for i in range(1,sqrted+1).__reversed__():
        if (up_ratio%i) == 0:
            num_x = i
            num_y = up_ratio//i
            break
    grid_x = tf.lin_space(-0.2, 0.2, num_x)
    grid_y = tf.lin_space(-0.2, 0.2, num_y)

    x, y = tf.meshgrid(grid_x, grid_y)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid


# if step_ratio < 4:
#     grid = gen_1d_grid(step_ratio)
#     expansion_ratio = step_ratio
# else:
#     grid = gen_grid(np.round(np.sqrt(step_ratio)).astype(np.int32))
#     expansion_ratio = (np.round(np.sqrt(step_ratio))**2).astype(np.int32)

# grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size))
# print('grid:', grid)
# grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
# print('grid:', grid)
# grid_feat = tf.tile(grid, [coarse.shape[0], self.num_coarse, 1])
# print('grid_feat', grid_feat)

def gen_2D_grid(num_grid_point):
    """
    output [num_grid_point, 2]
    """
    x = tf.lin_space(-0.2, 0.2, num_grid_point)
    x, y = tf.meshgrid(x, x)
    grid = tf.reshape(tf.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid

def gen_1d_grid(num_grid_point):
    """
    output [num_grid_point, 2]
    """
    x = tf.lin_space(-0.02, 0.02, num_grid_point)
    grid = tf.reshape(x, [1,-1])  # [2, 2, 2] -> [4, 2]
    return grid

def knn_query(k, support_pts, query_pts):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """
    neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
    return neighbor_idx.astype(np.int32)

def sampling(npoint, pts, feature=None):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * D, input point cloud
    output:
    sub_pts: B * npoint * D, sub-sampled point cloud
    '''
    batch_size = pts.get_shape()[0]
    fps_idx = tf_sampling.farthest_point_sample(npoint, pts)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, npoint,1))
    idx = tf.concat([batch_indices, tf.expand_dims(fps_idx, axis=2)], axis=2)
    idx.set_shape([batch_size, npoint, 2])
    if feature is None:
        return tf.gather_nd(pts, idx)
    else:
        return tf.gather_nd(pts, idx), tf.gather_nd(feature, idx)


def dilat_group(xyz, points, k, dilation=1, use_xyz=False):
    _, idx = knn_point(k*dilation+1, xyz, xyz)
    idx = idx[:, :, 1::dilation]

    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, k, 3)
    grouped_xyz -= tf.expand_dims(xyz, 2)  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, k, channel)
        if use_xyz:
            grouped_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, k, 3+channel)
    else:
        grouped_points = grouped_xyz

    return grouped_xyz, grouped_points, idx

def grouping(feature, K, src_xyz, q_xyz, use_xyz=True, use_knn=True, radius=0.2):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    '''

    batch_size = src_xyz.get_shape()[0]
    npoint = q_xyz.get_shape()[1]

    if use_knn:
        point_indices = tf.py_func(knn_query, [K, src_xyz, q_xyz], tf.int32)
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
        idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        idx.set_shape([batch_size, npoint, K, 2])
        grouped_xyz = tf.gather_nd(src_xyz, idx)
    else:
        point_indices, _ = tf_grouping.query_ball_point(radius, K, src_xyz, q_xyz)
        grouped_xyz = tf_grouping.group_point(src_xyz, point_indices)

    grouped_feature = tf.gather_nd(feature, idx)

    if use_xyz:
        grouped_feature = tf.concat([grouped_xyz, grouped_feature], axis=-1)

    return grouped_xyz, grouped_feature, idx

def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            net = conv2d(net, num_hidden_units, [1, 1],
                                padding = 'VALID', stride=[1, 1],
                                bn = True, is_training = is_training, activation_fn=activation_fn,
                                scope = 'wconv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

    return net

def SampleWeights(new_point, grouped_xyz, mlps, is_training, bn_decay=None, scope="SampleWeights", bn=True, scaled=True):
    """Input
        grouped_feature: (batch_size, npoint, nsample, channel) TF tensor
        grouped_xyz: (batch_size, npoint, nsample, 3)
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, 1)
    """
    with tf.variable_scope(scope) as sc:
        batch_size, npoint, nsample, channel = new_point.get_shape()
        bottleneck_channel = max(32,channel//2)
        normalized_xyz = grouped_xyz - tf.tile(tf.expand_dims(grouped_xyz[:, :, 0, :], 2), [1, 1, nsample, 1])
        new_point = tf.concat([normalized_xyz, new_point], axis=-1) # (batch_size, npoint, nsample, channel+3)

        transformed_feature = conv2d(new_point, bottleneck_channel * 2, bn=bn, is_training=is_training,
                                             scope='conv_kv_ds', bn_decay=bn_decay,
                                             activation_fn=None)
        transformed_new_point = conv2d(new_point, bottleneck_channel, bn=bn, is_training=is_training,
                                               scope='conv_query_ds', bn_decay=bn_decay,
                                               activation_fn=None)

        transformed_feature1 = transformed_feature[:, :, :, :bottleneck_channel]
        feature = transformed_feature[:, :, :, bottleneck_channel:]

        weights = tf.matmul(transformed_new_point, transformed_feature1, transpose_b=True)  # (batch_size, npoint, nsample, nsample)
        if scaled:
            weights = weights / tf.sqrt(tf.cast(bottleneck_channel, tf.float32))
        weights = tf.nn.softmax(weights, axis=-1)
        channel = bottleneck_channel

        new_group_features = tf.matmul(weights, feature)
        new_group_features = tf.reshape(new_group_features, (batch_size, npoint, nsample, channel))
        for i, c in enumerate(mlps):
            activation = tf.nn.relu if i < len(mlps) - 1 else None
            new_group_features = conv2d(new_group_features, c, [1, 1],
                                               padding='VALID', stride=[1, 1],
                                               bn=bn, is_training=is_training,
                                               scope='mlp2_%d' % (i), bn_decay=bn_decay, activation_fn=activation)
        new_group_weights = tf.nn.softmax(new_group_features, axis=2)  # (batch_size, npoint,nsample, mlp[-1)
        return new_group_weights


def SampleOffset(new_point, grouped_xyz, mlps, is_training, bn_decay=None, scope="SampleOffset", bn=True, scaled=True):
    """Input
        grouped_feature: (batch_size, npoint, nsample, channel) TF tensor
        grouped_xyz: (batch_size, npoint, nsample, 3)
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, 1)
    """
    with tf.variable_scope(scope) as sc:
        batch_size, npoint, nsample, channel = new_point.get_shape()
        bottleneck_channel = max(32,channel//2)
        normalized_xyz = grouped_xyz - tf.tile(tf.expand_dims(grouped_xyz[:, :, 0, :], 2), [1, 1, nsample, 1])
        new_point = tf.concat([normalized_xyz, new_point], axis=-1) # (batch_size, npoint, nsample, channel+3)

        transformed_feature = conv2d(new_point, bottleneck_channel * 2, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=bn, is_training=is_training,
                                             scope='conv_kv_ds', bn_decay=bn_decay,
                                             activation_fn=None)
        transformed_new_point = conv2d(new_point, bottleneck_channel, [1, 1],
                                               padding='VALID', stride=[1, 1],
                                               bn=bn, is_training=is_training,
                                               scope='conv_query_ds', bn_decay=bn_decay,
                                               activation_fn=None)

        transformed_feature1 = transformed_feature[:, :, :, :bottleneck_channel]
        feature = transformed_feature[:, :, :, bottleneck_channel:]

        weights = tf.matmul(transformed_new_point, transformed_feature1, transpose_b=True)  # (batch_size, npoint, nsample, nsample)
        if scaled:
            weights = weights / tf.sqrt(tf.cast(bottleneck_channel, tf.float32))
        weights = tf.nn.softmax(weights, axis=-1)
        channel = bottleneck_channel

        new_group_features = tf.matmul(weights, feature)
        new_group_features = tf.reshape(new_group_features, (batch_size, npoint, nsample, channel))
        new_group_features = tf.reduce_max(new_group_features,axis=-2)
        for i, c in enumerate(mlps):
            activation = tf.nn.relu if i < len(mlps) - 1 else None
            use_bn = bn if i < len(mlps) - 1 else False
            new_group_features = conv1d(new_group_features, c,
                                               bn=use_bn, is_training=is_training,
                                               scope='mlp2_%d' % (i), bn_decay=bn_decay,
                                               activation_fn=activation)
        range_max = 0.5
        offset = tf.sigmoid(new_group_features) * range_max * 2 - range_max #(batch_size, npoint,nsample, 3)
        return offset

def AdaptiveSampling(group_xyz, group_feature, num_neighbor, is_training, bn_decay, weight_decay, scope, bn):
    with tf.variable_scope(scope) as sc:
        nsample, num_channel = group_feature.get_shape()[-2:]
        if num_neighbor == 0:
            new_xyz = group_xyz[:, :, 0, :]
            new_feature = group_feature[:, :, 0, :]
            return new_xyz, new_feature
        shift_group_xyz = group_xyz[:, :, :num_neighbor, :]
        shift_group_points = group_feature[:, :, :num_neighbor, :]
        sample_weight = SampleWeights(shift_group_points, shift_group_xyz, [32, 1 + num_channel],
                                      is_training, bn_decay=bn_decay, bn=bn)
        new_weight_xyz = tf.tile(tf.expand_dims(sample_weight[:,:,:, 0],axis=-1), [1, 1, 1, 3])
        new_weight_feture = sample_weight[:,:,:, 1:]
        new_xyz = tf.reduce_sum(tf.multiply(shift_group_xyz, new_weight_xyz), axis=[2])
        new_feature = tf.reduce_sum(tf.multiply(shift_group_points, new_weight_feture), axis=[2])

        return new_xyz, new_feature


def PointNonLocalCell(feature,new_point,mlp,is_training, bn_decay, weight_decay, scope, bn=True, scaled=True, mode='dot'):
    """Input
        feature: (batch_size, ndataset, channel) TF tensor
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, channel)
    """
    with tf.variable_scope(scope) as sc:
        bottleneck_channel = mlp[0]
        batch_size, npoint, nsample, channel = new_point.get_shape()
        ndataset = feature.get_shape()[1]
        feature = tf.expand_dims(feature,axis=2) #(batch_size, ndataset, 1, channel)
        transformed_feature =  conv2d(feature, bottleneck_channel * 2, [1,1],
                                                padding='VALID', stride=[1,1],
                                                bn=bn, is_training=is_training,
                                                scope='conv_kv', bn_decay=bn_decay, weight_decay = weight_decay, activation_fn=None)
        transformed_new_point = conv2d(new_point, bottleneck_channel, [1,1],
                                                padding='VALID', stride=[1,1],
                                                bn=bn, is_training=is_training,
                                                scope='conv_query', bn_decay=bn_decay, weight_decay = weight_decay, activation_fn=None) #(batch_size, npoint, nsample, bottleneck_channel)
        transformed_new_point = tf.reshape(transformed_new_point, [batch_size, npoint*nsample, bottleneck_channel])
        transformed_feature1 = tf.squeeze(transformed_feature[:,:,:,:bottleneck_channel],axis=[2]) #(batch_size, ndataset, bottleneck_channel)
        transformed_feature2 = tf.squeeze(transformed_feature[:,:,:,bottleneck_channel:],axis=[2]) #(batch_size, ndataset, bottleneck_channel)
        if mode == 'dot':
            attention_map = tf.matmul(transformed_new_point, transformed_feature1,transpose_b=True) #(batch_size, npoint*nsample, ndataset)
            if scaled:
                attention_map = attention_map / tf.sqrt(tf.cast(bottleneck_channel,tf.float32))
        elif mode == 'concat':
            tile_transformed_feature1 = tf.tile(tf.expand_dims(transformed_feature1,axis=1),(1,npoint*nsample,1,1)) # (batch_size,npoint*nsample, ndataset, bottleneck_channel)
            tile_transformed_new_point = tf.tile(tf.reshape(transformed_new_point, (batch_size, npoint*nsample, 1, bottleneck_channel)), (1,1,ndataset,1)) # (batch_size,npoint*nsample, ndataset, bottleneck_channel)
            merged_feature = tf.concat([tile_transformed_feature1,tile_transformed_new_point], axis=-1)
            attention_map = conv2d(merged_feature, 1, [1,1],
                                                padding='VALID', stride=[1,1],
                                                bn=bn, is_training=is_training,
                                                scope='conv_attention_map', bn_decay=bn_decay, weight_decay = weight_decay)
            attention_map = tf.reshape(attention_map, (batch_size, npoint*nsample, ndataset))
        attention_map = tf.nn.softmax(attention_map, axis=-1)
        new_nonlocal_point = tf.matmul(attention_map, transformed_feature2) #(batch_size, npoint*nsample, bottleneck_channel)
        new_nonlocal_point = conv2d(tf.reshape(new_nonlocal_point,[batch_size,npoint, nsample, bottleneck_channel]), mlp[-1], [1,1],
                                                padding='VALID', stride=[1,1],
                                                bn=bn, is_training=is_training,
                                                scope='conv_back_project', bn_decay=bn_decay, weight_decay = weight_decay)
        new_nonlocal_point = tf.squeeze(new_nonlocal_point, axis=[1])  # (batch_size, npoints, mlp2[-1])

        return new_nonlocal_point

def PointASNLSetAbstraction(xyz, feature, npoint, nsample, mlp, is_training, bn_decay, weight_decay, scope, bn=True, use_knn=True, radius=None, as_neighbor=8, NL=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        batch_size, num_points, num_channel = feature.get_shape()
        '''Farthest Point Sampling'''
        if num_points == npoint:
            new_xyz = xyz
            new_feature = feature
        else:
            new_xyz, new_feature = sampling(npoint, xyz, feature)

        grouped_xyz, new_point, idx = grouping(feature, nsample, xyz, new_xyz,use_knn=use_knn,radius=radius)

        '''Adaptive Sampling'''
        if num_points != npoint:
            new_xyz, new_feature = AdaptiveSampling(grouped_xyz, new_point, as_neighbor, is_training, bn_decay, weight_decay, scope, bn)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
        new_point = tf.concat([grouped_xyz, new_point], axis=-1)

        '''Point NonLocal Cell'''
        if NL:
            new_nonlocal_point = PointNonLocalCell(feature, tf.expand_dims(new_feature, axis=1),
                                                   [max(32, num_channel//2), nl_channel],
                                                   is_training, bn_decay, weight_decay, scope, bn)

        '''Skip Connection'''
        skip_spatial = tf.reduce_max(new_point, axis=[2])
        skip_spatial = conv1d(skip_spatial, mlp[-1], 1,padding='VALID', stride=1,
                                     bn=bn, is_training=is_training, scope='skip',
                                     bn_decay=bn_decay, weight_decay=weight_decay)

        '''Point Local Cell'''
        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                new_point = conv2d(new_point, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)


        weight = weight_net_hidden(grouped_xyz, [32], scope = 'weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
        new_point = tf.transpose(new_point, [0, 1, 3, 2])
        new_point = tf.matmul(new_point, weight)
        new_point = conv2d(new_point, mlp[-1], [1,new_point.get_shape()[2].value],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

        new_point = tf.squeeze(new_point, [2])  # (batch_size, npoints, mlp2[-1])

        new_point = tf.add(new_point,skip_spatial)

        if NL:
            new_point = tf.add(new_point, new_nonlocal_point)

        '''Feature Fushion'''
        new_point = conv1d(new_point, mlp[-1], 1,
                                  padding='VALID', stride=1, bn=bn, is_training=is_training,
                                  scope='aggregation', bn_decay=bn_decay, weight_decay=weight_decay)

        return new_xyz, new_point


from gcn_lib import tf_vertex
from gcn_lib import tf_edge
from gcn_lib.tf_nn import MLP
from gcn_lib.gcn_utils import VertexLayer,EdgeLayer

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def build_gcn_backbone_block(inputs, is_training, bn_decay=None, n_layer=6, use_bn=False, K=16, filter=64, scope="gcn"):
    with tf.variable_scope(scope) as sc:
        GCN = "edgeconv"
        EDGE_LAY = ["dilated","knn"][1]
        stochastic_dilation = True
        sto_dilated_epsilon = 0.2

        skip_connect = ["residual", "dense", None][1]
        nn = MLP(kernel_size=[1, 1],
                 stride=[1, 1],
                 padding='VALID',
                 weight_decay=0.0,
                 bn=use_bn,
                 bn_decay=bn_decay,
                 is_dist=True)

        v_layer = tf_vertex.edge_conv_layer
        v_layer_builder = VertexLayer(v_layer, nn, K, filter)

        # Configure the gcn edge layer object
        if EDGE_LAY == 'dilated':
            dilations = [1] + list(range(1, n_layer))
            e_layer = wrapped_partial(tf_edge.dilated_knn_graph,
                                      stochastic=stochastic_dilation,
                                      epsilon=sto_dilated_epsilon)
        elif EDGE_LAY == 'knn':
            dilations = [None] * n_layer
            e_layer = tf_edge.knn_graph

        distance_metric = tf_util.pairwise_distance

        e_layer_builder = EdgeLayer(e_layer,K,distance_metric)

        '''Build the gcn backbone block'''
        input_graph = tf.expand_dims(inputs, -2)
        graphs = []

        for i in range(n_layer):
            if i == 0:
                neigh_idx = e_layer_builder.build(input_graph,
                                                     dilation=dilations[i],
                                                     is_training=is_training)
                vertex_features = v_layer_builder.build(input_graph,
                                                             neigh_idx=neigh_idx,
                                                             scope='adj_conv_' + str(i),
                                                             is_training=is_training)
                graph = vertex_features
                graphs.append(graph)
            else:
                neigh_idx = e_layer_builder.build(graphs[-1],
                                                     dilation=dilations[i],
                                                     is_training=is_training)
                vertex_features = v_layer_builder.build(graphs[-1],
                                                             neigh_idx=neigh_idx,
                                                             scope='adj_conv_' + str(i),
                                                             is_training=is_training)
                graph = vertex_features
                if skip_connect == 'residual':
                    graph = graph + graphs[-1]
                elif skip_connect == 'dense':
                    graph = tf.concat([graph, graphs[-1]], axis=-1)
                elif skip_connect == 'none':
                    graph = graph
                else:
                    raise Exception('Unknown connections')
                graphs.append(graph)


        return tf.squeeze(graphs[-1],axis=-2)
        #return tf.squeeze(tf.concat(graphs, axis=-1),axis=-2)  #num_layers*filter


from Common.pointnet_util import pointnet_sa_module, pointnet_fp_module

def hierachy_feature_extractor(inputs, is_training, bn_decay=None, use_bn=False, scope="hierachy_feature_extractor", npoints = [1024,384,128], radius=[0.1,0.2,0.4]):

    B, N, C = inputs.get_shape()

    l0_xyz = inputs
    l0_points = None

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=npoints[0], radius=radius[0],
                                                       nsample=64,
                                                       mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=npoints[1], radius=radius[1],
                                                       nsample=64,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=npoints[2], radius=radius[2] ,
                                                       nsample=64,
                                                       mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # PointNet
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[256, 256, 512], mlp2=None, group_all=True,
                                                       is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    # l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points, [512,512], is_training, bn_decay, scope='fa_layer0')
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512, 512], is_training, bn_decay,
                                   scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512, 256], is_training, bn_decay,
                                   scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay,
                                   scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay,
                                   scope='fa_layer4')


    # net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1',
    #                      bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')
    #
    # displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max

    return l0_points


def hierachy_feature_extractor2(inputs, is_training, bn_decay=None, use_bn=False, scope="hierachy_feature_extractor"):
    reuse = None
    bradius = 1.0
    up_ratio = 4
    with tf.variable_scope(scope,reuse=reuse) as sc:
        batch_size = inputs.get_shape()[0].value
        num_point = inputs.get_shape()[1].value
        l0_xyz = inputs
        l0_points = None

        # Layer 1
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=num_point, radius=bradius*0.05,bn=use_bn,
                                                           nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer1')

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point/2, radius=bradius*0.1,bn=use_bn,
                                                           nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer2')

        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point/4, radius=bradius*0.2,bn=use_bn,
                                                           nsample=32, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer3')

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point/8, radius=bradius*0.3,bn=use_bn,
                                                           nsample=32, mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                           is_training=is_training, bn_decay=bn_decay, scope='layer4')

        # Feature Propagation layers
        up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                       scope='fa_layer1',bn=use_bn)

        up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                       scope='fa_layer2',bn=use_bn)

        up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                       scope='fa_layer3',bn=use_bn)

        ###concat feature
        with tf.variable_scope('up_layer',reuse=reuse):
            new_points_list = []
            for i in range(up_ratio):
                concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=-1)
                concat_feat = tf.expand_dims(concat_feat, axis=2)
                concat_feat = tf_util.conv2d(concat_feat, 256, [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=False, is_training=is_training,
                                              scope='fc_layer0_%d'%(i), bn_decay=bn_decay)

                new_points = tf_util.conv2d(concat_feat, 128, [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=use_bn, is_training=is_training,
                                             scope='conv_%d' % (i),
                                             bn_decay=bn_decay)
                new_points_list.append(new_points)
            net = tf.concat(new_points_list,axis=1)

        #get the xyz
        coord = tf_util.conv2d(net, 64, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='fc_layer1', bn_decay=bn_decay)

        coord = tf_util.conv2d(coord, 3, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='fc_layer2', bn_decay=bn_decay,
                             activation_fn=None, weight_decay=0.0)  # B*(2N)*1*3
        coord = tf.squeeze(coord, [2])  # B*(2N)*3

    return coord,None

def PointDownscale(xyz, feature, npoint, nsample, mlp, is_training, bn_decay, scope, bn=True, use_knn=True, radius=None, as_neighbor=8, NL=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        batch_size, num_points, num_channel = feature.get_shape()
        '''Farthest Point Sampling'''
        if num_points == npoint:
            new_xyz = xyz
            new_feature = feature
        else:
            new_xyz, new_feature = sampling(npoint, xyz, feature)

        group_xyz, group_feature, idx = grouping(feature, nsample, xyz, new_xyz,use_knn=use_knn,radius=radius)
        nl_channel = mlp[-1]

        nsample, num_channel = group_xyz.get_shape()[-2:]

        shift_group_xyz = group_xyz[:, :, :as_neighbor, :]
        shift_group_points = group_feature[:, :, :as_neighbor, :]
        sample_weight = SampleWeights(shift_group_points, shift_group_xyz, [32, 1], is_training, bn_decay,scope, bn)
        new_weight_xyz = tf.tile(tf.expand_dims(sample_weight[:, :, :, 0], axis=-1), [1, 1, 1, 3])
        new_offset = tf.reduce_sum(tf.multiply(shift_group_xyz, new_weight_xyz), axis=[2])

        return new_xyz,new_offset


def PointDownscale3(xyz, feature, npoint, nsample, mlp, is_training, bn_decay, scope, use_bn=False, use_knn=True, radius=None, as_neighbor=8, NL=True, use_noise=False, use_sm=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        batch_size, num_points, num_channel = feature.get_shape()
        '''Farthest Point Sampling'''
        if num_points == npoint:
            new_xyz = xyz
            new_feature = feature
        else:
            new_xyz, new_feature = sampling(npoint, xyz, feature)

        group_xyz, group_feature, idx = grouping(feature, nsample, xyz, new_xyz,use_knn=use_knn,radius=radius)
        nl_channel = mlp[-1]

        nsample, num_channel = group_feature.get_shape()[-2:]


        shift_group_xyz = group_xyz[:, :, :as_neighbor, :]
        shift_group_points = group_feature[:, :, :as_neighbor, :]
        sample_weight = SampleWeights(shift_group_points, shift_group_xyz, [num_channel, num_channel], is_training, bn_decay, bn=use_bn)
        new_feature = tf.reduce_sum(tf.multiply(shift_group_points, sample_weight), axis=[2])

        #global_feat = tf.reduce_max(new_feature,axis=1,keep_dims=True)

        mlps = [num_channel,64,3]

        if use_noise:
            noise = tf.random_normal(shape=[tf.shape(new_feature)[0], tf.shape(new_feature)[1], 16], mean=0.0, stddev=1)
            new_feature = tf.concat(axis=2, values=[new_feature, noise])

        for i, c in enumerate(mlps):
            activation = tf.nn.relu if i < len(mlps) - 1 else None
            use_bn = use_bn if i < len(mlps) - 1 else False
            new_feature = conv1d(new_feature, c,
                                        bn=use_bn, is_training=is_training,
                                        scope='mlp2_%d' % (i), bn_decay=bn_decay,
                                        activation_fn=activation)
        new_offset = new_feature
        if use_sm:
            range_max = 0.5
            new_offset = tf.sigmoid(new_offset) * range_max * 2 - range_max

        return new_xyz,new_offset



def PointDownscale3_1(xyz, feature, npoint, nsample, mlp, is_training, bn_decay, scope, use_bn=False, use_knn=True, radius=None, as_neighbor=8, NL=True, use_noise=False, use_sm=True,weight_decay=None):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        batch_size, num_points, num_channel = feature.get_shape()
        '''Farthest Point Sampling'''
        if num_points == npoint:
            new_xyz = xyz
            new_feature = feature
        else:
            new_xyz, new_feature = sampling(npoint, xyz, feature)

        grouped_xyz, new_point, idx = grouping(feature, nsample, xyz, new_xyz, use_knn=use_knn, radius=radius)
        nl_channel = mlp[-1]

        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
        new_point = tf.concat([grouped_xyz, new_point], axis=-1)

        '''Point NonLocal Cell'''
        if NL:
            new_nonlocal_point = PointNonLocalCell(feature, tf.expand_dims(new_feature, axis=1),
                                                   [max(32, num_channel // 2), nl_channel],
                                                   is_training, bn_decay, weight_decay, scope, use_bn)

        '''Skip Connection'''
        skip_spatial = tf.reduce_max(new_point, axis=[2])
        skip_spatial = tf_util.conv1d(skip_spatial, mlp[-1], 1, padding='VALID', stride=1,
                                      bn=use_bn, is_training=is_training, scope='skip',
                                      bn_decay=bn_decay, weight_decay=weight_decay)

        '''Point Local Cell'''
        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                new_point = tf_util.conv2d(new_point, num_out_channel, [1, 1],
                                           padding='VALID', stride=[1, 1],
                                           bn=use_bn, is_training=is_training,
                                           scope='conv%d' % (i), bn_decay=bn_decay, weight_decay=weight_decay)

        weight = weight_net_hidden(grouped_xyz, [nsample], scope='weight_net', is_training=is_training, bn_decay=bn_decay,
                                   weight_decay=weight_decay)  # [B,N,S,S]
        new_point = tf.transpose(new_point, [0, 1, 3, 2]) # [B,N,C,S]
        new_point = tf.matmul(new_point, weight) # [B,N,C,S]
        new_point = tf_util.conv2d(new_point, mlp[-1], [1, new_point.get_shape()[2].value],
                                   padding='VALID', stride=[1, 1],
                                   bn=use_bn, is_training=is_training,
                                   scope='after_conv', bn_decay=bn_decay, weight_decay=weight_decay) # [B,N,1,C]
        new_point = tf.squeeze(new_point, [2])  # (batch_size, npoints, mlp2[-1]) # [B,N,C]

        new_point = tf.add(new_point, skip_spatial)

        if NL:
            new_point = tf.add(new_point, new_nonlocal_point)

        '''Feature Fushion'''
        new_point = tf_util.conv1d(new_point, mlp[-1], 1,
                                   padding='VALID', stride=1, bn=use_bn, is_training=is_training,
                                   scope='aggregation', bn_decay=bn_decay, weight_decay=weight_decay)


        coord = conv1d(new_point, 128, is_training=is_training, bn=False, bn_decay=bn_decay, scope='coord_layer1')
        coord = conv1d(coord, 64, is_training=is_training, bn=False, bn_decay=bn_decay, scope='coord_layer2')
        coord = conv1d(coord, 3, is_training=is_training, bn=False, bn_decay=bn_decay, scope='coord_layer3',
                       activation_fn=None, weight_decay=0.0)

        new_offset = coord
        if use_sm:
            range_max = 0.5
            new_offset = tf.sigmoid(new_offset) * range_max * 2 - range_max

        return new_xyz,new_offset



def PointDownscale4(xyz, feature, npoint, nsample, mlp, is_training, bn_decay, scope, use_bn=False, use_knn=True, radius=None, as_neighbor=8, NL=True, use_noise=False, use_sm=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        batch_size, num_points, num_channel = feature.get_shape()
        '''Farthest Point Sampling'''
        if num_points == npoint:
            new_xyz = xyz
            new_feature = feature
        else:
            new_xyz, new_feature = sampling(npoint, xyz, feature)

        nsample = 32
        group_xyz, group_feature, idx = grouping(feature, nsample, xyz, new_xyz,use_knn=use_knn,radius=radius)

        #group_feature = tf.reshape(group_feature,[tf.shape(xyz)[0],tf.shape(xyz)[1],-1])
        mlps = [num_channel,num_channel]
        new_feature = group_feature
        for i, c in enumerate(mlps):
            activation = tf.nn.relu if i < len(mlps) - 1 else None
            use_bn = use_bn if i < len(mlps) - 1 else False
            new_feature = conv2d(new_feature, c,
                                 bn=use_bn, is_training=is_training,
                                 scope='mlp1_2_%d' % (i), bn_decay=bn_decay,
                                 activation_fn=activation)

        new_feature = tf.reduce_max(new_feature,axis=2)
        mlps = [num_channel,64,3]

        if use_noise:
            noise = tf.random_normal(shape=[tf.shape(new_feature)[0], tf.shape(new_feature)[1], 16], mean=0.0, stddev=1)
            new_feature = tf.concat(axis=2, values=[new_feature, noise])

        for i, c in enumerate(mlps):
            activation = tf.nn.relu if i < len(mlps) - 1 else None
            use_bn = use_bn if i < len(mlps) - 1 else False
            new_feature = conv1d(new_feature, c,
                                        bn=use_bn, is_training=is_training,
                                        scope='mlp2_%d' % (i), bn_decay=bn_decay,
                                        activation_fn=activation)
        new_offset = new_feature
        if use_sm:
            range_max = 0.5
            new_offset = tf.sigmoid(new_offset) * range_max * 2 - range_max

        return new_xyz,new_offset

def PointDownscale2(xyz, feature, npoint, nsample, mlp, is_training, bn_decay, scope, bn=True, use_knn=True, radius=None, as_neighbor=8, NL=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        batch_size, num_points, num_channel = feature.get_shape()
        '''Farthest Point Sampling'''
        if num_points == npoint:
            new_xyz = xyz
            new_feature = feature
        else:
            new_xyz, new_feature = sampling(npoint, xyz, feature)

        group_xyz, group_feature, idx = grouping(feature, nsample, xyz, new_xyz,use_knn=use_knn,radius=radius)
        nl_channel = mlp[-1]

        nsample, num_channel = group_xyz.get_shape()[-2:]

        shift_group_xyz = group_xyz[:, :, :as_neighbor, :]
        shift_group_points = group_feature[:, :, :as_neighbor, :]
        offset = SampleOffset(shift_group_points, shift_group_xyz, [32, 3], is_training, bn_decay,scope, bn)

        return new_xyz,offset

def shuffle_up(inputs, scale):
    N, C, iH, iW = inputs.size()
    oH = iH * scale
    oW = iW * scale
    oC = C // (scale ** 2)
    output = inputs.view(N, oC, scale, scale, iH, iW)
    output = output.permute(0, 1, 4, 3, 5, 2).contiguous()
    output = output.view(N, oC, oH, oW)

def up_shuffle_layer(feature,is_training, up_ratio=4):
    B, N, C = feature.get_shape()
    outputs = tf.expand_dims(feature, axis=2)
    outputs = conv2d(outputs, up_ratio*C, is_training=is_training, bn=False, bn_decay=None, scope='up_shuffle_layer1')
    outputs = tf.squeeze(outputs, axis=2)

    outputs = tf.reshape(outputs, [B, N, C, up_ratio])
    outputs = tf.transpose(outputs, [0, 1, 3, 2])
    outputs = tf.reshape(outputs, [B, N*up_ratio, C])

    return outputs

def up_shuffle_layer2(feature,is_training, up_ratio=4):
    B, N, C = feature.get_shape()
    outputs = tf.expand_dims(feature, axis=2)
    outputs = conv2d(outputs, up_ratio*C, is_training=is_training, bn=False, bn_decay=None, scope='up_shuffle_layer1')
    outputs = tf.squeeze(outputs, axis=2)

    outputs = tf.reshape(outputs, [B, N, up_ratio, C])
    outputs = tf.reshape(outputs, [B, N*up_ratio, C])

    return outputs

def up_shuffle_layer3(pc, feature,is_training, up_ratio=4, use_bn=False,bn_decay=False,scope="up_shuffle_layer3"):
    B, N, C = feature.get_shape()
    with tf.variable_scope(scope) as sc:
        feature = conv1d(feature, C, is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='up_shuffle_layer0')
        #outputs = tf.expand_dims(feature, axis=2)
        up_feature = EdgeConv(feature, up_ratio*C, k=16, is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='up_shuffle_layer1')
        #outputs = tf.squeeze(outputs, axis=2)

        up_feature = tf.reshape(up_feature, [B, N, up_ratio, C])
        up_feature = tf.reshape(up_feature, [B, N*up_ratio,C])

        up_xyz = tf.tile(tf.expand_dims(pc, axis=2), (1, 1, up_ratio, 1))
        up_xyz = tf.reshape(up_xyz, [B, -1, 3])

        #up_feature = tf.concat([up_xyz, up_feature], axis=-1)

        return up_feature


def up_shuffle_layer3_raw(feature,is_training, up_ratio=4, use_bn=False,bn_decay=False):
    B, N, C = feature.get_shape()

    feature = conv1d(feature, C, is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='up_shuffle_layer0')
    #outputs = tf.expand_dims(feature, axis=2)
    outputs = EdgeConv(feature, up_ratio*C, k=16, is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='up_shuffle_layer1')
    #outputs = tf.squeeze(outputs, axis=2)

    outputs = tf.reshape(outputs, [B, N, up_ratio, C])
    outputs = tf.reshape(outputs, [B, N*up_ratio,C])

    return outputs

def up_shuffle_layer4(pc, feature,is_training, up_ratio=4, use_bn=False,bn_decay=False,scope="up_shuffle_layer4"):
    B, N, C = feature.get_shape()

    with tf.variable_scope(scope) as sc:


        r = up_ratio
        K = 16
        adj_matrix = tf_util.pairwise_distance(feature)
        nn_idx = tf_util.knn(adj_matrix, k=K)
        edge_feat = tf_util.get_edge_feature(feature, nn_idx, k=K) # [B,N,K,2C]
        #edge_pc = tf_util.get_edge_feature(pc, nn_idx, k=K) # [B,N,K,6]
        # [B,N, K,CC]
        BB, NN, KK, CC = edge_feat.get_shape()
        # [B,N,K/r,r*CC]
        temp_edge_feat = conv2d(edge_feat, r*CC, kernel_size=[1, r], stride=[1,r],is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='up_shuffle_layer0')
        # [B,N, r*CC, K/r]
        temp_edge_feat = tf.transpose(temp_edge_feat,[0,1,3,2])
        temp_edge_feat = tf.reshape(temp_edge_feat,[BB,  NN, CC, r, KK//r])
        temp_edge_feat = tf.reshape(temp_edge_feat, [BB, NN, CC, KK])
        # [B,N, KK, CC]
        temp_edge_feat = tf.transpose(temp_edge_feat,[0,1,3,2])
        # [B,N, 2KK, CC]
        merge_feat = tf.concat([edge_feat,temp_edge_feat],axis=2)
        # [B,N, 1, 2*CC]
        merge_feat = conv2d(merge_feat, CC//2*r, kernel_size=[1, 2*KK],is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='up_shuffle_layer1')

        merge_feat = tf.reshape(merge_feat, [BB, NN, r, CC//2])
        merge_feat = tf.reshape(merge_feat, [BB, r*NN, CC//2])

        return merge_feat


def PointShuffle(xyz, feature, nsample, mlp, is_training, bn_decay, scope, use_bn=True, use_knn=True, radius=None, NL=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        batch_size, num_points, num_channel = feature.get_shape()

        K = 16
        group_xyz, group_feature, idx = grouping(feature, K, xyz, xyz,use_knn=use_knn,radius=radius)
        nl_channel = mlp[-1]

        _, num_channel = group_xyz.get_shape()[-2:]

        shift_group_xyz = group_xyz[:, :, :nsample, :]
        shift_group_points = group_feature[:, :, :nsample, :]
        sample_weight = SampleWeights(shift_group_points, shift_group_xyz, [num_channel, num_channel], is_training,
                                      bn_decay, bn=use_bn)
        new_feature = tf.reduce_sum(tf.multiply(shift_group_points, sample_weight), axis=[2])

        return new_feature


def PointShuffle2(xyz, feature, nsample, mlp, is_training, bn_decay, scope, use_bn=False, use_knn=True, radius=None, NL=True,Local=True, use_noise=False, use_sm=True,weight_decay=None,refine_point=False):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:

        batch_size, num_points, num_channel = feature.get_shape()
        '''Farthest Point Sampling'''
        new_xyz = xyz
        new_feat = feature
        K = nsample
        grouped_xyz, grouped_feat, idx = grouping(feature, K, xyz, new_xyz, use_xyz=True, use_knn=use_knn, radius=radius)
        nl_channel = mlp[-1]

        center_xyz = tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
        grouped_xyz_raw = grouped_xyz
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
        grouped_feat = tf.concat([grouped_xyz, grouped_feat], axis=-1)

        if refine_point:
            new_xyz, new_feat = AdaptiveSampling(grouped_xyz, grouped_feat, K, is_training, bn_decay,
                                                    weight_decay, "noise_refine", use_bn)

        '''Point NonLocal Cell'''
        if NL:
            new_nonlocal_point = PointNonLocalCell(feature, tf.expand_dims(new_feat, axis=1),
                                                   [max(32, num_channel // 2), nl_channel],
                                                   is_training, bn_decay, weight_decay, scope, use_bn)

        '''Skip Connection'''
        skip_spatial = tf.reduce_max(grouped_feat, axis=[2])
        skip_spatial = tf_util.conv1d(skip_spatial, mlp[-1], 1, padding='VALID', stride=1,
                                      bn=use_bn, is_training=is_training, scope='skip',
                                      bn_decay=bn_decay, weight_decay=weight_decay)

        '''Point Local Cell'''
        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                grouped_feat = tf_util.conv2d(grouped_feat, num_out_channel, [1, 1],
                                           padding='VALID', stride=[1, 1],
                                           bn=use_bn, is_training=is_training,
                                           scope='conv%d' % (i), bn_decay=bn_decay, weight_decay=weight_decay)

        #norm = tf.norm(grouped_xyz,axis=-1,keep_dims=True)
        #weight_xyz = tf.concat([center_xyz,grouped_xyz, grouped_xyz_raw],axis=-1)
        weight = weight_net_hidden(grouped_xyz, [nsample], scope='weight_net', is_training=is_training, bn_decay=bn_decay,
                                   weight_decay=weight_decay)  # [B,N,S,S]
        grouped_feat = tf.transpose(grouped_feat, [0, 1, 3, 2]) # [B,N,C,S]
        grouped_feat = tf.matmul(grouped_feat, weight) # [B,N,C,S]
        grouped_feat = tf_util.conv2d(grouped_feat, mlp[-1], [1, grouped_feat.get_shape()[2].value],
                                   padding='VALID', stride=[1, 1],
                                   bn=use_bn, is_training=is_training,
                                   scope='after_conv', bn_decay=bn_decay, weight_decay=weight_decay) # [B,N,1,C]
        grouped_feat = tf.squeeze(grouped_feat, [2])  # (batch_size, npoints, mlp2[-1]) # [B,N,C]

        grouped_feat = tf.add(grouped_feat, skip_spatial)

        if NL and Local:
            grouped_feat = tf.add(grouped_feat, new_nonlocal_point)
        elif NL:
            grouped_feat = new_nonlocal_point


        '''Feature Fushion'''
        new_feat = tf_util.conv1d(grouped_feat, mlp[-1], 1,
                                   padding='VALID', stride=1, bn=use_bn, is_training=is_training,
                                   scope='aggregation', bn_decay=bn_decay, weight_decay=weight_decay)

        return new_xyz, new_feat

def coordinate_regressor(feature,is_training, use_bn=False,bn_decay=False,scope="coordinate_regressor",is_off=False, layer=512):

    with tf.variable_scope(scope) as sc:

        coord = conv1d(feature, 256,
                       bn=False, is_training=is_training,
                       scope='fc_layer0', bn_decay=None)

        coord = conv1d(coord, 64,
                       bn=False, is_training=is_training,
                       scope='fc_layer1', bn_decay=None)

        out = conv1d(coord, 3,
                             bn=False, is_training=is_training,
                             scope='fc_layer2', bn_decay=None,
                             activation_fn=None, weight_decay=0.0)

        if is_off:
            range_max = 0.5
            out = tf.sigmoid(out) * range_max * 2 - range_max

        return out



def up_shuffle_layer5(pc, feature,is_training, up_ratio=4, use_bn=False,bn_decay=False,scope="up_shuffle_layer4"):
    B, N, C = feature.get_shape()
    with tf.variable_scope(scope) as sc:
        K = 16

        adj_matrix = tf_util.pairwise_distance(feature)
        nn_idx = tf_util.knn(adj_matrix, k=K)
        edge_feat = tf_util.get_edge_feature(feature, nn_idx, k=K) # [B,N,K,2C]
        edge_pc = tf_util.get_edge_feature(pc, nn_idx, k=K) # [B,N,K,6]

        BB, NN, KK, CC = edge_feat.get_shape()

        w_feat = conv2d(edge_feat, CC,  is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='w_feat')
        w_pc = conv2d(edge_pc,  CC,   is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='w_pc')
        w = w_feat * w_pc
        w = conv2d(w, CC, is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='w')
        w = tf.nn.softmax(w, axis=-2)  # attention map

        # [B,N,K/2,CC]
        temp_edge_feat = conv2d(edge_feat, 2*CC,  kernel_size=[1, 2], stride=[1,2],is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='up_shuffle_layer0')
        # [B,N, 2CC, K/2]
        temp_edge_feat = tf.transpose(temp_edge_feat,[0,1,3,2])
        temp_edge_feat = tf.reshape(temp_edge_feat,[BB,NN,CC, 2,KK/2])
        temp_edge_feat = tf.reshape(temp_edge_feat, [BB, NN, CC, KK])
        # [B,N, KK, CC]
        temp_edge_feat = tf.transpose(temp_edge_feat,[0,1,3,2])
        temp_edge_feat = temp_edge_feat*w

        # [B,N, 2KK, CC]
        merge_feat = tf.concat([edge_feat,temp_edge_feat],axis=2)
        # [B,N, 1, 2*CC]
        merge_feat = conv2d(merge_feat, 2*CC, kernel_size=[1, 2*KK],is_training=is_training, bn=use_bn, bn_decay=bn_decay, scope='up_shuffle_layer1')
        merge_feat = tf.reshape(merge_feat, [BB, NN, 2, CC])
        merge_feat = tf.reshape(merge_feat, [BB, 2*NN, CC])

        return merge_feat


def duplicate_up(pc, feature,is_training, up_ratio=4, use_bn=False,bn_decay=False,scope="up_shuffle_layer4", atten=False,edge=False):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        B, N, C = feature.get_shape()
        net = tf.expand_dims(feature,axis=2)

        # if up_ratio<2:
        #     grid = gen_1d_grid(up_ratio)
        # else:

        grid = gen_grid(up_ratio) #[1,R,2]

        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1,tf.shape(net)[1]])   # [B,R,2*N]

        #grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], tf.shape(net)[1], 1])  # [B,N*R,2]
        # [B,N*R,1,2]
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
        #grid = tf.expand_dims(grid, axis=2)


        net = tf.tile(net, [1, up_ratio, 1, 1])



        net = tf.concat([net, grid], axis=-1)

        if atten:
            net = attention_unit(net, is_training=is_training)

        if edge:
            net = tf.squeeze(net, axis=2)
            net = EdgeConv(net, 256, k=16, is_training=is_training, bn=use_bn, bn_decay=bn_decay,
                           scope='shuffle_layer_0')

            net = EdgeConv(net, 128, k=16, is_training=is_training, bn=use_bn, bn_decay=bn_decay,
                           scope='shuffle_layer_1')
        else:
            net = conv2d(net, 256, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=False, is_training=is_training,
                                     scope='conv1', bn_decay=bn_decay)
            net = conv2d(net, 128, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv2', bn_decay=bn_decay)

        net = tf.squeeze(net,axis=2)

    return net

def duplicate_up_edge(pc, feature,is_training, up_ratio=4, use_bn=False,bn_decay=False,scope="up_shuffle_layer4"):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        B, N, C = feature.get_shape()
        net = tf.expand_dims(feature,axis=2)

        # if up_ratio<2:
        #     grid = gen_1d_grid(up_ratio)
        # else:

        grid = gen_grid(up_ratio) #[1,R,2]

        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1,tf.shape(net)[1]])   # [B,R,2*N]

        #grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], tf.shape(net)[1], 1])  # [B,N*R,2]
        # [B,N*R,1,2]
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
        #grid = tf.expand_dims(grid, axis=2)


        net = tf.tile(net, [1, up_ratio, 1, 1])



        net = tf.concat([net, grid], axis=-1)

        #net = attention_unit(net, is_training=is_training)
        net = tf.squeeze(net, axis=2)
        net = EdgeConv(net, 256, k=16, is_training=is_training, bn=use_bn, bn_decay=bn_decay,
                       scope='shuffle_layer_0')

        net = EdgeConv(net, 128, k=16, is_training=is_training, bn=use_bn, bn_decay=bn_decay,
                       scope='shuffle_layer_1')

        #net = tf.squeeze(net,axis=2)

    return net

def duplicate_up2(pc, feature,is_training, up_ratio=4, use_bn=False,bn_decay=False,scope="up_shuffle_layer4", patch_num=256):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = tf.expand_dims(feature,axis=2)
        B,N,C  = feature.get_shape()
        grid = gen_grid(patch_num*up_ratio) #[N,2]
        grid = tf.tile(tf.expand_dims(grid, 0), [B, 1, 1])  # [batch_size, num_point*4, 2])
        grid = tf.expand_dims(grid,axis=2)

        net = tf.tile(net, [1, up_ratio, 1, 1])
        net = tf.concat([net, grid], axis=-1)

        #net = attention_unit(net, is_training=is_training)

        net = conv2d(net, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)
        net = tf.squeeze(net,axis=2)


    return net

def PointUpscale(xyz, feature, npoint, is_training, bn_decay=None, scope="PointUpscale", bn=False):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        B, N, C = feature.get_shape()
        #up_feature = up_shuffle_layer(feature,is_training=is_training,up_ratio=npoint//N)
        #up_feature = up_shuffle_layer2(feature,is_training=is_training,up_ratio=npoint//N)
        up_feature = up_shuffle_layer3(feature,is_training=is_training,up_ratio=npoint//N)

        coord = up_feature
        coord = conv1d(coord, C, is_training=is_training, bn=False, bn_decay=bn_decay, scope='coord_layer0')
        coord = conv1d(coord, 128, is_training=is_training, bn=False, bn_decay=bn_decay, scope='coord_layer1')
        coord = conv1d(coord, 64, is_training=is_training, bn=False, bn_decay=bn_decay, scope='coord_layer2')
        coord = conv1d(coord, 3, is_training=is_training, bn=False, bn_decay=bn_decay, scope='coord_layer3', activation_fn=None, weight_decay=0.0)

        return coord



##################################################################################
# Back projection Blocks
##################################################################################

def shuffle_down(inputs, scale):
    N, C, iH, iW = inputs.size()
    oH = iH // scale
    oW = iW // scale
    output = inputs.view(N, C, oH, scale, oW, scale)
    output = output.permute(0, 1, 5, 3, 2, 4).contiguous()
    return output.view(N, -1, oH, oW)

def shuffle_up(inputs, scale):
    N, C, iH, iW = inputs.size()
    oH = iH * scale
    oW = iW * scale
    oC = C // (scale ** 2)
    output = inputs.view(N, oC, scale, scale, iH, iW)
    output = output.permute(0, 1, 4, 3, 5, 2).contiguous()
    output = output.view(N, oC, oH, oW)

    return output

def PointShuffler(inputs, scale=2):
    #inputs: B x N x 1 X C
    #outputs: B x N*scale x 1 x C//scale
    outputs = tf.reshape(inputs,[tf.shape(inputs)[0],tf.shape(inputs)[1],1,tf.shape(inputs)[3]//scale,scale])
    outputs = tf.transpose(outputs,[0, 1, 4, 3, 2])

    outputs = tf.reshape(outputs,[tf.shape(inputs)[0],tf.shape(inputs)[1]*scale,1,tf.shape(inputs)[3]//scale])

    return outputs


def up_block(inputs, up_ratio, scope='up_block', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = inputs
        dim = inputs.get_shape()[-1]
        out_dim = dim*up_ratio
        grid = gen_grid(up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1,tf.shape(net)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
            #grid = tf.expand_dims(grid, axis=2)

        net = tf.tile(net, [1, up_ratio, 1, 1])
        net = tf.concat([net, grid], axis=-1)

        net = attention_unit(net, is_training=is_training)

        net = conv2d(net, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)

    return net

def down_block(inputs,up_ratio,scope='down_block',is_training=True,bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = inputs
        net = tf.reshape(net,[tf.shape(net)[0],up_ratio,-1,tf.shape(net)[-1]])
        net = tf.transpose(net, [0, 2, 1, 3])

        net = conv2d(net, 256, [1, up_ratio],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)

    return net

def feature_extraction_down(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):

    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):

        use_bn = False
        use_ibn = False
        growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 32, scope='layer0', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay)

        l0_features = conv2d(l0_features, 64, scope='layer1', is_training=is_training, bn=use_bn,
                             bn_decay=bn_decay)

    return l0_features

def feature_extraction_up(inputs, scope='feature_extraction2', growth_rate=24, is_training=True, bn_decay=None,use_bn=False):

    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):

        #use_bn = False
        #use_ibn = False
        #growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer1", is_training=is_training, bn=use_bn,
                                                  bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84

        l2_features = conv1d(l1_features, comp, 1,  # 24
                                     padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144

        l3_features = conv1d(l2_features, comp, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204

        l4_features = conv1d(l3_features, comp, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264

        #l4_features = tf.expand_dims(l4_features, axis=2)

    return l4_features


def feature_extraction_GCN(inputs, scope='feature_extraction2', growth_rate=24, is_training=True, bn_decay=None,use_bn=False,dense_block=2):

    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):

        #use_bn = False
        #use_ibn = False
        #growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2) #24

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer1", is_training=is_training, bn=use_bn,
                                                  bn_decay=bn_decay)
        out_feat = tf.concat([l1_features, l0_features], axis=-1)  #In+ (comp + 3*F) =  24 + (24 + 24*3) = 120

        if dense_block > 1:
            l2_features = conv1d(out_feat, comp, 1,  # 48
                                         padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn,
                                         bn_decay=bn_decay)
            l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                      scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
            out_feat = tf.concat([l2_features, out_feat], axis=-1) #In+ (2F + 3*F) =  120 + (48 + 24*3) = 240

        if dense_block > 2:
            l3_features = conv1d(out_feat, comp, 1,  # 48
                                         padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn,
                                         bn_decay=bn_decay)  # 48
            l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                      scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
            out_feat = tf.concat([l3_features, out_feat], axis=-1)  #In+ (2F + 3*F) =  240 + (48 + 24*3) = 360

        if dense_block > 3:
            l4_features = conv1d(out_feat, comp, 1,  # 48
                                         padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn,
                                         bn_decay=bn_decay)  # 48
            l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                      scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
            out_feat = tf.concat([l4_features, out_feat], axis=-1)  #In+ (2F + 3*F) =  240 + (48 + 24*3) = 480

        #l4_features = tf.expand_dims(l4_features, axis=2)

    return out_feat

def feature_extraction_up2(inputs, scope='feature_extraction2', growth_rate=24, is_training=True, bn_decay=None):

    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):

        use_bn = False
        use_ibn = False
        #growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer1", is_training=is_training, bn=use_bn,
                                                  bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84

        l2_features = conv1d(l1_features, comp, 1,  # 24
                                     padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144

        l3_features = conv1d(l2_features, comp, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204

        l4_features = conv1d(l3_features, comp, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264

        #l4_features = tf.expand_dims(l4_features, axis=2)

    return l4_features


def contract_expand_operation(inputs,up_ratio,scope="contraction"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        B, N, C = inputs.get_shape()
        net = tf.expand_dims(inputs,axis=2)
        net = tf.reshape(net, [tf.shape(net)[0], up_ratio, -1, tf.shape(net)[-1]])
        net = tf.transpose(net, [0, 2, 1, 3])

        net = conv2d(net,C,
                           [1, up_ratio],
                           scope='down_conv1',
                           stride=[1, 1],
                           padding='VALID',
                           weight_decay=0.00001,
                           activation_fn=tf.nn.relu)
        net = conv2d(net,
                           C*up_ratio,
                           [1, 1],
                           scope='down_conv2',
                           stride=[1, 1],
                           padding='VALID',
                           weight_decay=0.00001,
                           activation_fn=tf.nn.relu)

        net = tf.reshape(net, [tf.shape(net)[0], -1, up_ratio,C])
        net = conv2d(net,
                           C,
                           [1, 1],
                           scope='down_conv3',
                           stride=[1, 1],
                           padding='VALID',
                           weight_decay=0.00001,
                           activation_fn=tf.nn.relu)
        net=tf.reshape(net,[tf.shape(net)[0], -1, C])
        return net

def up_projection_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs = tf.expand_dims(inputs,axis=2)
        L = conv2d(inputs, 128, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='conv0', bn_decay=bn_decay)

        H0 = up_block(L,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='up_0')

        L0 = down_block(H0,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='down_0')
        E0 = L0-L
        H1 = up_block(E0,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='up_1')
        H2 = H0+H1
    return H2

def weight_learning_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape().as_list()[-1]
        grid = gen_1d_grid(tf.reshape(up_ratio,[]))

        out_dim = dim * up_ratio

        ratios = tf.tile(tf.expand_dims(up_ratio,0),[1,tf.shape(grid)[1]])
        grid_ratios = tf.concat([grid,tf.cast(ratios,tf.float32)],axis=1)
        weights = tf.tile(tf.expand_dims(tf.expand_dims(grid_ratios,0),0),[tf.shape(inputs)[0],tf.shape(inputs)[1], 1, 1])
        weights.set_shape([None, None, None, 2])
        weights = conv2d(weights, dim, [1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='conv_1', bn_decay=None)


        weights = conv2d(weights, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training,
                         scope='conv_2', bn_decay=None)
        weights = conv2d(weights, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training,
                         scope='conv_3', bn_decay=None)

        s = tf.matmul(hw_flatten(inputs), hw_flatten(weights), transpose_b=True)  # # [bs, N, N]

    return tf.expand_dims(s,axis=2)


def coordinate_reconstruction_unit(inputs,scope="reconstruction",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        coord = conv2d(inputs, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer1', bn_decay=None)

        coord = conv2d(coord, 3, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer2', bn_decay=None,
                           activation_fn=None, weight_decay=0.0)
        outputs = tf.squeeze(coord, [2])

        return outputs


def attention_unit(inputs, scope='attention_unit',is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        #inputs = tf.expand_dims(inputs,axis=2)
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = conv2d(inputs,layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f', bn_decay=None)

        g = conv2d(inputs, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g', bn_decay=None)

        h = conv2d(inputs, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h', bn_decay=None)


        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

        #x = tf.squeeze(x,axis=2)

    return x


##################################################################################
# Other function
##################################################################################
def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


# def conv1d(inputs,
#            num_output_channels,
#            kernel_size=1,
#            scope=None,
#            stride=1,
#            padding='VALID',
#            use_xavier=True,
#            stddev=1e-3,
#            weight_decay=0.00001,
#            activation_fn=tf.nn.relu,
#            bn=False,
#            ibn=False,
#            bn_decay=None,
#            use_bias=True,
#            is_training=None,
#            reuse=None):
#     """ 1D convolution with non-linear operation.
#
#     Args:
#         inputs: 3-D tensor variable BxHxWxC
#         num_output_channels: int
#         kernel_size: int
#         scope: string
#         stride: a list of 2 ints
#         padding: 'SAME' or 'VALID'
#         use_xavier: bool, use xavier_initializer if true
#         stddev: float, stddev for truncated_normal init
#         weight_decay: float
#         activation_fn: function
#         bn: bool, whether to use batch norm
#         bn_decay: float or float tensor variable in [0,1]
#         is_training: bool Tensor variable
#
#     Returns:
#         Variable tensor
#     """
#     with tf.variable_scope(scope, reuse=reuse):
#         if use_xavier:
#             initializer = tf.contrib.layers.xavier_initializer()
#         else:
#             initializer = tf.truncated_normal_initializer(stddev=stddev)
#
#         outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
#                                    kernel_initializer=initializer,
#                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
#                                        weight_decay),
#                                    bias_regularizer=tf.contrib.layers.l2_regularizer(
#                                        weight_decay),
#                                    use_bias=use_bias, reuse=None)
#         assert not (bn and ibn)
#         if bn:
#             outputs = tf.layers.batch_normalization(
#                 outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
#             # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
#         if ibn:
#             outputs = instance_norm(outputs, is_training)
#
#         if activation_fn is not None:
#             outputs = activation_fn(outputs)
#
#         return outputs

# def conv2d(inputs,
#            num_output_channels,
#            kernel_size=[1,1],
#            scope=None,
#            stride=[1, 1],
#            padding='VALID',
#            use_xavier=True,
#            stddev=1e-3,
#            weight_decay=0.00001,
#            activation_fn=tf.nn.relu,
#            bn=False,
#            ibn = False,
#            bn_decay=None,
#            use_bias = True,
#            is_training=None,
#            reuse=tf.AUTO_REUSE):
#   """ 2D convolution with non-linear operation.
#
#   Args:
#     inputs: 4-D tensor variable BxHxWxC
#     num_output_channels: int
#     kernel_size: a list of 2 ints
#     scope: string
#     stride: a list of 2 ints
#     padding: 'SAME' or 'VALID'
#     use_xavier: bool, use xavier_initializer if true
#     stddev: float, stddev for truncated_normal init
#     weight_decay: float
#     activation_fn: function
#     bn: bool, whether to use batch norm
#     bn_decay: float or float tensor variable in [0,1]
#     is_training: bool Tensor variable
#
#   Returns:
#     Variable tensor
#   """
#   with tf.variable_scope(scope,reuse=reuse) as sc:
#       if use_xavier:
#           initializer = tf.contrib.layers.xavier_initializer()
#       else:
#           initializer = tf.truncated_normal_initializer(stddev=stddev)
#
#       outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
#                                  kernel_initializer=initializer,
#                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
#                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
#                                  use_bias=use_bias,reuse=None)
#       assert not (bn and ibn)
#       if bn:
#           outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
#           #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
#       if ibn:
#           outputs = instance_norm(outputs,is_training)
#
#
#       if activation_fn is not None:
#         outputs = activation_fn(outputs)
#
#       return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

def dense_conv0(feature, n=3,growth_rate=64, k=16, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = conv2d(y, growth_rate, scope='l%d' % i, **kwargs)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y,idx


def dense_conv(feature, n=3,growth_rate=64, k=16, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:

                y = tf.concat([
                    conv2d(y, growth_rate, scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y,idx

def dense_conv2(feature, growth_rate, n, k, scope, idx=None, **kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=idx)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y, idx



def EdgeConv(inputs, C=64, k=16, scope=None, is_training=None, bn=False, bn_decay=None, activation=tf.nn.relu):
  '''
    EdgeConv layer:
      Wang, Y, Yongbin S, Ziwei L, Sanjay S, Michael B, Justin S.
      "Dynamic graph cnn for learning on point clouds."
      arXiv:1801.07829 (2018).
  '''

  adj_matrix = tf_util.pairwise_distance(inputs)
  nn_idx = tf_util.knn(adj_matrix, k=k)

  edge_features = tf_util.get_edge_feature(inputs, nn_idx, k)
  out = conv2d(edge_features, C,
               bn=bn, is_training=is_training,
               scope=scope, bn_decay=bn_decay,
               activation_fn=activation)
  vertex_features = tf.reduce_max(out, axis=-2)

  return vertex_features

def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)


def tf_covariance(data):
    ## x: [batch_size, num_point, k, 3]
    batch_size = data.get_shape()[0].value
    num_point = data.get_shape()[1].value

    mean_data = tf.reduce_mean(data, axis=2, keep_dims=True)  # (batch_size, num_point, 1, 3)
    mx = tf.matmul(tf.transpose(mean_data, perm=[0, 1, 3, 2]), mean_data)  # (batch_size, num_point, 3, 3)
    vx = tf.matmul(tf.transpose(data, perm=[0, 1, 3, 2]), data) / tf.cast(tf.shape(data)[0], tf.float32)  # (batch_size, num_point, 3, 3)
    data_cov = tf.reshape(vx - mx, shape=[batch_size, num_point, -1])

    return data_cov



def add_scalar_summary(name, value,collection='train_summary'):
    tf.summary.scalar(name, value, collections=[collection])
def add_hist_summary(name, value,collection='train_summary'):
    tf.summary.histogram(name, value, collections=[collection])

def add_train_scalar_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])

def add_train_hist_summary(name, value):
    tf.summary.histogram(name, value, collections=['train_summary'])

def add_train_image_summary(name, value):
    tf.summary.image(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update
