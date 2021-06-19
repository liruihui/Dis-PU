# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 10:26 AM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : loss_utils.py

import tensorflow as tf
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from Common import pc_util

sys.path.append(os.path.join(os.getcwd(),"tf_ops/sampling"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/nn_distance"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/approxmatch"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/grouping"))
sys.path.append(os.path.join(os.getcwd(),"tf_ops/interpolation"))

import tf_nndistance
import tf_approxmatch
from tf_sampling import gather_point, farthest_point_sample
from tf_grouping import query_ball_point, group_point, knn_point,knn_point_2

#from tf_ops.nn_distance import tf_nndistance
#from tf_ops.approxmatch import tf_approxmatch
# from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point,knn_point_2
# from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
import numpy as np
import math


def pc_distance(pcd1,pcd2,dis_type='EMD',radius=1):
    if dis_type == 'CD':
        return chamfer(pcd1,pcd2,radius=radius)
    else:
        return earth_mover(pcd1,pcd2,radius=radius)

def classify_loss(pre_label,label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_label, labels=label)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss

def chamfer(pred, gt, radius=1.0, forward_weight=1.0, threshold=None, return_hd=False):
    """
    pred: BxNxC,
    label: BxN,
    forward_weight: relative weight for forward_distance
    """
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
    if threshold is not None:
        forward_threshold = tf.reduce_mean(dists_forward, keepdims=True, axis=1) * threshold
        backward_threshold = tf.reduce_mean(dists_backward, keepdims=True, axis=1) * threshold
        # only care about distance within threshold (ignore strong outliers)
        dists_forward = tf.where(dists_forward < forward_threshold, dists_forward, tf.zeros_like(dists_forward))
        dists_backward = tf.where(dists_backward < backward_threshold, dists_backward, tf.zeros_like(dists_backward))
    # dists_forward is for each element in gt, the closest distance to this element
    dists_forward = tf.reduce_mean(dists_forward, axis=1)
    dists_backward = tf.reduce_mean(dists_backward, axis=1)
    CD_dist = forward_weight * dists_forward + dists_backward
    CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_mean(CD_dist_norm)
    return cd_loss


def hausdorff_loss(pred, gt, radius=1.0, forward_weight=1.0, threshold=None):
    """
    pred: BxNxC,
    label: BxN,
    forward_weight: relative weight for forward_distance
    """
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
    # only care about distance within threshold (ignore strong outliers)
    if threshold is not None:
        dists_forward = tf.where(dists_forward < threshold, dists_forward, tf.zeros_like(dists_forward))
        dists_backward = tf.where(dists_backward < threshold, dists_backward, tf.zeros_like(dists_backward))
    # dists_forward is for each element in gt, the closest distance to this element
    dists_forward = tf.reduce_max(dists_forward, axis=1)
    dists_backward = tf.reduce_max(dists_backward, axis=1)
    CD_dist = forward_weight * dists_forward + dists_backward
    CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_max(CD_dist_norm)
    return cd_loss

def get_Geometric_Loss(predictedPts, targetpoints, return_all=False):
    # calculate shape loss
    square_dist = pairwise_l2_norm2_batch(targetpoints, predictedPts)
    dist = tf.sqrt(square_dist)
    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)
    densityWeight = 1.0
    # calculate density loss
    square_dist2 = pairwise_l2_norm2_batch(targetpoints, targetpoints)
    dist2 = tf.sqrt(square_dist2)

    nnk = 8
    knndis = tf.nn.top_k(tf.negative(dist), k=nnk)
    knndis2 = tf.nn.top_k(tf.negative(dist2), k=nnk)
    densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))

    gt_offsets = knndis2.values
    gt_offsets_norm = tf.reduce_sum(gt_offsets**2,keep_dims=True)
    gt_offsets_ = gt_offsets / (gt_offsets_norm + 1e-8)

    pt_offsets = knndis.values
    pt_offsets_norm = tf.reduce_sum(pt_offsets**2,keep_dims=True)
    pt_offsets_ = pt_offsets / (pt_offsets_norm + 1e-8)
    direction_diff = tf.reduce_sum(gt_offsets_ * pt_offsets_)  # (N)

    # gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # (N), float
    # gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
    # pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
    # pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
    # direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)  # (N)
    # offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

    # neighbour1 = get_knn_neighbour(targetpoints, knndis.indices)
    # cov1 = tf_covariance2(neighbour1)
    # neighbour2 = get_knn_neighbour(predictedPts, knndis2.indices)
    # cov2 = tf_covariance2(neighbour2)
    # densityLoss_2 = tf.reduce_mean(tf.abs(cov1 - cov2))

    # if return_all:
    #     densityLoss_3 = tf.reduce_mean(tf.abs(tf.reduce_sum(cov1, axis=-1) - tf.reduce_sum(cov2, axis=-1)))



    return shapeLoss, densityLoss,direction_diff


def pairwise_distance(x, y, scope=None):
    """Compute pairwise distance of a point cloud.

    Args:
      x: tensor (batch_size, num_points, num_dims)
      y: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        y_T = tf.transpose(y, perm=[0, 2, 1])
        x_y = -2 * tf.matmul(x, y_T)

        x_square = tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True)
        y_square = tf.reduce_sum(tf.square(y), axis=-1, keep_dims=True)
        y_square_T = tf.transpose(y_square, perm=[0, 2, 1])
        return x_square + x_y + y_square_T

def pairwise_l2_norm2_batch(x, y, scope=None, num=2048):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 2)

        return square_dist

def earth_mover(pcd1, pcd2,radius=1.0):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    cost = cost/radius
    return tf.reduce_mean(cost / num_points)

def py_uniform_loss(points,idx,pts_cn,radius):
    #print(type(idx))
    B,N,C = points.shape
    _,npoint,nsample = idx.shape
    uniform_vals = []
    for i in range(B):
        point = points[i]
        for j in range(npoint):
            number = pts_cn[i,j]
            coverage = np.square(number - nsample) / nsample
            if number<5:
                uniform_vals.append(coverage)
                continue
            _idx = idx[i, j, :number]
            disk_point = point[_idx]
            if disk_point.shape[0]<0:
                pair_dis = pc_util.get_pairwise_distance(disk_point)#(batch_size, num_points, num_points)
                nan_valid = np.where(pair_dis<1e-7)
                pair_dis[nan_valid]=0
                pair_dis = np.squeeze(pair_dis, axis=0)
                pair_dis = np.sort(pair_dis, axis=1)
                shortest_dis = np.sqrt(pair_dis[:, 1])
            else:
                shortest_dis = pc_util.get_knn_dis(disk_point,disk_point,2)
                shortest_dis = shortest_dis[:,1]
            disk_area = math.pi * (radius ** 2) / disk_point.shape[0]
            #expect_d = math.sqrt(disk_area)
            expect_d = np.sqrt(2 * disk_area / 1.732)  # using hexagon
            dis = np.square(shortest_dis - expect_d) / expect_d
            uniform_val = coverage * np.mean(dis)

            uniform_vals.append(uniform_val)

    uniform_dis = np.array(uniform_vals).astype(np.float32)

    uniform_dis = np.mean(uniform_dis)
    return uniform_dis

#whole version, slower
def get_uniform_loss2(pcd, percentages=[0.002,0.004,0.006,0.008,0.010,0.012,0.015], radius=1.0):
    B,N,C = pcd.get_shape().as_list()
    npoint = int(N * 0.05)
    loss=[]
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        #print(npoint,nsample)
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)#(batch_size, npoint, nsample)

        uniform_val = tf.py_func(py_uniform_loss, [pcd, idx, pts_cnt, r], tf.float32)

        loss.append(uniform_val*math.sqrt(p*100))
    return tf.add_n(loss)/len(percentages)


#[0.004,0.006,0.008,0.010,0.012]
#[0.006,0.008,0.010,0.012,0.015]
#[0.010,0.012,0.015,0.02,0.025]
#simplfied version, faster
def get_uniform_loss(pcd, percentages=[0.004,0.006,0.008,0.010,0.012], radius=1.0):
    B,N,C = pcd.get_shape().as_list()
    npoint = int(N * 0.05)
    loss=[]
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        disk_area = math.pi *(radius ** 2) * p/nsample
        #print(npoint,nsample)
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)#(batch_size, npoint, nsample)

        #expect_len =  tf.sqrt(2*disk_area/1.732)#using hexagon
        expect_len = tf.sqrt(disk_area)  # using square

        grouped_pcd = group_point(pcd, idx)
        grouped_pcd = tf.concat(tf.unstack(grouped_pcd, axis=1), axis=0)

        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = tf.sqrt(tf.abs(uniform_dis+1e-8))
        uniform_dis = tf.reduce_mean(uniform_dis,axis=[-1])
        uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = tf.reshape(uniform_dis, [-1])

        mean, variance = tf.nn.moments(uniform_dis, axes=0)
        mean = mean*math.pow(p*100,2)
        #nothing 4
        loss.append(mean)
    return tf.add_n(loss)/len(percentages)



def get_repulsion_loss(pred, nsample=20, radius=0.07, knn=False, use_l1=False, h=0.001):

    if knn:
        _, idx = knn_point_2(nsample, pred, pred)
        pts_cnt = tf.constant(nsample, shape=(30, 1024))
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    # get the uniform loss
    if use_l1:
        dists = tf.reduce_sum(tf.abs(grouped_pred), axis=-1)
    else:
        dists = tf.reduce_sum(grouped_pred ** 2, axis=-1)

    val, idx = tf.nn.top_k(-dists, 5)
    val = val[:, :, 1:]  # remove the first one

    if use_l1:
        h = np.sqrt(h)*2
    print(("h is ", h))

    val = tf.maximum(0.0, h + val)  # dd/np.sqrt(n)
    repulsion_loss = tf.reduce_mean(val)
    return repulsion_loss

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(d_real, d_fake, gan_type='lsgan'):
    real_loss = tf.reduce_mean(tf.square(d_real - 1.0))
    fake_loss = tf.reduce_mean(tf.square(d_fake))

    loss = 0.5*(real_loss + fake_loss)

    return loss

def generator_loss(d_fake):
    fake_loss = tf.reduce_mean(tf.square(d_fake - 1.0))
    return fake_loss


def discriminator_loss_(D, input_real, input_fake, Ra=False, gan_type='lsgan'):
    real = D(input_real)
    fake = D(input_fake)
    real_loss = tf.reduce_mean(tf.square(real - 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake))

    loss = 0.5*(real_loss + fake_loss)

    return loss

def generator_loss_(D,input_fake):
    fake = D(input_fake)
    fake_loss = tf.reduce_mean(tf.square(fake - 1.0))
    return fake_loss


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss
