# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:11 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : data_loader.py

import numpy as np
import h5py
import queue
import threading
from Common import point_operation
import os
# def normalize_point_cloud(input):
#     if len(input.shape)==2:
#         axis = 0
#     elif len(input.shape)==3:
#         axis = 1
#     centroid = np.mean(input, axis=axis, keepdims=True)
#     input = input - centroid
#     furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
#     input = input / furthest_distance
#     return input, centroid,furthest_distance


def normalize_point_cloud(inputs):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    #print("shape",input.shape)
    C = inputs.shape[-1]
    pc = inputs

    centroid = np.mean(pc, axis=1, keepdims=True)
    pc = inputs - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / furthest_distance
    return pc,centroid,furthest_distance

def batch_sampling(input_data,num):
    B,N,C = input_data.shape
    out_data = np.zeros([B,num,C])
    for i in range(B):
        idx = np.arange(N)
        np.random.shuffle(idx)
        idx = idx[:num]
        out_data[i,...] = input_data[i,idx]
    return out_data

def load_h5_data(h5_filename='', in_num=0,out_num=0, random=True,normalized=True):

    print("h5_filename : ",h5_filename)
    if random:
        f = h5py.File(h5_filename)
        input = f['poisson_%d'%out_num][:]
        gt = f['poisson_%d'%out_num][:]
    else:
        print("Do not randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_%d' % in_num][:]
        gt = f['poisson_%d' % out_num][:]

    #name = f['name'][:]
    assert len(input) == len(gt)

    data_radius = np.ones(shape=(len(input)))
    if normalized:
        gt,centroid,furthest_distance = normalize_point_cloud(gt)
        input = input - centroid
        input = input / furthest_distance

    print("total %d samples" % (len(input)))
    return input, gt, data_radius


class Fetcher(threading.Thread):
    def __init__(self, opts,augment=True, shuffle=True):
        super(Fetcher,self).__init__()
        self.opts = opts
        self.random = self.opts.random
        in_num = self.opts.patch_num_point
        out_num = in_num * self.opts.up_ratio
        h5_file = os.path.join(self.opts.data_dir,"PUGAN_poisson_%d_poisson_%d.h5"%(in_num,out_num))
        self.input_data, self.gt_data, self.radius_data = load_h5_data(h5_file,in_num,out_num,random=self.random)
        self.batch_size = self.opts.batch_size
        self.length = self.input_data.shape[0]
        self.patch_num_point = self.opts.patch_num_point
        self.num_batches = self.length//self.batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.reset()

    def __len__(self):
        return self.length

    def reset(self):
        self.idxs = np.arange(0, self.length)
        if self.shuffle:
            np.random.shuffle(self.idxs)
            self.input_data = self.input_data[self.idxs]
            self.gt_data = self.gt_data[self.idxs]
            # self.shape_names = self.shape_names[self.idxs]
            # self.labels = self.labels[self.idxs]

        self.num_batches = (self.length + self.batch_size - 1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self):
        ''' returned dimension may be smaller than self.batch_size '''
        self.batch_idx += 1

        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx + 1) * self.batch_size, self.length)
        bsize = end_idx - start_idx

        batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()
        batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()
        radius = self.radius_data[start_idx:end_idx].copy()

        if self.random:
            new_batch_input = np.zeros((self.batch_size, self.patch_num_point, batch_input_data.shape[2]))
            for i in range(self.batch_size):
                idx = point_operation.nonuniform_sampling(self.input_data.shape[1], sample_num=self.patch_num_point)
                new_batch_input[i, ...] = batch_input_data[i][idx]
            batch_input_data = new_batch_input

        if self.augment:
            batch_input_data = point_operation.jitter_perturbation_point_cloud(batch_input_data,
                                                                               sigma=self.opts.jitter_sigma,
                                                                               clip=self.opts.jitter_max)
            batch_input_data, batch_data_gt = point_operation.rotate_point_cloud_and_gt(batch_input_data, batch_data_gt)
            batch_input_data, batch_data_gt, scales = point_operation.random_scale_point_cloud_and_gt(batch_input_data,
                                                                                                      batch_data_gt,
                                                                                                      scale_low=0.8,
                                                                                                      scale_high=1.2)
            # batch_data = point_operation.random_point_dropout(batch_data)

        return batch_input_data, batch_data_gt,radius







