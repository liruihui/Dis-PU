# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:04 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import warnings
warnings.filterwarnings('ignore')
import os,sys
import os.path as osp
import tensorflow as tf
from DisPU.generator import Generator
from DisPU.generator import Generator as Generator2

from DisPU.discriminator import Discriminator
from Common.visu_utils import plot_pcd_three_views,point_cloud_three_views
from Common.ops import add_scalar_summary,add_hist_summary
from DisPU.dataset import Fetcher
from Common import model_utils
from Common import pc_util
from Common import loss_utils
from Common.loss_utils import pc_distance,get_uniform_loss,get_repulsion_loss,discriminator_loss,generator_loss
#from tf_ops.sampling.tf_sampling import farthest_point_sample,gather_point
sys.path.append(os.path.join(os.getcwd(),"tf_ops/sampling"))
from tf_sampling import gather_point, farthest_point_sample
import logging
from tqdm import tqdm
from glob import glob
import math
from time import time
from termcolor import colored
import numpy as np
from Common.utils import AverageMeter
from Common.loss_utils import chamfer,earth_mover

MODEL_SAVER_ID = "models.ckpt"


class Model(object):
  def __init__(self,opts, name='Cascaded-PU'):
      self.opts = opts
      self.graph = tf.get_default_graph()
      self.name = name

  def allocate_placeholders(self):
      self.epoch = tf.get_variable("epoch", [], initializer=tf.constant_initializer(0), trainable=False)
      self.increment_epoch = self.epoch.assign_add(tf.constant(1.0))
      self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
      self.input = tf.placeholder(tf.float32, shape=[self.opts.batch_size, self.opts.patch_num_point, 3])
      self.gt = tf.placeholder(tf.float32, shape=[self.opts.batch_size, int(self.opts.up_ratio * self.opts.patch_num_point), 3])
      self.pc_radius = tf.placeholder(tf.float32, shape=[self.opts.batch_size])
      feq = 10
      # self.weight_fine = tf.train.piecewise_constant(self.epoch, [feq * 1.0, feq * 2.0, feq * 3.0],
      #                                                   [0.01, 0.5, 2.0, 5.0], 'weight_fine')

      feq = 10
      self.weight_fine = tf.train.piecewise_constant(self.epoch, [feq * 1.0, feq * 2.0, feq * 3.0],
                                                     [0.01, 0.1, 0.5, 1.0], 'weight_fine')

  def backup(self):
      source_folder = os.path.join(os.getcwd(), "DisPU")
      common_folder = os.path.join(os.getcwd(), "Common")

      os.system("cp %s/configs.py '%s/configs.py'" % (source_folder, self.opts.log_dir))
      os.system("cp %s/model.py '%s/model.py'" % (source_folder, self.opts.log_dir))
      os.system("cp %s/generator2.py '%s/generator2.py'" % (source_folder, self.opts.log_dir))
      os.system("cp %s/dataset.py '%s/dataset.py'" % (source_folder, self.opts.log_dir))
      os.system("cp %s/loss_utils.py '%s/loss_utils.py'" % (common_folder, self.opts.log_dir))
      os.system("cp %s/ops2.py '%s/ops2.py'" % (common_folder, self.opts.log_dir))

  def build_model(self):

      self.allocate_placeholders()

      self.G = Generator2(self.opts,self.is_training,name='generator')
      self.D = Discriminator(self.opts, self.is_training, name='discriminator')

      # X -> Y
      self.coarse, self.fine = self.G(self.input)
      self.dis_coarse_cd = 1000.0 * chamfer(self.coarse, self.gt, radius=self.pc_radius)
      self.dis_coarse_hd = 100.0 * loss_utils.hausdorff_loss(self.coarse, self.gt, radius=self.pc_radius)
      #self.dis_fine_emd = 10.0 * earth_mover(self.fine, self.gt, radius=self.pc_radius)
      self.dis_fine_cd = 1000.0 * chamfer(self.fine, self.gt, radius=self.pc_radius)
      self.dis_fine_hd = 100.0 * loss_utils.hausdorff_loss(self.fine, self.gt, radius=self.pc_radius)
      #self.shapeLoss, self.densityLoss, self.directionLoss = loss_utils.get_Geometric_Loss(self.fine,self.gt)

      if self.opts.use_repulse:
          self.repulsion_loss = self.opts.repulsion_w*get_repulsion_loss(self.fine)
      else:
          self.repulsion_loss = 0
      self.uniform_loss = 10.0 * get_uniform_loss(self.fine)
      self.pu_loss = self.dis_coarse_cd + self.weight_fine * (self.dis_fine_cd) + self.repulsion_loss + tf.losses.get_regularization_loss()

      self.total_gen_loss = self.pu_loss

      self.use_gan = True
      if self.use_gan:
          print("-------------use gan-------------")
      else:
          print("-------------not use gan-------------")
      logit = self.D(self.fine,self.gt)
      self.real_logit = logit[:, :, 0, :]
      self.fake_logit = logit[:, :, 1, :]

      # self.fake_logit = self.D(self.fine,self.gt)
      # self.real_logit = self.D(self.gt,self.gt)

      self.G_gan_loss = generator_loss(self.fake_logit)
      self.D_loss = discriminator_loss(self.real_logit, self.fake_logit)

      if self.use_gan:
          self.total_gen_loss = self.total_gen_loss + self.G_gan_loss

      self.setup_optimizer()
      self.summary_all()

      self.visualize_ops = [self.input[0], self.coarse[0], self.fine[0], self.gt[0]]
      self.visualize_titles = ['input_x', 'coarse', 'fine', 'gt']

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
      self.sess.run(tf.global_variables_initializer())
      self.step = self.sess.run(self.global_step)

      self.backup()


  def build_model_test(self,final_ratio=4,step_ratio=4):
      self.allocate_placeholders()

      is_training = tf.placeholder_with_default(False, shape=[], name='is_training')

      self.G = Generator2(self.opts,is_training,name='generator')

      # X -> Y
      self.coarse, self.pred_pc = self.G(self.input)
      for i in range(round(math.pow(final_ratio, 1 / step_ratio)) - 1):
          self.coarse, self.pred_pc = self.G(self.pred_pc)

      #self.visualize_ops = [self.input[0], self.coarse[0], self.fine[0], self.gt[0]]
      self.visualize_titles = ['input_x', 'coarse', 'fine', 'gt']

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
      self.sess.run(tf.global_variables_initializer())
      self.step = self.sess.run(self.global_step)

      self.saver = tf.train.Saver()


  def summary_all(self):

      # summary
      add_scalar_summary('loss/dis_coarse_cd', self.dis_coarse_cd, collection='gen')
      add_scalar_summary('loss/dis_fine_cd', self.dis_fine_cd, collection='gen')
      add_scalar_summary('loss/dis_coarse_hd', self.dis_coarse_hd, collection='gen')
      add_scalar_summary('loss/dis_fine_hd', self.dis_fine_hd, collection='gen')

      add_scalar_summary('loss/repulsion_loss', self.repulsion_loss,collection='gen')
      add_scalar_summary('loss/uniform_loss', self.uniform_loss,collection='gen')

      # add_scalar_summary('loss/shapeLoss', self.shapeLoss, collection='gen')
      # add_scalar_summary('loss/densityLoss', self.densityLoss, collection='gen')
      # add_scalar_summary('loss/directionLoss', self.directionLoss, collection='gen')


      add_scalar_summary('loss/G_loss', self.G_gan_loss,collection='gen')
      add_scalar_summary('loss/total_gen_loss', self.total_gen_loss, collection='gen')

      add_scalar_summary("weights/lr_g", self.lr_g, collection='gen')
      add_scalar_summary("weights/lr_d", self.lr_g, collection='gen')
      add_scalar_summary('weights/weight_fine', self.weight_fine, collection='gen')


      add_hist_summary('D/true', self.fake_logit, collection='dis')
      add_hist_summary('D/fake', self.real_logit, collection='dis')
      add_scalar_summary('loss/D_loss', self.D_loss,collection='dis')

      self.g_summary_op = tf.summary.merge_all('gen')
      self.d_summary_op = tf.summary.merge_all('dis')

      self.visualize_x_titles = ['input_x', 'fake_y', 'real_y']
      self.visualize_x_ops = [self.input[0], self.fine[0], self.gt[0]]
      self.image_merged = tf.placeholder(tf.float32, shape=[None, 2000, 1500, 1])
      self.image_summary = tf.summary.image('Upsampling', self.image_merged, max_outputs=1)

  def setup_optimizer(self):
      self.lr_d = self.opts.base_lr_d
      if self.opts.lr_decay:
          learning_rate_d = tf.train.exponential_decay(
              self.opts.base_lr_d,
              self.epoch,
              self.opts.decay_step,
              decay_rate=self.opts.lr_decay_rate,
              staircase=True,
              name="learning_rate_d_decay",
          )
          self.lr_d = tf.maximum(self.lr_d, 1e-6)

      self.lr_g = self.opts.base_lr_g
      if self.opts.lr_decay:
          self.lr_g = tf.train.exponential_decay(
              self.opts.base_lr_g,
              self.epoch,
              self.opts.decay_step,
              decay_rate=self.opts.lr_decay_rate,
              staircase=True,
              name="learning_rate_g_decay",
          )
          self.lr_g = tf.maximum(self.lr_g, self.opts.lr_clip)

      t_vars = tf.global_variables()
      # create pre-generator ops
      gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
      gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

      with tf.control_dependencies(gen_update_ops):
        self.G_optimizers = tf.train.AdamOptimizer(self.lr_g, beta1=self.opts.beta).minimize(self.total_gen_loss, var_list=gen_tvars)

      dis_tvars = [var for var in t_vars if var.name.startswith("discriminator")]
      self.D_clip = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dis_tvars]
      self.D_optimizers = tf.train.AdamOptimizer(self.lr_d, beta1=self.opts.beta).minimize(self.D_loss,var_list=dis_tvars)

  def train(self):

      self.build_model()

      self.saver = tf.train.Saver(max_to_keep=None)
      self.writer = tf.summary.FileWriter(self.opts.log_dir, self.sess.graph)

      restore_epoch = 0
      if self.opts.restore:
          restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
          self.saver.restore(self.sess, checkpoint_path)
          #self.saver.restore(self.sess, tf.train.latest_checkpoint(self.opts.log_dir))
          self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
          #tf.assign(self.global_step, restore_epoch * self.train_dataset.num_batches).eval()
          #restore_epoch += 1
      else:
          self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')

      with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
          for arg in sorted(vars(self.opts)):
              log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments

      step = self.sess.run(self.global_step)
      start = time()

      self.train_dataset = Fetcher(self.opts, augment=True)
      self.log_string("train_dataset: %d" % len(self.train_dataset))
      best_acc = math.inf
      for i in range(restore_epoch, self.opts.training_epoch):
          (
              d_loss,
              g_loss,
              coarse_cd_loss,
              coarse_hd_loss,
              fine_cd_loss,
              fine_hd_loss,
              duration,
          ) = self.train_one_epoch(i)
          self.train_dataset.reset()
          epoch = int(self.sess.run(self.increment_epoch))
          # logging.info('**** EPOCH %03d ****\t' % (epoch))
          self.log_string(
              "epoch %04d d_loss=%.9f g_loss=%.9f  coarse_cd=%.9f  coarse_hd=%.9f  fine_cd=%.9f fine_hd=%.9f  time=%.4f" % (
                  epoch, d_loss, g_loss, coarse_cd_loss, coarse_hd_loss, fine_cd_loss, fine_hd_loss, duration / 60.0))

          if (epoch % self.opts.epoch_per_save) == 0 and fine_cd_loss < best_acc:
              best_acc = fine_cd_loss
              self.saver.save(self.sess, os.path.join(self.opts.log_dir, 'model'), epoch)
              print(colored('Model saved at %s, accuracy %.5f' % (self.opts.log_dir,best_acc), 'white', 'on_blue'))

  def train_one_epoch(self,epoch=0):

      n_examples = int(len(self.train_dataset))

      epoch_g_loss = AverageMeter()
      epoch_d_loss = AverageMeter()
      epoch_coarse_loss = AverageMeter()
      epoch_coarse_hd_loss = AverageMeter()
      epoch_fine_loss = AverageMeter()
      epoch_fine_hd_loss = AverageMeter()

      n_batches = int(n_examples / self.opts.batch_size) - 1
      start_time = time()

      for _ in tqdm(range(n_batches)):

          batch_input_x, batch_input_y, batch_radius = self.train_dataset.next_batch()

          feed_dict = {self.input: batch_input_x,
                       self.gt: batch_input_y,
                       self.pc_radius: batch_radius,
                       self.is_training: True}

          if self.use_gan:
              # Update D network
              _, _, d_loss, d_summary = self.sess.run([self.D_optimizers, self.D_clip, self.D_loss, self.d_summary_op],
                                                      feed_dict=feed_dict)
              self.writer.add_summary(d_summary, self.step)
              epoch_d_loss.update(d_loss)
          # Update G network
          _, g_loss, coarse_loss, fine_loss, coarse_hd_loss,fine_hd_loss, summary = self.sess.run(
              [self.G_optimizers, self.total_gen_loss, self.dis_coarse_cd, self.dis_fine_cd,
               self.dis_coarse_hd, self.dis_fine_hd,
               self.g_summary_op], feed_dict=feed_dict)
          self.writer.add_summary(summary, self.step)

          epoch_g_loss.update(g_loss)
          epoch_coarse_loss.update(coarse_loss)
          epoch_fine_loss.update(fine_loss)
          epoch_fine_hd_loss.update(fine_hd_loss)
          epoch_coarse_hd_loss.update(coarse_hd_loss)

          if True:
              self.step += 1
              if True and self.step % self.opts.steps_per_print == 0:

                  feed_dict = {self.input: batch_input_x,
                               self.is_training: False}

                  coarse,fine = self.sess.run([self.coarse,self.fine], feed_dict=feed_dict)

                  image_sparse = point_cloud_three_views(batch_input_x[0])
                  image_coarse = point_cloud_three_views(coarse[0])
                  image_fine = point_cloud_three_views(fine[0])
                  image_gt = point_cloud_three_views(batch_input_y[0])
                  image_merged = np.concatenate([image_sparse, image_coarse, image_fine, image_gt], axis=1)
                  image_merged = np.transpose(image_merged, [1, 0])
                  image_merged = np.expand_dims(image_merged, axis=0)
                  image_merged = np.expand_dims(image_merged, axis=-1)
                  image_summary = self.sess.run(self.image_summary, feed_dict={self.image_merged: image_merged})
                  self.writer.add_summary(image_summary, self.step)

              if self.opts.visulize and (self.step % self.opts.steps_per_visu == 0):
                  feed_dict = {self.input: batch_input_x,
                               self.gt: batch_input_y,
                               self.pc_radius: batch_radius,
                               self.is_training: False}
                  pcds = self.sess.run([self.visualize_ops], feed_dict=feed_dict)
                  pcds = np.squeeze(pcds)  # np.asarray(pcds).reshape([3,self.opts.num_point,3])
                  plot_path = os.path.join(self.opts.log_dir, 'plots',
                                           'epoch_%d_step_%d.png' % (epoch, self.step))
                  plot_pcd_three_views(plot_path, pcds, self.visualize_titles)

      duration = time() - start_time

      return (
          epoch_d_loss.avg,
          epoch_g_loss.avg,
          epoch_coarse_loss.avg,
          epoch_coarse_hd_loss.avg,
          epoch_fine_loss.avg,
          epoch_fine_hd_loss.avg,
          duration,
      )


  def patch_prediction(self, patch_point):
      # normalize the point clouds
      patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
      patch_point = np.expand_dims(patch_point, axis=0)
      pred,pred_coarse = self.sess.run([self.pred_pc,self.coarse], feed_dict={self.input: patch_point})
      pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
      pred_coarse = np.squeeze(centroid + pred_coarse * furthest_distance, axis=0)
      return pred,pred_coarse

  def pc_prediction(self, pc):
      ## get patch seed from farthestsampling
      points = tf.convert_to_tensor(np.expand_dims(pc,axis=0),dtype=tf.float32)
      start= time()
      #print('------------------patch_num_point:',self.opts.patch_num_point)
      seed1_num = int(pc.shape[0] / self.opts.patch_num_point * self.opts.patch_num_ratio)

      ## FPS sampling
      seed = farthest_point_sample(seed1_num, points).eval(session=self.sess)[0]
      seed_list = seed[:seed1_num]
      #print("farthest distance sampling cost", time() - start)
      #print("number of patches: %d" % len(seed_list))
      input_list = []
      up_point_list=[]
      up_coarse_list = []

      patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, self.opts.patch_num_point)
      #for point in tqdm(patches, total=len(patches)):
      for point in patches:
            up_point,up_coarse = self.patch_prediction(point)
            #up_point = np.squeeze(up_point,axis=0)
            #up_coarse = np.squeeze(up_coarse, axis=0)
            input_list.append(point)
            up_point_list.append(up_point)
            up_coarse_list.append(up_coarse)

      return input_list, up_point_list,up_coarse_list

  def test(self):
      self.opts.batch_size = 1
      final_ratio = self.opts.final_ratio
      step_ratio = 4
      self.opts.up_ratio = step_ratio
      self.build_model_test(final_ratio=self.opts.final_ratio, step_ratio=step_ratio)

      saver = tf.train.Saver()
      restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
      print(checkpoint_path)
      self.saver.restore(self.sess, checkpoint_path)
      #self.restore_model(self.opts.log_dir, epoch=self.opts.restore_epoch, verbose=True)

      samples = glob(self.opts.test_data)
      point = pc_util.load(samples[0])
      self.opts.num_point = point.shape[0]
      out_point_num = int(self.opts.num_point * final_ratio)

      for point_path in samples:
          logging.info(point_path)
          start = time()
          pc = pc_util.load(point_path)[:, :3]
          pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)

          input_list, pred_list, coarse_list = self.pc_prediction(pc)

          end = time()
          print("total time: ", end - start)
          pred_pc = np.concatenate(pred_list, axis=0)
          pred_pc = (pred_pc * furthest_distance) + centroid

          pred_pc = np.reshape(pred_pc, [-1, 3])
          idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval(session=self.sess)[0]
          pred_pc = pred_pc[idx, 0:3]
          # path = os.path.join(self.opts.out_folder, point_path.split('/')[-1][:-4] + '.ply')
          # np.savetxt(path[:-4] + '.xyz',pred_pc,fmt='%.6f')
          in_folder = os.path.dirname(self.opts.test_data)
          path = os.path.join(self.opts.out_folder, point_path.split('/')[-1][:-4] + '_X%d.xyz' % final_ratio)
          np.savetxt(path, pred_pc, fmt='%.6f')


  def log_string(self,msg):
      #global LOG_FOUT
      logging.info(msg)
      self.LOG_FOUT.write(msg + "\n")
      self.LOG_FOUT.flush()

  def restore_model(self, model_path, epoch, verbose=False):
      """Restore all the variables of a saved model.
      """
      self.saver.restore(
          self.sess, osp.join(model_path, "model-" + str(int(epoch)))
      )

      if self.epoch.eval(session=self.sess) != epoch:
          warnings.warn("Loaded model's epoch doesn't match the requested one.")
      else:
          if verbose:
              print("Model restored in epoch {0}.".format(epoch))











