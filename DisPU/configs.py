import argparse
import os

def str2bool(x):
    return x.lower() in ('true')


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train',help="train/test")
parser.add_argument('--log_dir', default='log')
parser.add_argument('--data_dir', default='data')
parser.add_argument('--augment', type=str2bool, default=True)
parser.add_argument('--restore', action='store_true')
parser.add_argument('--more_up', type=int, default=0)
parser.add_argument('--training_epoch', type=int, default=401)
parser.add_argument('--batch_size', type=int, default=28)
parser.add_argument('--random', type=str2bool, default=True)
parser.add_argument('--jitter', type=str2bool, default=False)
parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
parser.add_argument('--up_ratio', type=int, default=4)
parser.add_argument('--final_ratio', type=int, default=4, help="[4,16]")
#parser.add_argument('--num_point', type=int, default=256)
parser.add_argument('--patch_num_point', type=int, default=256)
parser.add_argument('--patch_num_ratio', type=int, default=3)
parser.add_argument('--base_lr_d', type=float, default=0.0001)
parser.add_argument('--base_lr_g', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--lr_decay', type=str2bool, default=True)
parser.add_argument('--decay_step', type=int, default=30)
parser.add_argument('--start_decay_step', type=int, default=40)
parser.add_argument('--lr_decay_steps', type=int, default=40)
parser.add_argument('--lr_decay_rate', type=float, default=0.7)
parser.add_argument('--lr_clip', type=float, default=1e-6)
parser.add_argument('--steps_per_print', type=int, default=50)
parser.add_argument('--visulize', type=str2bool, default=False)
parser.add_argument('--steps_per_visu', type=int, default=100)
parser.add_argument('--epoch_per_save', type=int, default=20)
parser.add_argument('--use_repulse', type=str2bool, default=True)
parser.add_argument('--repulsion_w', default=1.0, type=float, help="repulsion_weight")
parser.add_argument('--fidelity_w', default=100.0, type=float, help="fidelity_weight")
parser.add_argument('--uniform_w', default=10.0, type=float, help="uniform_weight")
parser.add_argument('--gan_w', default=1.0, type=float, help="gan_weight")
parser.add_argument('--gen_update', default=2, type=int, help="gen_update")

FLAGS = parser.parse_args()

