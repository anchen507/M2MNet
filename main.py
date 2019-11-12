import argparse
from glob import glob

import tensorflow as tf

from model import denoiser
from utils import *
import os
import numpy as np


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--is_color', dest='is_color', type=int, default=1, help='color flag, 1 for color image and 0 for grayscale image')
parser.add_argument('--sigma', dest='sigma', type=int, default=20, help='noise level')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--paras', dest='paras', default='checkpoint/color_20', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--train_path', dest='train_path', default='./data/color_img_clean_pats.npy', help='training file')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--test_path', dest='test_path', default='data/color_test_PHD/football/', help='dataset for testing')
args = parser.parse_args()


def denoiser_train(denoiser, lr):
    with load_data(filepath=args.train_path) as data:
        print(len(data))
        lines = open('./data/test_color.list', 'r')
        eval_files = list(lines)
        denoiser.train(data, eval_files, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr,
                       sample_dir=args.sample_dir)


def denoiser_test(denoiser):
    denoiser.test(args.test_path, paras=args.paras, save_dir=args.test_dir)


def main(_):
    if args.phase == 'train':
        if not os.path.exists(args.sample_dir):
           os.makedirs(args.sample_dir)
    elif args.phase == 'test':
         if not os.path.exists(args.test_dir):
            os.makedirs(args.test_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[40:] = lr[0] / 10.0
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("GPU\n")
        config = tf.ConfigProto() 
        config.gpu_options.allow_growth = False #
        with tf.Session(config=config) as sess:
            model = denoiser(sess, is_color=args.is_color,sigma=args.sigma,batch_size=args.batch_size)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess,is_color=args.is_color, sigma=args.sigma,batch_size=args.batch_size)
            if args.phase == 'train':
               denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
