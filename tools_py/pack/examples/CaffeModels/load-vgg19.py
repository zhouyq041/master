#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: load-vgg19.py

from __future__ import print_function
import cv2
import tensorflow as tf
import numpy as np
import os
import argparse

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow.dataset import ILSVRCMeta


def tower_func(image):
    with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu):
        logits = (LinearWrap(image)
                  .Conv2D('conv1_1', 64)
                  .Conv2D('conv1_2', 64)
                  .MaxPooling('pool1', 2)
                  # 112
                  .Conv2D('conv2_1', 128)
                  .Conv2D('conv2_2', 128)
                  .MaxPooling('pool2', 2)
                  # 56
                  .Conv2D('conv3_1', 256)
                  .Conv2D('conv3_2', 256)
                  .Conv2D('conv3_3', 256)
                  .Conv2D('conv3_4', 256)
                  .MaxPooling('pool3', 2)
                  # 28
                  .Conv2D('conv4_1', 512)
                  .Conv2D('conv4_2', 512)
                  .Conv2D('conv4_3', 512)
                  .Conv2D('conv4_4', 512)
                  .MaxPooling('pool4', 2)
                  # 14
                  .Conv2D('conv5_1', 512)
                  .Conv2D('conv5_2', 512)
                  .Conv2D('conv5_3', 512)
                  .Conv2D('conv5_4', 512)
                  .MaxPooling('pool5', 2)
                  # 7
                  .FullyConnected('fc6', 4096, activation=tf.nn.relu)
                  .Dropout('drop0', 0.5)
                  .FullyConnected('fc7', 4096, activation=tf.nn.relu)
                  .Dropout('drop1', 0.5)
                  .FullyConnected('fc8', 1000)())
    tf.nn.softmax(logits, name='prob')


def run_test(path, input):
    param_dict = dict(np.load(path))
    predict_func = OfflinePredictor(PredictConfig(
        inputs_desc=[InputDesc(tf.float32, (None, 224, 224, 3), 'input')],
        tower_func=tower_func,
        session_init=DictRestore(param_dict),
        input_names=['input'],
        output_names=['prob']   # prob:0 is the probability distribution
    ))

    im = cv2.imread(input)
    assert im is not None, input
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224, 224)).reshape((1, 224, 224, 3)).astype('float32')

    # VGG19 requires channelwise mean substraction
    VGG_MEAN = [103.939, 116.779, 123.68]
    im -= VGG_MEAN[::-1]
    outputs = predict_func(im)[0]
    prob = outputs[0]
    ret = prob.argsort()[-10:][::-1]
    print("Top10 predictions:", ret)

    meta = ILSVRCMeta().get_synset_words_1000()
    print("Top10 class names:", [meta[k] for k in ret])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', required=True,
                        help='.npz model file generated by tensorpack.utils.loadcaffe')
    parser.add_argument('--input', help='an input image', required=True)
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    run_test(args.load, args.input)
