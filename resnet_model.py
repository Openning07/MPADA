#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)


### return the shortcut given l as the input and n_out as the desired number of shortcut output channels
def resnet_shortcut(l, n_out, stride, nl = tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]

    if n_in != n_out:   # change dimension via 1x1 conv when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)

    else:
        return l


### return the processed branch and the shortcut branch
def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)

    else:
        shortcut = l

    return l, shortcut


### return a BatchNorm layer given x, name and zero_init
def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    Besides, moving average deserves a more careful control
    """
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


### a typical module of resnet block with pre-activation
### i --> 3x3 conv --> 3x3 conv --> + --> o
###    |                            |
###    -----( possible 1x1 conv )----
def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


### another version of resnet block with bottlebneck
### i --> 1x1 conv --> 3x3 conv --> 1x1 conv --> + -->
###    |                                         |
###    ----------------> 1x1 conv ----------------
def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


###
def preresnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)

    return l


###
def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, stride = stride, nl = BNReLU)
    tf.identity(l, 'BNReLU1')

    l = Conv2D('conv2', l, ch_out, 3, nl = get_bn(zero_init=True))
    tf.identity(l, 'BNReLU2')

    return l + resnet_shortcut(shortcut, ch_out, stride, nl = get_bn(zero_init = False))


###
def resnet_bottleneck(l, ch_out, stride, stride_first = False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride = stride if stride_first else 1, nl = BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride = 1 if stride_first else stride, nl = BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl = get_bn(zero_init = True))

    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl = get_bn(zero_init = False))


###
def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
    l = l * tf.reshape(squeeze, [-1, ch_out * 4, 1, 1])

    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


###
def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)

                # end of each block need an activation
                # @ 20171122 to make it convenient to extract the output
                #            here we set the name as ReLU_output
                l = tf.nn.relu(l, name = 'ReLU_output')

    return l


### architecture of ResNet
### 7x7 conv -> four cascaded groups with increased channels -> global average pooling & FC -> softmax
def resnet_backbone(image, num_blocks, group_func, block_func, output_dims):
    with argscope(Conv2D, nl = tf.identity, use_bias = False, W_init = variance_scaling_initializer(mode = 'FAN_OUT')):
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride = 2, nl = BNReLU)
                  .MaxPooling('pool0', shape = 3, stride = 2, padding = 'SAME')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)
                  .GlobalAvgPooling('gap')
                  .FullyConnected('linear_C{}'.format(output_dims), output_dims, \
                                  nl = tf.identity)())

    return logits


### THE END
