#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py

import os
import re
import sys
import argparse

import numpy as np
import tensorflow as tf

from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import TrainConfig, SyncMultiGPUTrainerParameterServer
from tensorpack.dataflow import imgaug, FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.sessinit import DictRestore

from imagenet_utils import (
    fbresnet_augmentor, get_AVA2012_dataflow, ImageNetModel,
    eval_on_AVA2012, viz_CAM)
from resnet_model import (
    preresnet_group, resnet_group,
    preresnet_basicblock, resnet_basicblock,
    preresnet_bottleneck, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)


### CONSTANTS
TOTAL_BATCH_SIZE = 64
MAX_EPOCH = 110
CAM_DIR_PKL = None  # <-- to make it convenient to send the parameter...

### add @ 20180705
# to check whether AESTHETIC_LEVEL can improve the performance, due to some kind of regularization effects
AESTHETIC_LEVEL = 2

### definition of necessary functions or classes

### Built on the definition of ImageNetModel:
### __init__
### _get_inputs()
### and some parts of _build_graph(): image_preprocess(), regularize_cost and cost
### only remains the get_logits() undefined ...
class Model(ImageNetModel):
    def __init__(self, depth, output_dims, data_format = 'NCHW', mode = 'resnet', JensenFactor = 0.0):
        # change @ 20180705 for AESTHETIC_LEVEL experiment
        super(Model, self).__init__(data_format, JensenFactor = JensenFactor, output_dims = output_dims)

        self.mode = mode
        self.depth = depth
        self.output_dims = output_dims

        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se'    : se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18:  ([2, 2, 2, 2],  basicblock),
            34:  ([3, 4, 6, 3],  basicblock),
            50:  ([3, 4, 6, 3],  bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

        # add @ 20171229: introduce JensenFactor
        self.JensenFactor = JensenFactor

    # implement the get_logits function in ImageNetModel
    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format = self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preresnet_group if self.mode == 'preact' else resnet_group,
                self.block_func, self.output_dims)


### prepare the data augmentation pipeline and other related ones
### NOTE: the data pre-processing in within the range of ImageNetModel
def get_data(name, data_dir, batch, crop_method, color_augmentation = False,
    CAM_dir_pkl = None, repeat_times = 1, strict_order = False, remainder_TR = False):
    isTrain = name == 'train'

    augmentors = fbresnet_augmentor(isTrain, crop_method, color_augmentation)

    return get_AVA2012_dataflow(data_dir, name, batch, augmentors, \
        CAM_dir_pkl = CAM_dir_pkl, CAMCropR = ((isTrain) and (crop_method == 'CAMCropR')), \
        repeat_times = repeat_times, strict_order = strict_order, remainder_TR = remainder_TR)


### the setting of configuration during the training process
def get_config(model, data_dir, crop_method_TR, color_augmentation, crop_method_TS):
    nr_tower = max(get_nr_gpu(), 1)
    batch = TOTAL_BATCH_SIZE // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))

    # data pipelines of train and validation
    dataset_train = get_data('train', data_dir, batch, crop_method_TR, \
        color_augmentation = color_augmentation, CAM_dir_pkl = CAM_DIR_PKL)
    dataset_val = get_data('val', data_dir, batch, crop_method_TS)

    # TODO
    callbacks = [
        # class callbacks.ModelSaver(max_to_keep = 10, keep_checkpoint_every_n_hours = 0.5,
        #     checkpoint_dir = None, var_collections = 'variables')
        ModelSaver(max_to_keep = MAX_EPOCH),
        # @ 20171129: finetune on ResNet d18 from ImageNet
        # maybe moderate learning_rate is perferable
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, 1e-3), (20, 5e-4), (40, 1e-4), (60, 1e-5)]),
        HumanHyperParamSetter('learning_rate'),
    ]

    # 0 or 1
    infs = [ClassificationError('wrong-top1', 'val-error-top1')]

    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))

    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model           = model,
        dataflow        = dataset_train,
        callbacks       = callbacks,
        #steps_per_epoch = 5000,
        max_epoch       = MAX_EPOCH,
        nr_tower        = nr_tower
    )


###
def name_conversion(caffe_layer_name):
    """ Convert a caffe parameter name to a tensorflow parameter name as
        defined in the above model """
    # beginning & end mapping
    NAME_MAP = {'bn_conv1/beta': 'conv0/bn/beta',
                'bn_conv1/gamma': 'conv0/bn/gamma',
                'bn_conv1/mean/EMA': 'conv0/bn/mean/EMA',
                'bn_conv1/variance/EMA': 'conv0/bn/variance/EMA',
                'conv1/W': 'conv0/W', 'conv1/b': 'conv0/b',
                'fc1000/W': 'linear/W', 'fc1000/b': 'linear/b'}
    if caffe_layer_name in NAME_MAP:
        return NAME_MAP[caffe_layer_name]

    s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
    if s is None:
        s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
        layer_block_part1 = s.group(3)
        layer_block_part2 = s.group(4)
        assert layer_block_part1 in ['a', 'b']
        layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)

    else:
        layer_block = ord(s.group(3)) - ord('a')

    layer_type = s.group(1)
    layer_group = s.group(2)

    layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
    assert layer_branch in [1, 2]
    if layer_branch == 2:
        layer_id = re.search('_branch[0-9]([a-z])/', caffe_layer_name).group(1)
        layer_id = ord(layer_id) - ord('a') + 1

    TYPE_DICT = {'res': 'conv{}', 'bn': 'conv{}/bn'}
    layer_type = TYPE_DICT[layer_type].format(layer_id if layer_branch == 2 else 'shortcut')

    tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
    tf_name = 'group{}/block{}/{}'.format(
        int(layer_group) - 2, layer_block, layer_type) + tf_name
    return tf_name


### in order to conver data from .npy file to the format of tensorflow
### @ 20171129: the original version of convert_param_name converts the
### resnet model from a caffe version to the defined one in a tensorflow
### version showed as in resnet_model.py
def convert_param_name(param):
    print('--> convert_param_name ...')
    resnet_param = {}
    for k in param.keys():
        logger.info("Load the weights of the module {}".format(k.split(":")[0]))
        resnet_param[k.split(":")[0]] = param[k]

    return resnet_param


###      ###
### MAIN ###
###      ###
if __name__ == '__main__':
    # input parser
    # @ 20171120: input with instruction items first
    #             we need to swift to the mode of changing the hyper-parameters sometime
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',         help = 'comma separated list of GPU(s) to use.')

    # add @ 20171229 for JensenEnhanced Loss
    parser.add_argument('--JensenFactor', help = 'JensenFactor in JensenEnhanced Loss [None]',
        type = float, default = 0.0)

    parser.add_argument('--data',        help = 'AVA2012 dataset dir')
    parser.add_argument('--data_format', help = 'Specify the data_format: [NCHW] or NHWC',
        type = str, choices = ['NHWC', 'NCHW'], default = 'NCHW')
    # add @ 20180705
    # AESTHETIC_LEVEL, 3 VS 2 ?
    parser.add_argument('--aesthetic_level', help = 'HOW many units do you want to apply in aesthetic assessment [2]',
        type = int, default = AESTHETIC_LEVEL)

    # add @ 20171125 for CAMCrop reverse
    # inspired by Grid loss and the paper of weak segmentation built on CAM
    # maybe we can try to reverse CAMCrop. That is to enforce the training with many more difficult instances
    parser.add_argument('--crop_method_TR', help = 'The crop method during the training process [ShortestEdgeResize]',
        type = str, choices = ['GoogleNetResize', 'ShortestEdgeResize', 'GlobalWarp', 'CAMCrop', 'CAMCropR'],
        default = 'ShortestEdgeResize')
    # add @ 20171124 for CAMCrop
    #                    and CAMCropR
    parser.add_argument('--CAM_dir_pkl',    help = 'The directory to load CAM_7x7 (*.pkl) for robustness/de-noising',
        type = str, default = None)
    parser.add_argument('--color_augmentation', help = 'Whether to perform augmentation/jittering in color [False]',
        action = 'store_true')
    parser.add_argument('--crop_method_TS', help = 'The crop method during the validation/test process [CenterCrop]',
        type = str, choices = ['RandomCrop', 'CenterCrop'], default = 'CenterCrop')
    parser.add_argument('--repeat_times',   help = 'How many times to repeat each data in the evaluation stage [1]',
        type = int, default = 1)

    parser.add_argument('--load',        help = 'the file to load the existing model, be careful of the loaded variables',
        type = str, default = None)
    # add @ 20171128
    parser.add_argument('--load_npy',    help = 'the npy file to initialize the model, as well as the session',
        type = str, default = None)
    parser.add_argument('--mode',        help = 'variants of resnet to use [resnet]',
        choices = ['resnet', 'preact', 'se'], default = 'resnet')
    parser.add_argument('-d', '--depth', help = 'the specific number of depth of CNN to run [18]',
        type = int, default = 18, choices = [18, 34, 50, 101, 152])

    parser.add_argument('--eval',        help = 'test the performance of the loaded model on AVA2012 [False]',
        action = 'store_true')
    parser.add_argument('--CAM_viz',     help = 'which dataset to extract the Class Activation Map (CAM) viz [None]',
        type = str, choices = ['train', 'val', None], default = None)
    parser.add_argument('--CAM_viz_PKL', help = 'whether to save the pkl files of the CAM viz results [False]',
        action = 'store_true')
    parser.add_argument('--CAM_viz_REP', help = 'whether to save the rep files during the viz_CAM process [False]',
        action = 'store_true')
    parser.add_argument('--CAM_dir',     help = 'the directory/folder to save the visualization images [./for_CAM_viz]',
        type = str, default = 'for_CAM_viz')

    args = parser.parse_args()

    # variables initialization
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode == 'se':
        assert args.depth >= 50

    # when the crop_method_TS is CenterCrop, the value of repeat_times seems to be of non-sense
    if args.crop_method_TS == 'CenterCrop':
        args.repeat_times = 1

    # add @ 20171128: load and load_npy cannot be of Non-None at the same time ...
    assert (args.load == None) or (args.load_npy == None), '*** load and load_npy cannot be of Non-None at the same time ***'

    # add @ 20171124: to check the necessary circumstance to perform CAMCrop
    # TODO: only for train, not for CAM_viz
    if (args.crop_method_TR == 'CAMCrop') or (args.crop_method_TR == 'CAMCropR'):
        assert os.path.isdir(args.CAM_dir_pkl), '*** THE CAM_dir_pkl DOES NOT EXIST FOR CAMCrop & CAMCropR TO PERFORM ***'
        CAM_DIR_PKL = args.CAM_dir_pkl

    # add @ 20171229: to check the value setting of JensenFactor
    if (args.JensenFactor != None):
        assert args.JensenFactor >= 0.0, '*** JensenFactor MUST BE A POSITIVE FLOAT ***'

    # add @ 20180705
    assert args.aesthetic_level >= 2, '*** aesthetic_level MUST BE AN INTEGER NUMBER LARGER OR EQUAL THAN 2  ***'

    # TODO
    # the definition of model
    model = Model(args.depth, args.aesthetic_level, data_format = args.data_format, \
                  mode = args.mode, JensenFactor = args.JensenFactor)

    # CAM_viz or eval or train
    if args.CAM_viz != None:
        batch = 128

        # @ 20171125
        if args.CAM_viz == 'train':
            assert args.crop_method_TR not in ['CAMCrop', 'CAMCropR'], '*** CAMCrop or CAMCropR IS NOT ALLOWED IN CAM_viz ***'

        # @ 20171122 : visualize the CAM for better understanding of the CNN
        #              introduce strict_order to make it convenient for image saving process
        # @ 20171124 : introduce remainder_TR to extract all the CAM_viz of images in the training set
        ds = get_data(args.CAM_viz, args.data, batch, args.crop_method_TS if args.CAM_viz == 'val' else args.crop_method_TR, \
                      repeat_times = 1, strict_order = True, remainder_TR = True if args.CAM_viz == 'train' else False)

        viz_CAM(model, get_model_loader(args.load), args.CAM_viz, ds, args.CAM_dir,
            save_PKL = args.CAM_viz_PKL, save_REP = args.CAM_viz_REP)

    elif args.eval:
        batch = 128 # something that can run on one gpu

        # @ 20171121 to make it convenient to calculate average estimation
        batch = int(batch // args.repeat_times * args.repeat_times)

        ds = get_data('val', args.data, batch, args.crop_method_TS, repeat_times = args.repeat_times, strict_order = True)

        eval_on_AVA2012(model, get_model_loader(args.load), ds, args.repeat_times)

    else:
        # add @ 20171128: the strategy of parameter initalization with a ImageNet pre-trained model
        #                 should be recorded within the name string of the directory of training log
        initial_strategy = '_fromScratch'
        if args.load:
            initial_strategy = '_preTrainedModel'
        elif args.load_npy:
            initial_strategy = '_preTrainedImageNetModel'

        # change @ 20180705
        # introduce An for the AESTHETIC_LEVEL is set to n
        logger.set_logger_dir('./train_log/AVA2012{6}-{0}-d{1}-{2}-{3}{4}{5}_LRT3'.format(args.mode, args.depth, \
            args.crop_method_TR, args.crop_method_TS, initial_strategy, \
            '' if args.JensenFactor == 0.0 else '_JE{}'.format(args.JensenFactor), \
            '' if args.aesthetic_level == AESTHETIC_LEVEL else '-A{}'.format(args.aesthetic_level)))

        config = get_config(model, args.data, args.crop_method_TR, args.color_augmentation, args.crop_method_TS)

        # load pre-trained model if it exists
        # TODO: layer-cascade or freeze-layer ? rely-backpropagation ?
        #       layer-wise adaptive scale rate ?
        if args.load:
            print('--> initialize the session with the checkpoint file %s', args.load)
            config.session_init = get_model_loader(args.load)

        elif args.load_npy:
            print('--> initalize the session with the npy file %s', args.load)
            # add @ 20171128: adopt the ImageNet pre-trained model for initialization purpose
            #                 load params from npy file, convert them into the desired formation,
            #                 and apply DictRestore to initial config.session_init
            param = np.load(args.load_npy, encoding = 'latin1')
            param = convert_param_name(param)
            config.session_init = DictRestore(param)

        SyncMultiGPUTrainerParameterServer(config).train()


### THE END
