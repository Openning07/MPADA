#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py

import os
import sys
import time
import pickle

from sklearn.metrics import confusion_matrix

import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (
    AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData, RepeatedDataPoint)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import viz


class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(224, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(224).augment(out)
        return out


# @ 20171120: introduce crop_method for TR and TS; specific the color_augmentation
def fbresnet_augmentor(isTrain, crop_method, color_augmentation):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    execution_lst = []

    if isTrain:
        augmentors = [
            # 1. crop_method
            # a) GoogleNetResize
            GoogleNetResize(),
            # b) ShortestEdgeResize
            imgaug.ResizeShortestEdge(256),
            # c) GlobalWarp
            imgaug.Resize(226),  # NOTE: for CAM generation
            imgaug.RandomCrop((224, 224)),
            # d) CAMCrop
            # (when CAMCrop is set, the output from the original DataFlow has already been cropped)
            # 2. color_augmentation
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]

        #
        if crop_method == 'GoogleNetResize':
            print('--> perform GoogleNetResize cropping method during the training pipeline')
            execution_lst.extend([0])
        elif crop_method == 'ShortestEdgeResize':
            print('--> perform ShortestEdgeResize cropping method during the training pipeline')
            execution_lst.extend([1, 3])
        elif crop_method == 'GlobalWarp':
            print('--> perform GlobalWarp cropping method during the training pipeline')
            execution_lst.extend([2, 3])
        elif crop_method == 'CAMCrop':
            # enable CAMCrop @ 20171124
            print('*** Perform CAMCrop to better the training dynamics and the results ***')

        if color_augmentation:
            print('--> perform color augmentation during the training pipeline')
            execution_lst.extend([4])
        else:
            print('--> discard the color jittering process during the training pipeline')

        # perform mirror reflection augmentation anyway
        execution_lst.extend([5])

    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
            imgaug.RandomCrop((224, 224)),
        ]

        if crop_method == 'RandomCrop':
            execution_lst.extend([0, 2])

        elif crop_method == 'CenterCrop':
            execution_lst.extend([0, 1])

    return [ item_ for id_, item_ in enumerate(augmentors) if id_ in execution_lst ]


# @ 20171121: introduce repeat_times for the purpose of average estimation during the eval stage
# @ 20171122: introduce strict_order to keep the order of get_image_lst('val')[0] and the order of
#             dataflow in harmony ...
# @ 20171124: to extract the CAM_viz of all the images of the train set, we introduce remainder_TR
# @ 20171125: for CAMCropR
def get_AVA2012_dataflow(datadir, name, batch_size, augmentors,
                         CAM_dir_pkl = None, CAMCropR = False,
                         repeat_times = 1, strict_order = False, remainder_TR = False):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val']
    assert datadir is not None
    assert isinstance(augmentors, list)

    isTrain = name == 'train'
    cpu = min(2, multiprocessing.cpu_count())
    if isTrain:
        # change @ 20171122
        # change @ 20171125: for CAMCropR
        ds = dataset.AVA2012(datadir, name, CAM_dir_pkl = CAM_dir_pkl, CAMCropR = CAMCropR,
                             shuffle = (strict_order == False))
        print('--> information about the original dataFlow from AVA2012 [TRAIN]:')
        print(ds)

        ds = AugmentImageComponent(ds, augmentors, copy = False)
        # change @ 20171122
        # NOTE: When ``nr_proc=1``, the dataflow produces the same data as ``ds`` in the same order
        ds = PrefetchDataZMQ(ds, cpu if strict_order == False else 1)
        ds = BatchData(ds, batch_size, remainder = remainder_TR)

    else:
        ds = dataset.AVA2012(datadir, name, shuffle = False)
        print('--> information about the original dataFlow from AVA2012 [VALIDATION]:')
        print(ds)

        # add @ 20171121
        ds = RepeatedDataPoint(ds, repeat_times)

        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            im, cls = dp
            im = aug.augment(im)
            return im, cls

        # change @ 20171122
        # BUG @ 20171128
        # NOTE: The order of data points out of MultiThreadMapData.get_data()
        #       is not the same as ds.get_data() !
        # NOTE: buffer_size is the minimum number of images loaded from a given folder
        ds = MultiThreadMapData(ds, cpu if strict_order == False else 1, \
            mapf, buffer_size = 500, strict = True)
        ds = BatchData(ds, batch_size, remainder = True)
        ds = PrefetchDataZMQ(ds, 1)

    return ds


### adapt from eval_on_ILSVRC12
# @ 20171121: introduce repeat_times for the purpose of average estimation during the eval stage
# @ 20171122: introduce confusion_matrix from sklearn for details about the classification performance
# @ 20171127: introduce fusion_method to refine the approach of fusion
# @ 20171201: introduce output_predictions to save the interval results for further experiments
# TODO: feature extraction and other experiments for explaination and exploration
def eval_on_AVA2012(model, sessinit, dataflow, repeat_times, fusion_method = 'GlobalAverage',
                    output_predictions = True):
    pred_config = PredictConfig(
        model        = model,
        session_init = sessinit,
        input_names  = ['input', 'label'],
        output_names = ['softmax-logits', 'label']
    )

    # add @ 20171127
    def DeMaxMin_GlobalAverage(dps):
        res01 = []

        for cls in range(2):
            s = dps[:, cls]
            s_sum_denoise = np.sum(s) - np.max(s) - np.min(s)
            res01.append(s_sum_denoise / (s.shape[0] - 2))

        return res01

    # add @ 20171127
    def Median(dps):
        res01 = []

        for cls in range(2):
            res01.append(np.median(dps[:, cls]))

        return res01

    #
    def accuracyEstimation(log01_, GTs01_):
        batch_size = log01_.shape[0]
        acc01_ = np.zeros((int(batch_size // repeat_times),))
        y_True_Pred = np.zeros((int(batch_size // repeat_times), 2))
        avg_p = np.zeros((int(batch_size // repeat_times), log01_.shape[1]))

        #
        for i in range(acc01_.shape[0]):
            # change @ 20171127 : refine the fusion approaches
            if fusion_method == 'GlobalAverage':
                avgLog01__ = np.average(log01_[(i * repeat_times) : ((i + 1) * repeat_times), :], axis = 0)
            elif fusion_method == 'DeMaxMin_Average':
                assert log01_.shape[0] >= 3, '***  ***'
                avgLog01__ = DeMaxMin_GlobalAverage(log01_[(i * repeat_times) : ((i + 1) * repeat_times), :])
            elif fusion_method == 'Median':
                avgLog01__ = Median(log01_[(i * repeat_times) : ((i + 1) * repeat_times), :])

            pred01__ = 0 if avgLog01__[0] > avgLog01__[1] else 1   # TODO: confidence gap? or what if aesthetic_level > 2
            # GTs01_ vs pred01__
            acc01_[i] = int(pred01__ == GTs01_[(i * repeat_times)])
            # add @ 20171122
            y_True_Pred[i, 0] = GTs01_[(i * repeat_times)]
            y_True_Pred[i, 1] = pred01__
            # add @ 20171201
            avg_p[i, :] = avgLog01__[:]

        return acc01_, y_True_Pred, avg_p

    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1 = RatioCounter()
    y_True_Pred_list = []
    interval_results = []

    # add @ 20180709: to estimate the time consumption
    elapsed_times = []
    for pred_res in pred.get_result():
        # for each image, we perform the prediction pipeline for repeat_times
        # therefore, ...
        logs01 = pred_res[0]
        GTs01  = pred_res[1]
        batch_size = logs01.shape[0]
        assert batch_size % repeat_times == 0, \
            '*** batch_size % repeat_times != 0, which makes the accuracyEstimation difficult ***'

        start_time = time.time()
        # change @ 20171122
        # change @ 20171201
        acc01, y_True_Pred_, avg_p = accuracyEstimation(logs01, GTs01)
        elapsed_times.append(time.time() - start_time)

        y_True_Pred_list.append(y_True_Pred_)

        acc1.feed(acc01.sum(), acc01.shape[0])

        # add @ 20171201
        interval_results.append(np.hstack( (y_True_Pred_, avg_p) ))

    # performance exhibition
    print("--> detailed performance exhibition")
    print("    Top1 Accuracy: {}".format(acc1.ratio))

    # add @ 20171122
    y_True_Pred_Matrix = np.vstack(y_True_Pred_list)
    conf_matrix = confusion_matrix(y_True_Pred_Matrix[:, 0], y_True_Pred_Matrix[:, 1])
    print("    Confusion matrix is:")
    print(conf_matrix)
    print("        Accuracy of Negative Prediction: ", (conf_matrix[0, 0]) / (conf_matrix[0, 0] + conf_matrix[1, 0]))
    print("        Accuracy of Positive Prediction: ", (conf_matrix[1, 1]) / (conf_matrix[0, 1] + conf_matrix[1, 1]))
    print("        Recall of Negative Instances   : ", (conf_matrix[0, 0]) / (conf_matrix[0, 0] + conf_matrix[0, 1]))
    print("        Recall of Positive Instances   : ", (conf_matrix[1, 1]) / (conf_matrix[1, 0] + conf_matrix[1, 1]))

    # add @ 20171201
    if output_predictions:
        print('    and save interval_results to ./interval_results_AVA2012.pkl for further investigation ...')
        with open('./interval_results_AVA2012.pkl', 'wb') as output_stream:
            pickle.dump({'interval_results' : interval_results}, output_stream)

    # add @ 20180709: exhibit the information of time consumption
    print('--> average time consumption per image : {0:.3f}ms'.format( \
          1000 * np.sum(elapsed_times) / y_True_Pred_Matrix.shape[0]))


### add @ 20171223 to extract the reps from the given specific graph module
def feature_extraction_on_AVA2012(model, sessinit, dataflow, data_format, output_name, batch_size_times, feature_dir):
    # set the configuration during the prediction process
    # and apply the SimpleDatasetPredictor to extract the output_name
    pred_config = PredictConfig(
        model        = model,
        session_init = sessinit,
        # NOTE: the names in input_names & output_names depends on the definitions in the loaded model
        input_names  = ['input', 'label'],
	output_names = [output_name])
    pred = SimpleDatasetPredictor(pred_config, dataflow)

    # BEGIN
    cnt = 0
    rep_dict = { output_name : [] }
    for outp in pred.get_result():
        #
        rep = outp[0]
        if (len(rep.shape) == 4) and (data_format == 'NCHW'):
            rep = np.transpose(rep, (0, 2, 3, 1))
        rep_dict[output_name].append(rep)

        #
        cnt += 1
        if cnt >= batch_size_times:
            rep_dict[output_name] = np.concatenate(rep_dict[output_name], axis = 0)
            print('    early stop for getting enough reps')
            break;

    #
    with open("{0}/reps_{1}.pkl".format(feature_dir, output_name.replace('/', '-')), 'wb') as output_stream:
        print('--> save {} to a local pkl file ...'.format(output_name))
        print('    shape : ', end = '')
        print(rep_dict[output_name].shape)
        pickle.dump(rep_dict, output_stream, protocol = 2)


### @ 20171122 to extract the CAM visualization results for an enhanced understanding on
###            the aesthetic awareness within the neural networks
def viz_CAM(model, sessinit, name, dataflow, CAM_dir, save_PKL = False, save_REP = False):
    # set the configuration during the prediction process
    # and apply the SimpleDatasetPredictor to extract the output_names
    pred_config = PredictConfig(
        model        = model,
        session_init = sessinit,
        # NOTE: the names in input_names & output_names depends on the definitions in the loaded model
        input_names  = ['input', 'label'],
	output_names = ['wrong-top1', 'group3/block1/ReLU_output', 'linear_C2/W'],
        return_input = True)
    pred = SimpleDatasetPredictor(pred_config, dataflow)

    # create or clear CAM_dir for the output of results of CAM visualization
    CAM_dir = '{}{}'.format(CAM_dir, name)
    if os.path.isdir(CAM_dir):
        print('--> clear the existing results in the directory {}'.format(CAM_dir))
        os.system('rm -r {}'.format(CAM_dir))
    os.system('mkdir -p {}'.format(CAM_dir))

    # for the sake of the ease of file government, we save
    # jpgs, pkls and reps into three different directories
    print('--> during the viz_CAM, we will generate the jpgs', end = '')
    os.system('mkdir -p {}'.format(CAM_dir + '/jpg'))
    if save_PKL:
        print(', pkl', end = '')
        os.system('mkdir -p {}'.format(CAM_dir + '/pkl'))
    if save_REP:
        print(', rep', end = '')
        os.system('mkdir -p {}'.format(CAM_dir + '/rep'))
    print(' files for furthre usage')

    # get the img_lab_list for proper formation of result recording
    img_lab_list = dataset.AVA2012Meta().get_image_list(name)[0]

    # BEGIN
    cnt = 0
    for inp, outp in pred.get_result():
        #
        images, labels = inp
        wrongs, convmaps, W = outp
        batch = wrongs.shape[0]

        #
        for i in range(batch):
            convmap = convmaps[i, :, :, :] # 512 x 7 x 7
            weight0 = W[:, 0].T   # 512 x 1 for negative
            mergedmap0_7x7 = np.matmul(weight0, convmap.reshape((512, -1))).reshape(7, 7)
            mergedmap0 = cv2.resize(mergedmap0_7x7, (224, 224))
            heatmap0 = viz.intensity_to_rgb(mergedmap0)
            blend0 = images[i] * 0.5 + heatmap0 * 0.5

            weight1 = W[:, 1].T   # 512 x 1 for positive
            mergedmap1_7x7 = np.matmul(weight1, convmap.reshape((512, -1))).reshape(7, 7)
            mergedmap1 = cv2.resize(mergedmap1_7x7, (224, 224))
            heatmap1 = viz.intensity_to_rgb(mergedmap1)
            blend1 = images[i] * 0.5 + heatmap1 * 0.5

            concat = np.concatenate((images[i], heatmap0, blend0, heatmap1, blend1), axis = 1)

            imgName, lab01 = img_lab_list[cnt]
            assert lab01 == labels[i], \
                '*** in viz_CAM: lab01 ({0}) != labels[i] ({1}) in image {2}'.format(lab01, labels[i], imgName)

            # save image of CAM visualization
            cv2.imwrite('{0}/jpg/cam_{1}_{2}_{3}.jpg'.format(CAM_dir, os.path.splitext(imgName)[0], \
                lab01, int(wrongs[i])), concat)
            # add @20171123: for CAMCrop
            if save_PKL:
                with open('{0}/pkl/{1}.pkl'.format(CAM_dir, os.path.splitext(imgName)[0]), 'wb') as output_stream:
                    pickle.dump({"GT01" : lab01, "CAM0" : mergedmap0_7x7, "CAM1" : mergedmap1_7x7}, output_stream)

            if save_REP:
                with open('{0}/rep/{1}.rep'.format(CAM_dir, os.path.splitext(imgName)[0]), 'wb') as output_stream:
                    pickle.dump({"convmap" : convmap, "W" : W, "GT01" : lab01}, output_stream)

            cnt += 1

    #
    print('=== Finish CAM_viz on all the images in the validation dataset in AVA2012')


### TODO
### follow the procedures as they are given the ImageNet datasets (e.g., BGR)
### the zero-mean-unit-variance normalization is for ResNet
### please ensure the harmony between your loaded model and your data pipeline
def image_preprocess(image, bgr = True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)

        # stats of images from ImageNet, in RGB channel format
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]

        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std

        return image


### TODO
### update @ 20171229: introduce JensenFactor for the purpose of JensenEnhanced Loss
### update @ 20180705: introduce output_dims for the purpose of AESTHETIC_LEVEL experiments
def compute_loss_and_error(logits, label, JensenFactor = 0.0, output_dims = 2):
    # update @ 20171229
    if JensenFactor == 0.0:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = label)
        loss = tf.reduce_mean(loss, name = 'xentropy-loss')

    else:
        # weighted cross entropy with JensenFactor
        # L = \sum (1 - P^{\JensenFactor}) \cdot log(P)
        softmax_01 = tf.nn.softmax(logits, dim = -1)
        # choose the corresponding softmax_01 due to label
        label_matrix = tf.one_hot(label, output_dims, axis = -1)
        target_p = tf.reduce_sum(tf.multiply(label_matrix, softmax_01), axis = 1)
        weights_JensenFactor = tf.ones_like(target_p, dtype = 'float32') - tf.pow(target_p, JensenFactor)

        # add @ 20180308 to track the trendency of weights_JensenFactor as optimization progresses
        #       in order to better understand the function of weights_JensenFactor
        label_float_pos = tf.cast(label, tf.float32)
        weights_JensenFactor_pos = tf.reduce_sum(tf.multiply(weights_JensenFactor, label_float_pos)) / \
            (tf.reduce_sum(label_float_pos) + 1e-7)
        add_moving_summary(tf.reduce_mean(weights_JensenFactor_pos, name = "weights_JensenFactor_pos"))

        label_float_neg = tf.ones_like(label_float_pos) - label_float_pos
        weights_JensenFactor_neg = tf.reduce_sum(tf.multiply(weights_JensenFactor, label_float_neg)) / \
            (tf.reduce_sum(label_float_neg) + 1e-7)
        add_moving_summary(tf.reduce_mean(weights_JensenFactor_neg, name = "weights_JensenFactor_neg"))

        # BUG: reduction ?
        loss = tf.losses.sparse_softmax_cross_entropy(label, logits, weights = weights_JensenFactor, \
            reduction = tf.losses.Reduction.NONE)

        loss = tf.reduce_mean(loss, name = 'JEentropy-loss')

    def prediction_incorrect(logits, label, topk = 1, name = 'incorrect_vector'):
        with tf.name_scope('prediction_incorrect'):
            x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
        return tf.cast(x, tf.float32, name = name)

    wrong = prediction_incorrect(logits, label, 1, name = 'wrong-top1')
    add_moving_summary(tf.reduce_mean(wrong, name = 'train-error-itop1'))

    return loss


### the proto-type of the ImageNetModel
### SPECIFIC: image_dtype, data_format, size of X & Y, preprocessing, loss function & regularization method
### TODO    : get_logits
class ImageNetModel(ModelDesc):

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    # update @ 20171229: introduce JensenFactor for JensenEnhanced Loss
    # update @ 20180705: introduce output_dims for AESTHETIC_LEVEL experiments
    def __init__(self, data_format = 'NCHW', JensenFactor = 0.0, output_dims = 2, weight_decay = 1e-4):
        if data_format == 'NCHW':
            assert tf.test.is_gpu_available()
        self.data_format = data_format
        self.weight_decay = weight_decay
        self.JensenFactor = JensenFactor

    def _get_inputs(self):
        # despite the setting of self.data_format
        # the original format of the input data is batch_size x H x W x C
        return [InputDesc(self.image_dtype, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image_preprocess(image, bgr = True)
        # transform the data_format if necessary
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        # UNDEFINED
        logits = tf.identity(self.get_logits(image), name = 'logits')
        softmax_logits = tf.nn.softmax(logits, name = 'softmax-logits')

        # definition of loss
        # update @ 20180705: for AESTHETIC_LEVEL experiments, introduce output_dims
        loss = compute_loss_and_error(logits, label, JensenFactor = self.JensenFactor, output_dims = self.output_dims)

        wd_loss = regularize_cost('.*/W', tf.contrib.layers.l2_regularizer(self.weight_decay),
            name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)

        self.cost = tf.add_n([loss, wd_loss], name = 'cost')

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of 224x224 in ``self.data_format``

        Returns:
            batch_size x output_dims logits
        """

    def _get_optimizer(self):
        # NOTE: the learning_rate is set to be 0.1 for initialization
        lr = tf.get_variable('learning_rate', initializer = 0.1, trainable = False)
        tf.summary.scalar('learning_rate', lr)

        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov = True)


### THE END
