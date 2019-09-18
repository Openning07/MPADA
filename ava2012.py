#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ava2012.py
# Author: Kekai Sheng
import os
import six
import tqdm
import pickle
import tarfile

import numpy as np
import cv2

from ...utils import logger
from ...utils.fs import mkdir_p, download, get_dataset_path
from ...utils.timer import timed_operation
from ..base import RNGDataFlow

__all__ = ['AVA2012Meta', 'AVA2012']


###
INTERPOLATION_POOL = [ cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC ]


###
class AVA2012Meta(object):
    """
    Provide methods to access metadata for AVA2012 dataset.
    """

    def __init__(self, dir=None):
        if dir is None:
            dir = get_dataset_path('AVA2012')
        self.dir = dir
        mkdir_p(self.dir)

    ### read the (img_name, aesthetic_score) pairs from [train, val].txt
    def get_image_list(self, name, delta_moderate = 0, raw_score_flag = False):
        """
        Args:
            name (str): 'train' or 'val'
        Returns:
            ret, num_pos, num_neg, num_discarded
            ret          : list of (image filename, aesthetic score)
                           The aesthetic score of each image is the average value of the aesthetic assessment
                           from the OnlineEvaluators in DpChallenge.com. Range from 1 to  10
            num_pos      : the number of positive instance in the return ret
            num_neg      : the number of negative instance in the return ret
            num_discarded: the number of images discarded, whose aesthetic values are in
                           the ambiguous range (5 - delta_moderate, 5 + delta_moderate)
        """
        assert name in ['train', 'val'], \
            "*** At the present stage, AVA2012 dataset has only train.txt or val.txt for experiments ***"

        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname), \
            "*** The txt file of image list ({0}) to load does not exist  ***".format(fname)

        with open(fname) as f:
            ret = []
            num_pos = 0
            num_neg = 0
            num_discarded = 0
            for line in f.readlines():
                line_ = line.strip().split()
                img_name = line_[0]
                aesthetic_score = float(line_[1])

                ### as it is performed in other papers that: to mitigate the side effect from the
                ### disambiguation in the aesthetic dataset, we remove the images with the score
                ### ranging from (5 - delta_moderate, 5 + delta_moderate)
                if (aesthetic_score > 5 - delta_moderate) and (aesthetic_score < 5 + delta_moderate):
                    num_discarded += 1
                    continue;
                ### 0 for [1, 5 - delta_moderate] and 1 for [5 + delta_moderate, 10]
                if aesthetic_score >= 5 + delta_moderate:
                    num_pos += 1
                else:
                    num_neg += 1

                ret.append((img_name.strip(), 1 if (aesthetic_score >= 5 + delta_moderate) else 0))

        assert len(ret), \
            "*** Failed to load a list of (img_name, aesthetic_score) out of {0} or delta_moderate {1} is too large ***".format(fname, delta_moderate)

        return ret, num_pos, num_neg, num_discarded


class AVA2012(RNGDataFlow):
    """
    Produces uint8 AVA2012 images of shape [h, w, 3(BGR)], and a float value for the aesthetic score
    """
    def __init__(self, dir, name, delta_moderate = 0, meta_dir = None,
                 CAM_dir_pkl = None, CAMCropR = False, CropSize = 224,
                 shuffle = None, class_aware = False, raw_score_flag = False):
        """
        Args:
            dir (str)        : A directory containing a subdir named ``name``
            name (str)       : 'train' or 'val'
            delta_moderate   : a float value for the specific boundary, i.e.,
                               (5 - delta_moderate, 5 + delta_moderate), to
                               be disregared in the low([1, 5 - delta_moderate]) vs
                               high([5 + delta_moderate, 10]) quality image classification
                             
            CAM_dir_pkl(str) : the directory to load variables for the purpose of CAMCrop
            shuffle (bool)   : shuffle the dataset. Defaults to True if name=='train'.
            class_aware (b)  : the flag to control class aware balanced sampling

        Examples:

        `dir` should have the following structure:
        dir/
            images/
                ***.jpg

        """
        assert name in ['train', 'val'], \
            "*** At the present, only train.txt or val.txt exist in AVA2012, {0} does not exist ***".format(name)
        assert os.path.isdir(dir), \
            "*** The inputed dir for the root dir of AVA2012 ({0}) does not exist ***".format(dir)

        self.full_dir = os.path.join(dir, 'images') ### for convenience, we put all images in a folder called images
                                                    ### in the root directory of AVA2012
        assert os.path.isdir(self.full_dir), \
            "*** The directory to load images ({0}) does not exist ***".format(self.full_dir)

        self.name = name

        # add @ 20171124 for the purpose of CAMCrop
        # whenever we can load the corresponding .pkl file from CAM_dir_pkl, we perform CAMCrop
        # TODO
        self.CAM_dir_pkl = CAM_dir_pkl
        # add @ 20171125 for a positive experiment ...
        self.CAMCropR = CAMCropR
        self.CropSize = CropSize

        if shuffle is None:
            shuffle = name == 'train'

        assert delta_moderate >= 0, \
            "*** The inputed delta_moderate ({0}) for disambiguation must leq zero ***".format(delta_moderate)
        self.delta_moderate = delta_moderate

        self.shuffle = shuffle

        ### TODO to refine ...
        meta = AVA2012Meta(meta_dir)
        self.imglist, self.num_pos, self.num_neg, self.num_discarded = meta.get_image_list(\
            name, delta_moderate, raw_score_flag)

        ### added @ 2017/10/31
        ### class_aware only works iff it is for train data-flow and class_aware is set to be True
        self.class_aware = ((name == 'train') and (class_aware))
        self.imglist_byCls = {0 : [], 1 : []}
        for imgName, imgCls in self.imglist:
            self.imglist_byCls[imgCls].append(imgName)

        ### added @ 2017/10/31
        ### for future usage (e.g., encode more information into the representation space for aesthetic assessment purpose)
        self.raw_score_flag = raw_score_flag

    ### return the size of data points
    def size(self):
        return (2 * max(self.num_pos, self.num_neg)) if (True == self.class_aware) else (self.num_pos + self.num_neg)

    # add @ 20171124 for CAMcrop with CAM to adaptively crop 224x224
    def sample_position_by_CAM(self, im, imgName):
        # @ 2017/09/16 for random interpolation
        interpolation_type = np.random.randint(len(INTERPOLATION_POOL))

        # resizeShortestEdge to 340 if self.CropSize == 299 else 256 (for 224)  <-- NOTE
        desired_shortestEdge = 340.0 if (self.CropSize == 299) else 256
        h, w = im.shape[:2]
        if h > w:
            h = int(desired_shortestEdge * h / w)
            w = int(desired_shortestEdge)
        else:
            w = int(desired_shortestEdge * w / h)
            h = int(desired_shortestEdge)
        #print('*=* DEBUG: h = %d, w = %d' %(h, w))
        im = cv2.resize(im, (w, h), interpolation = INTERPOLATION_POOL[ interpolation_type ])    ## <-- NOTE

        # load CAMap
        # @ 2017/09/18 maybe we should resize the CAM to a larger size (says, 16x16
        #   to introduce more variety during the p_position sampling process
        CAMfile = self.CAM_dir_pkl + '/' + os.path.splitext(imgName)[0] + '.pkl'
        if os.path.isfile(CAMfile):
            # if the CAMfile does not exist (due to the remainder = True)
            # then randomly value the CAM7x7
            with open(CAMfile, 'rb') as input_stream:
                CAM = pickle.load(input_stream)
            CAM7x7 = CAM['CAM{}'.format(CAM['GT01'])]  ## <-- NOTE
        else:
            CAM7x7 = np.random.random((7, 7))
            #assert False, '*** NO PKL OF CAM FOR {} ***'.format(imgName)

        # normalize CAM7x7 for the possibility
        CAM7x7_min = CAM7x7.min()
        CAM7x7_max = CAM7x7.max()
        CAM7x7 = (CAM7x7 - CAM7x7_min + 1e-7) / (CAM7x7_max - CAM7x7_min + 1e-7)

        # add @ 20171125 for CAMCropR
        if self.CAMCropR:
            CAM7x7 = 1.0 - CAM7x7

        Z_CAM7x7 = np.sum(CAM7x7)
        CAM7x7 = CAM7x7 / Z_CAM7x7

        # sample based on the p in CAM7x7
        pos_lst = [ (CAM7x7[r, c], (r, c)) for r in range(CAM7x7.shape[0]) for c in range(CAM7x7.shape[1]) ]
        pos_acc_lst = [ (sum( [ dpp[0] for dpp in pos_lst[:i+1] ] ), dp[1]) for i, dp in enumerate(pos_lst) ]
        p_sample = np.random.random(1)
        p_index = len(pos_acc_lst) - 1
        while p_index > 0:
            if pos_acc_lst[p_index - 1][0] <= p_sample:
                break;
            else:
                p_index -= 1
        p_position = pos_acc_lst[p_index][1]

        # 7x7 -> hxw
        p_position = (int(p_position[0] / 7.0 * h), int(p_position[1] / 7.0 * w))
        # @ 2017/09/18, in order to increase the variety in the CAMcrop sampling
        #p_position = (np.random.randint(round(p_position[0] / 8.0 * h), round((p_position[0] + 1) / 8.0 * h) + 1), \
        #              np.random.randint(round(p_position[1] / 8.0 * w), round((p_position[1] + 1) / 8.0 * w) + 1))
        #print('*=* DEBUG: p_position = ', p_position)
        # @ 2017/09/22  it seems that the follow two lines of code pose detrimental effect on the training ...
        #               To verify ...  TRUE
        #               If it is the truth, then we should pay attention to the design of the way to adopt
        #               CAMcrop during the training

        # adapted @ 20171124
        # GoogleNetResize -> ShortestEdgeResize, since the ShortestEdgeResize seems to be better in our case
        desired_h = self.CropSize
        desired_w = self.CropSize
        #print('*=* DEBUG: desired_h = %d, desired_w = %d' %( desired_h, desired_w ))

        p_position = (min(max((desired_h+2)//2, p_position[0]), h - (desired_h+2)//2), \
                      min(max((desired_w+2)//2, p_position[1]), w - (desired_w+2)//2))
        p_xyxy = (int(p_position[0] - desired_h // 2), int(p_position[0] + desired_h // 2), \
                  int(p_position[1] - desired_w // 2), int(p_position[1] + desired_w // 2))
        #print('*=* DEBUG: p_xyxy = ', p_xyxy)
        CROP_SIZE = self.CropSize  ## <-- NOTE
        return cv2.resize(im[p_xyxy[0] : p_xyxy[1] + 1, p_xyxy[2] : p_xyxy[3] + 1, :], \
                          (CROP_SIZE, CROP_SIZE), interpolation = INTERPOLATION_POOL[ interpolation_type ])


    ### create the image data generator
    def get_data(self):
        if self.class_aware:
            # for class balance sampling
            idxs = np.arange(max(self.num_pos, self.num_neg))
            if self.shuffle:
                np.random.shuffle(idxs)

            for k in idxs:
                for kk in self.imglist_byCls.keys():
                    aesthetic_score = kk
                    fname = self.imglist_byCls[kk][k % len(self.imglist_byCls[kk])]
                    fname = os.path.join(self.full_dir, fname)

                    im = cv2.imread(fname, cv2.IMREAD_COLOR)
                    assert im is not None, fname

                    if im.ndim == 2:
                        im = np.expand_dims(im, 2).repeat(3, 2)

                    yield [im, aesthetic_score]
            # LOG: class aware balancing sampling seems to pose detrimental effect ...
            # TODO

        else:
            idxs = np.arange(len(self.imglist))
            if self.shuffle:
                np.random.shuffle(idxs)
         
            for k in idxs:
                fname, aesthetic_score = self.imglist[k]
                fname = os.path.join(self.full_dir, fname)
                #print('    DEBUG: load %s (cls = %d)' %(fname, aesthetic_score))
         
                im = cv2.imread(fname, cv2.IMREAD_COLOR)
                assert im is not None, fname
                ### NOTE: whether there is an alternative method to deal with GRAY image ?
                if im.ndim == 2:
                    im = np.expand_dims(im, 2).repeat(3, 2)

                # add @ 20171124
                if (self.name == 'train') and (self.CAM_dir_pkl != None):
                    im = self.sample_position_by_CAM(im, self.imglist[k][0])
         
                yield [im, aesthetic_score]

    ### for the convenience of exhibition ...
    def __str__(self):
        return "==> This is an instance of AVA2012 data iterator ...\n"    + \
               "    full_dir         : {}\n".format(self.full_dir)         + \
               "    name             : {}\n".format(self.name)             + \
               "    delta_moderate   : {}\n".format(self.delta_moderate)   + \
               "    number of images : {}\n".format(len(self.imglist))     + \
               "    number of pos    : {0} ({1:.3f})\n".format(self.num_pos, (self.num_pos) / (self.num_pos+self.num_neg)) + \
               "    number of neg    : {0} ({1:.3f})\n".format(self.num_neg, (self.num_neg) / (self.num_pos+self.num_neg)) + \
               "    number of med    : {}\n".format(self.num_discarded)    + \
               "    class_aware      : {}\n".format(self.class_aware)      + \
               "    size of queue    : {}\n".format(self.size())           + \
               "    shuffle          : {}\n".format(self.shuffle)          + \
               "    raw_score_flag   : {}\n".format(self.raw_score_flag)   + \
               "    CAM_dir_pkl      : {}\n".format(self.CAM_dir_pkl)      + \
               "    CAMCropR         : {}\n".format(self.CAMCropR)         + \
               "    CropSize (if)    : {}\n".format(self.CropSize)

### The end of definition of AVA2012


try:
    import cv2
except ImportError:
    from ...utils.develop import create_dummy_class
    AVA2012 = create_dummy_class('AVA2012', 'cv2')  # noqa

if __name__ == '__main__':
    meta = AVA2012Meta()


##### THE END
