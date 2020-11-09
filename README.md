# MPADA
The implementation of $MP_{ada}$ in Attention-based Multi-patch Aggregation for Image Aesthetic Assessment [pdf](http://chongyangma.com/publications/am/2018_am_paper.pdf), the method for SOTA aesthetic visual assessment performance on AVA benchmark. For more comparisons on AVA, please refer to the [page](https://paperswithcode.com/sota/aesthetics-quality-assessment-on-ava) on PaperWithCode.

## Framework
<p align="center">
  <img src="https://github.com/Openning07/MPADA/blob/master/FromPaper/SystemOverview.png" alt="CMM" width="52%">
</p>

System overview. We use an attention-based objective to enhance training signals by assigning relatively
larger weights to misclassified image patches.

## Experiments
### Requirements
* python == 3.6
* tensorflow == 1.2.1
* tensorpack == 0.6
#### Notes
 - Tensorpack does not implement AVA2012. You need to put the **ava2012.py** in *AVA_info* in the folder of tensorpack.dataflow.dataset.
 - For the information of training and test split of AVA benchmark, please refer to **AVA_train.lst** and **AVA_test.lst** in *AVA_info*.

### Instructions for Results in the paper
    python AVA2012-resnet_20180808_Revised.py --gpu 2 --data $YOUR_DATA_DIR$/AVA2012
            --aesthetic_level 2 --crop_method_TS RandomCrop --repeat_times 15
            --load $YOUR_CHECKPOINT_DIR$/checkpoint --mode resnet -d 18 --eval 
### Desired Outputs
    TODO

#### Notes
 - $YOUR_DATA_DIR$ : The directory you put *images* of the AVA benchmark.
 - $YOUR_CHECKPOINT_DIR$ : The directory you save the *checkpoint* files of the models.
 - Result might not be reproduced due to several factors: different version of cv2, different CUDA version, different split of training/test.

## Citation
Please cite the following paper if you use this repository in your reseach~ Thank you ^ . ^
```
@inproceedings{sheng2018attention,
  title={Attention-based multi-patch aggregation for image aesthetic assessment},
  author={Sheng, Kekai and Dong, Weiming and Ma, Chongyang and Mei, Xing and Huang, Feiyue and Hu, Bao-Gang},
  booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
  pages={879--886},
  year={2018},
  organization={ACM}
}
```
