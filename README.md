# MPADA
The implementation of $MP_{ada}$ in Attention-based Multi-patch Aggregation for Image Aesthetic Assessment [[pdf](http://chongyangma.com/publications/am/2018_am_paper.pdf)].

## Framework
![SystemOverview](https://github.com/Openning07/MPADA/blob/master/FromPaper/SystemOverview.png "MPADA")

System overview. We use an attention-based objective to enhance training signals by assigning relatively
larger weights to misclassified image patches.

## Experiments
### Requirements
- python == 3.6
- tensorflow == 1.2.1
- tensorpack == 0.6
#### Notes


### Instructions for Results in the paper
    python AVA2012-resnet_20180808_Revised.py --gpu 2 --data $YOUR_DATA_DIR$/AVA2012 -aesthetic_level 2 --crop_method_TS RandomCrop --repeat_times 15 --load $YOUR_CHECKPOINT_DIR$/checkpoint --mode resnet -d 18 --eval 
#### Notes
 - $YOUR_DATA_DIR$
 - $YOUR_CHECKPOINT_DIR$
### Desired Outputs

## Citation
Please cite the following paper if you use this repository in your reseach.
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
