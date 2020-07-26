# 3DHPose

3DHPose is an improved version of [mvpose](https://github.com/zju3dv/mvpose).

## Installation

Please follow the installation and usage steps from the original [repo](https://github.com/zju3dv/mvpose)

## Purpose

This project aims to estimate multi-person 3D Human poses from multiple views. Because the original [repo](https://github.com/zju3dv/mvpose) achieves high accuracy, the main purpose
of this project is to speed up the process as much as possible.

## Changes

All the backbone networks were changed in order to achieve a better inference time.
### Object detector used: [CenterNet](https://github.com/xingyizhou/CenterNet)
### ReId module used: [OsNet](https://github.com/KaiyangZhou/deep-person-reid)
### 2D Human pose detector: [CPN](https://github.com/chenyilun95/tf-cpn) using only GlobalNet with ResNet50 as the backbone network.

## Results
| Method | Time | Accuracy |
| ------ |---|---|
|Original Repo| 12:21(min) | 96.36% |
|3DHPose| 02:14(min) | 96.21% |

Both configurations were tested using a Tesla P100 GPU, on Google Colab. The test dataset is the campus one from the original [repo](https://github.com/zju3dv/mvpose).

## Citation
``` @article{dong2019fast,
  title={Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views},
  author={Dong, Junting and Jiang, Wen and Huang, Qixing and Bao, Hujun and Zhou, Xiaowei},
  journal={CVPR},
  year={2019}
} ```
