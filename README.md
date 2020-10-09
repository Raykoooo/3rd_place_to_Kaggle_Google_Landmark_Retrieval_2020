## Introduction
This is the source code of the 3rd place to [Kaggle Google Landmark Retrieval 2020](https://www.kaggle.com/c/landmark-retrieval-2020).

A detailed description of our solution: https://arxiv.org/abs/2008.10480

## Envs
- Pytorch 1.1
- Python 3.6.5
- CUDA 9.0 
- Nvidia Tesla P40 * 8

install
```
pip install -r requirements.txt
```

`apex` :  Tools for easy mixed precision and distributed training in Pytorch
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Data
Data directory structure:
- `data_list/label_81313.txt`: GLD v2 clean 
- `data_list/label_sub_212843.txt`: GLD v2 cluster
- `train`: GLD v2 train, download from https://github.com/cvdfoundation/google-landmark

Download pretrained models (ImageNet)
- resnet-152: https://drive.google.com/file/d/1EqIqTlwaSp9Ta3fBZmzIKw-DkQp22pW-/view?usp=sharing
- resnest-200: https://drive.google.com/file/d/1l0IGCgP8zQuTFPxkIPp5GuX4DP5ltEye/view?usp=sharing

```
|-- data
  |-- data_list
    |-- label_81313.txt
    |-- label_sub_212843.txt
  |-- train # GLD v2
    |-- 0
    |-- 1
    ...
    |-- f
|-- pretrained_models
  |-- 7x7resnet152-imagenet.pth
  |-- resnest200-75117900.pth
...
```

## Training
GPU_NUM: 8

```
# ResNest200
sh scripts/cls/resnest200_v2clean.sh 8

sh scripts/cls/resnest200_v2cluster.sh 8

# ResNet152
sh scripts/cls/resnet152_v2clean.sh 8

sh scripts/cls/resnet152_v2cluster.sh 8
```

## Convert
pytorch -> onnx -> tensorflow

```
cd scripts/interface

sh merge_512.py

sh convert_merge.sh merge_second.onnx 512
```

final submission: `scripts/interface/merge_second`