# Google DeepDream Local API

Lei Mao

9/17/2017

Department of Computer Science

University of Chicago

## Description

The development of this API is still in progress. More functions will be added in the future.

## Requirements

* Python 3.6

## Dependencies

* tensorflow 1.3
* numpy
* PIL
* os
* sys
* zipfile
* six
* argparse

## 

### List the available layers and the number of channels
Input
```shell
python deepdream_api.py -l
```
Output
```shell
import/conv2d0_pre_relu/conv 64
import/conv2d1_pre_relu/conv 64
import/conv2d2_pre_relu/conv 192
import/mixed3a_1x1_pre_relu/conv 64
import/mixed3a_3x3_bottleneck_pre_relu/conv 96
import/mixed3a_3x3_pre_relu/conv 128
import/mixed3a_5x5_bottleneck_pre_relu/conv 16
import/mixed3a_5x5_pre_relu/conv 32
import/mixed3a_pool_reduce_pre_relu/conv 32
import/mixed3b_1x1_pre_relu/conv 128
import/mixed3b_3x3_bottleneck_pre_relu/conv 128
import/mixed3b_3x3_pre_relu/conv 192
import/mixed3b_5x5_bottleneck_pre_relu/conv 32
...
```
Left is the layer name, right is the number of channels in the layer.


### Preview the feature pattern of the neural network
Input
```shell
python deepdream_api.py -p mixed4d_3x3_bottleneck_pre_relu 20 feature_pattern.jpeg
```
Output
![](outputs/feature_pattern.jpeg)


Input
```shell
python deepdream_api.py -r inputs/pilatus800.jpg mixed4d_3x3_bottleneck_pre_relu 20 railgun_deepdream.jpeg
```
![](inputs/pilatus800.jpg)
Output
![](outputs/railgun_deepdream.jpeg)



