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

## Usage

### List the available layers and the number of channels

To check the available layer names and channel numbers for the deepdream program. 

*Input*

```shell
python deepdream_api.py -l
```
*Output*
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
In the output, the layer name is on the left, the number of channels in the layer is on the right.


### Preview the feature pattern of the neural network

To preview the feature pattern learned in a certain channel of a certain layer in the neural network. This is helpful for the user to select layers and channels used for image modification.

*Input*

-p layer_name channel_number, --preview layer_name channel_number

```shell
python deepdream_api.py -p mixed4d_3x3_bottleneck_pre_relu 20 feature_pattern.jpeg
```

*Output*

![](outputs/feature_pattern.jpeg)


### Render the image with the features from the neural network

Apply feature pattern learned in a certain channel of a certain layer in the neural network to the image that the user provided.

*Input*

-r image_path layer_name channel_number, --render image_path layer_name channel_number

```shell
python deepdream_api.py -r inputs/pilatus800.jpg mixed4d_3x3_bottleneck_pre_relu 20 railgun_deepdream.jpeg
```

![](inputs/pilatus800.jpg)

*Output*

![](outputs/railgun_deepdream.jpeg)

### References

* [Google Official DeepDream Tutorial in Caffe](https://github.com/google/deepdream/blob/master/dream.ipynb)
* [Google Official DeepDream Tutorial in TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb)
* [Google Research Blog](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
* [Siraj Raval's Video Tutorial](https://www.youtube.com/watch?v=MrBzgvUNr4w)
* [Siraj Raval's Code](https://github.com/llSourcell/deep_dream_challenge/blob/master/deep_dream.py)



