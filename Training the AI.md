## Introduction

#### What Will I Learn?
In this tutorial we will setup and train a Tensorflow-AI to classify the [Special Database 19](https://www.nist.gov/srd/nist-special-database-19).
Our AI will be a Convoluted Neural Network (CNN). If you have never worked with Neural Networks before, this tutorial 
is too advanced. You should first get comfortable with Densely Connected Neural Networks in Tensorflow.  

#### Requirements

- [Preprocessing data for training with tensorflow](https://utopian.io/utopian-io/@leatherwolf/preprocessing-data-for-training-with-tensorflow): 
 In this tutorial I explain how to prepare data for training with tensorflow. I will be
working the data from this tutorial. 
- Python 3
- Tensorflow for Python (`pip install tensorflow`)
- Basic knowledge about numpy. I recommend [this](https://utopian.io/utopian-io/@scipio/learn-python-series-11-numpy-part-1) tutorial by scipio. 

#### Difficulty
- Advanced

### References
All the files and scripts presented are available on my GitHub Page [here](https://github.com/frereit/TensorflowHandwritingRecognition).
This tutorial is part of a series. I will explain in detail how I achieved the handwriting recognition engine with 
tensorflow.

##So, let's get started!

We will be working with [training_32x32.py](https://github.com/frereit/TensorflowHandwritingRecognition/blob/master/training_32x32.py).
I recommend that you open the finished Python file while reading this tutorial so that you can follow along, but I will
also show the code here.

#####1. Imports
```python
import numpy as np
import tensorflow as tf
import math
import sys
import time
import datetime
import os
```
We need numpy to work with the data arrays. Tensorflow will be the key component, as we are using it to create the AI.
Math provides some neat helper functions. We need sys to read in the arguments. We will use that to specify the amount
of training cycles via the command line. time and datetime are used for logging and os is used to set the tensorflow 
debug level. 

#####2. Handling the data
I created a small Helper class to work with our train and test sets. It will supply the batches for training. 

```python
class NISTHelper():
    def __init__(self, train_img, train_label, test_img, test_label):
        self.i = 0
        self.test_i = 0
        self.training_images = train_img
        self.training_labels = train_label
        self.test_images = test_img
        self.test_labels = test_label
```
`self.i` and `self.test_i` are used to keep track of the current index in the train and test arrays.\
`self.training_images` and `self.test_images` will be numpy arrays with the shape [?,32,32,1]. ? is the number of images.
32,32 corresponds to the resolution of our images The 1 corresponds to the amount of our color channels. Our images are
greyscale, so we only have 1 color channel. \
`self.training_labels` and `self.test_labels` will be numpy array with the shape [?,47]. We have 47 different labels in
our set and we one-hot-encode them. One-hot-encoding means that instead of storing our result as a value between 0 and 46,
we store the result in an array with 46 zeros and a one at the index of the class. E.g an image with class 3 might have
this label: [0,0,0,3,0,...,0]. 

```python
    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size]
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y
```
This function returns the next `batch_size` images. `x` are the images and `y` are the corresponding label. \
`self.i = (self.i + batch_size) `: If we reached the limit of our array, put `self.i` back to the beginning. \
`return x,y`: Return a tuple with the arrays.

`test_batch(self, batch_size)` is basically the same function, except we use `self.test_i` and `self.test_images`/`self.
test_labels`. 

And that's our Helper function done.

In `main()`

