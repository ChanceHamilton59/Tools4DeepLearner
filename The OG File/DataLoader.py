from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.io import loadmat

x = loadmat('test.mat')
print(np.shape(x))