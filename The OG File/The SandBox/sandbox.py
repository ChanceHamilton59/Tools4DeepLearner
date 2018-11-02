# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:49:24 2018

@author: cjhamilton4176
"""

from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.io import loadmat
import matlab.engine

eng = matlab.engine.start_matlab()

x = loadmat('test.mat')

tdata = x['X']
tdata = tdata[:,0]

tdata = np.asarray(eng.lowpassFilter('test.mat')) 
print(tdata.shape)