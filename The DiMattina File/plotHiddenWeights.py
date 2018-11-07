# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:53:40 2018

@author: cdimattina

Description:    This simple module imports and plots the hidden weights of 
                a three-layer neural network
"""

import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

fName           = "./AutoEncoderSparseSigFinal.ckpt"

sess            = tf.Session()
saver           = tf.train.Saver()
saver.restore(sess,fName)
   
X               = final_W1

nHid            = X.shape[1]            # get the number of hidden units
imSz            = int(np.sqrt(X.shape[0]))   # size of each image patch
plotGap2        = 1                     # half the gap to leave between each image patch
imSzPlot        = int(imSz + 2*plotGap2)    

layoutSqDim     = int(np.ceil(np.sqrt(nHid)))  # find dimensions for layout
arrayImage      = np.zeros((layoutSqDim*imSzPlot,layoutSqDim*imSzPlot))

k = 0
for i in range(layoutSqDim):
    for j in range(layoutSqDim):
        if k < nHid:    
            littleImPlot = np.zeros((imSzPlot,imSzPlot))
            littleImPlot[(plotGap2):(imSzPlot-plotGap2),(plotGap2):(imSzPlot-plotGap2)] = np.reshape(X[:,k],(imSz,imSz)) 
            arrayImage[ (i*imSzPlot ):((i+1)*imSzPlot) , (j*imSzPlot ):((j+1)*imSzPlot) ] = littleImPlot;
            k = k + 1
        

plt.imshow(arrayImage,  cmap='Greys')
plt.show()

sess.close()