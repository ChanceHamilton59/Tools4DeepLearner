# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:11:33 2018

@author: cdimattina

compLinAutoEncoder.py: Compressive linear auto-encoder for 8x8 image patches

"""

# import modules
import os
import numpy as np

import tensorflow as tf
from scipy.io import loadmat 



# simulation meta-parameters
wscale      = 0.1;  # scale of the small random weights for model initialization
learn_rate  = 0.1;
batchSz     = 1000;
nEpochs     = 10000;
fracTrain   = 0.8;

alpha       = 1; 
beta        = 0.25;  # hyper-parameter which determines penality size 
gamma       = 0;  

# define list of training data filenames 
trainDir            = 'TrainingData'
TrainingDataFiles   = []

for r, d, f in os.walk(trainDir):
    for file in f:
        TrainingDataFiles.append(os.path.join(r,file))
        
# load files and put them into a matrix
tdata = loadmat(TrainingDataFiles[0])['X']          
for i in range(1,len(TrainingDataFiles)):            
    td      = loadmat(TrainingDataFiles[i])['X']
    tdata   = np.concatenate((tdata,td), axis = 1)

# put observations in rows
tdata   = tdata.transpose()

# get total number of stimuli
nIn     = tdata.shape[1]
nStim   = tdata.shape[0]
nTrain  = round(fracTrain*nStim)

# we compress the images by a factor of 4
nHid  = round(nIn)   

print("Training Set Size   : " + str(nTrain))
print("Stimulus Dimensions : " + str(nIn))
print("Hidden Layer        : " + str(nHid))

"""
This part of the code uses Tensorflow(R) to set up our definitions for 
the neural network. The neural network will be comprised of a single linear
hidden layer which compresses the input image down to nHid dimensions
"""

W1mat               = tf.random_normal([nIn,nHid], mean=0.0, stddev=wscale, dtype=tf.float32, name = 'W1mat')
# W1mat = tf.eye(nIn,num_columns = None, batch_shape = None ,dtype=tf.float32, name ='W1mat')

# define tf variables and initial values for hidden layer, output layer
hidden_layer_vals   = {'weights':tf.Variable(W1mat)}
output_layer_vals   = {'weights':tf.Variable(tf.transpose(W1mat))}

# input layer is simply a placeholder 
input_layer         = tf.placeholder('float',[None, nIn])
hidden_layer        = tf.contrib.layers.fully_connected(tf.matmul(input_layer,hidden_layer_vals['weights']), nHid, activation_fn=None )  # linear gain
y_pred              = tf.matmul(hidden_layer,output_layer_vals['weights'])    # multiply by transpose
y_true              = tf.placeholder('float',[None, nIn])

# define our cost function
meansq              = tf.reduce_mean(tf.square(y_pred-y_true)) 
hiddenpenalty       = tf.reduce_mean(tf.abs(hidden_layer))
weightpenalty       = tf.reduce_mean(tf.square(hidden_layer_vals['weights']))


penalizedmeansq     = alpha*meansq + beta*hiddenpenalty + gamma*weightpenalty

# define which optimizer we are using
optimizer           = tf.train.GradientDescentOptimizer(learn_rate).minimize(penalizedmeansq)

# intialize saver
saver               = tf.train.Saver()

# intialize tensorflow session
init                = tf.global_variables_initializer()
sess                = tf.Session() 
sess.run(init)

## go through the set
for epoch in range(nEpochs):
    epoch_loss = 0
    epoch_hidden_loss = 0
    epoch_meansq = 0
    
    # define which optimizer we are using
    #optimizer           = tf.train.GradientDescentOptimizer(learn_rate).minimize(penalizedmeansq)
    
    
    for i in range(int(nTrain/batchSz)):
        epoch_x = tdata[i*batchSz:(i+1)*batchSz]
        [_, c, d, e]    = sess.run([optimizer,penalizedmeansq, hiddenpenalty, meansq], feed_dict={input_layer: epoch_x, y_true: epoch_x}) 
        epoch_loss +=c
        epoch_hidden_loss +=d
        epoch_meansq +=e
    
    print('Epoch',epoch + 1, '/', nEpochs, '-- loss:',epoch_loss,'-- msq :',epoch_meansq , '-- sparsity:',epoch_hidden_loss)
#    if(np.fmod(epoch + 1,1000)==0):
#        save_path = saver.save(sess,"./sparseLinTrainFinal.ckpt")
#        print('saving data!')
#        time.sleep(0.5)



save_path = saver.save(sess,"./sparseLinTrainFinal.ckpt")

sess.close()