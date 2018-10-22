# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:11:33 2018

@author: cdimattina

compLinAutoEncoder.py: Sparse linear auto-encoder for 8x8 image patches
        

"""
# import modules
import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat 

# simulation meta-parameters
wscale      = 0.1;  # scale of the small random weights for model initialization
batchSz     = 1000;
nBat        = 1;
nIterBat    = 1; 
fracTrain   = 0.8;

alpha       = 0.0000; 
beta        = 0.00000;  # hyper-parameter which determines penality size 
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

#W1mat               = wscale*tf.random_normal([nIn,nHid], mean=0.0, stddev=1.0, dtype=tf.float32, name = 'W1mat')
W1mat                = tf.eye(nIn)

# define tf variables and initial values for hidden layer, output layer
hidden_layer_vals   = {'weights':tf.Variable(W1mat)}
output_layer_vals   = {'weights':tf.Variable(tf.transpose(W1mat))}

# input layer is simply a placeholder 
input_layer         = tf.placeholder('float',[None, nIn])
hidden_layer        = tf.contrib.layers.fully_connected(tf.matmul(input_layer,hidden_layer_vals['weights']), nHid, activation_fn=None )  # linear gain
y_pred              = tf.contrib.layers.fully_connected(tf.matmul(hidden_layer,output_layer_vals['weights']), nIn , activation_fn=None )    # multiply by transpose
y_true              = tf.placeholder('float',[None, nIn])

# define our cost function
meansq              = tf.reduce_sum(tf.square(y_pred-y_true)) 
hiddenpenalty       = tf.reduce_sum(tf.abs(hidden_layer))
#weightpenalty       = tf.reduce_sum(tf.square(hidden_layer_vals['weights']))

gradW1_mse          = tf.gradients(ys = meansq, xs = [hidden_layer_vals['weights']]) 
gradW1_sparse       = tf.gradients(ys = hiddenpenalty, xs = [hidden_layer_vals['weights']])
new_W1              = hidden_layer_vals['weights'].assign(hidden_layer_vals['weights'] - alpha * gradW1_mse[0] - beta * gradW1_sparse[0] )
new_W2              = output_layer_vals['weights'].assign(tf.transpose(hidden_layer_vals['weights']))

#gradW2_mse          = tf.gradients(ys = meansq, xs = [output_layer_vals['weights']]) 
#new_W2              = output_layer_vals['weights'].assign(output_layer_vals['weights'] - alpha * gradW2_mse[0] )


# - beta * gradW1_sparse[0]

# overall cost function
# penalizedmeansq     = alpha*meansq + beta*hiddenpenalty #+ gamma*weightpenalty

# define which optimizer we are using
# optimizer           = tf.train.GradientDescentOptimizer(learn_rate).minimize(penalizedmeansq)

# intialize saver
saver               = tf.train.Saver()

# intialize tensorflow session
init                = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    
    for bat in range(nBat):
    
        epoch_hidden_loss = 0
        epoch_meansq = 0
        
        temp = np.random.permutation(nTrain)
        bat_ind = temp[0:batchSz]
        epoch_x = tdata[bat_ind,:]
        
        for i in range(nIterBat):
            [_,_, c, d] = sess.run([new_W1,new_W2, meansq,hiddenpenalty],feed_dict={input_layer: epoch_x, y_true: epoch_x})
            # sess.run([new_W2,meansq],feed_dict={input_layer: epoch_x, y_true: epoch_x})
        epoch_meansq =c
        epoch_hidden_loss =d
            
        print('Mini-Batch',bat + 1, '/', nBat,'-- msq :',epoch_meansq , '-- sparsity:',epoch_hidden_loss)

    save_path = saver.save(sess,"./sparseLinTrainTestFinal.ckpt")

sess.close()

## go through the set
#for bat in range(nBat):
#    epoch_loss = 0
#    epoch_hidden_loss = 0
#    epoch_meansq = 0
#    
#    temp = np.random.permutation(nTrain)
#    bat_ind = temp[0:batchSz]
#    epoch_x = tdata[bat_ind,:]
#    
#    for i in range(nIterBat):  
      #  [_, c, d, e]    = sess.run([optimizer,penalizedmeansq, hiddenpenalty, meansq], feed_dict={input_layer: epoch_x, y_true: epoch_x}) 
      #  epoch_loss +=c
      #  epoch_hidden_loss +=d
      #  epoch_meansq +=e
    
      # print('Mini-Batch',bat + 1, '/', nBat, '-- loss:',epoch_loss,'-- msq :',epoch_meansq , '-- sparsity:',epoch_hidden_loss)

    

