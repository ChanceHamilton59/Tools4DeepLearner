"""
Created on Mon Jun 25 11:54:52 2018

@author: Chance Hamilton, Chris DiMattina

Description: This is a generic Autoencoder that can be used for a range of tasks
                This was modified 
             
             
             
Training Data: Currently this is set up to train and test on the MNIST Dataset
               But I leave it up to the team to repurpose it to train on the data
               set that Dr. Dimattina wants.
"""

from __future__ import division, print_function, absolute_import
from scipy.io import loadmat 
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Define training parameters
###############################################################################
wscale              = 0.01;  # scale of random weights for model initialization
beta                = 5;
gamma               = 0.01;
actprob             = 0.01;  # desired proportion of active units 

learning_rate       = 0.1
num_steps           = 8000
batch_size          = 1000
display_step        = 200
examples_to_show    = 4

###############################################################################
# This section loads the training dataset
###############################################################################
#trainDir            = 'TrainingData8'
#TrainingDataFiles   = []

#for r, d, f in os.walk(trainDir):
#    for file in f:
#        TrainingDataFiles.append(os.path.join(r,file))
        
# load files and put them into a matrix
#tdata = loadmat(TrainingDataFiles[0])['X']          
#for i in range(1,len(TrainingDataFiles)):            
#    td      = loadmat(TrainingDataFiles[i])['X']
#    tdata   = np.concatenate((tdata,td), axis = 1)

# load whitening matrix
tdata = loadmat('FieldWht8.mat')['Xtrain']    
#tdata = np.matmul(wMat,tdata)    

# put observations in rows
tdata       = tdata.transpose()

###############################################################################
# Define neural network graph
###############################################################################
num_input   = tdata.shape[1]
num_train   = tdata.shape[0]
pixel_size  = int(np.sqrt(float(num_input)))

# compressive auto-encoder reduces dimensionality
# num_hidden  = round(num_input/2)
# sparse auto-encoder keeps dimensionality same or larger
num_hidden = 64 #round(num_input/2)

# this placeholder will hold the training batch
X = tf.placeholder("float", [None, num_input])


## These are a hash map to variables that are used as the weights and biases in 
## the neural network layers. Note that the layers have differnet number of filters
## and that the input passes through these layers and take the shape of the layers.
## It helps to think of a network as a pipline.
weights = {
    'encoder_h1': tf.Variable(wscale*(-1*tf.ones([num_input, num_hidden]) + 2*tf.random_uniform([num_input, num_hidden]))),
    'decoder_h1': tf.Variable(wscale*(-1*tf.ones([num_hidden, num_input]) + 2*tf.random_uniform([num_hidden, num_input])))
}

biases = {
    'encoder_h1' : tf.Variable(tf.zeros([1, num_hidden])),
    'decoder_h1' : tf.Variable(tf.zeros([1, num_input]))
}

## Building the encoder This layer has two layers that are concatinated into a 
## a single layer. This layer should reduce the input vector to a matrix of diffrent
## deminsions.
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.sigmoid(tf.matmul(x, weights['encoder_h1']) + biases['encoder_h1'] )
    return layer_1

## Building the decoder This is simular to the process that is happening in the
## encoder code but this layer further reduces the demensionality.
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
   # layer_1 = tf.matmul(x, weights['decoder_h1'])
   layer_1 =  tf.sigmoid(tf.matmul(x, weights['decoder_h1'] + biases['decoder_h1'] ) )
   return layer_1

## Construct model by forming the encoded input and the decoded encoder
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

## Prediction 
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

## Define loss and optimizer, minimize the squared error
errorterm       = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

## Define the sparsity penalty used in the Stanford CS294 example
actgoal         = actprob*tf.ones([1,num_hidden],tf.float32)
oneminactgoal   = 1-actgoal
hidact          = tf.reduce_mean(encoder_op,0)
oneminhidact    = 1-hidact

meanhidact      = tf.reduce_mean(hidact)

sparsevec       = tf.multiply(actgoal, tf.log(tf.divide(actgoal,hidact))) + tf.multiply(oneminactgoal,tf.log(tf.divide(oneminactgoal,oneminhidact)))
sparseterm      = (tf.reduce_mean(sparsevec))

weightpenalty   = tf.reduce_mean(tf.pow(weights['decoder_h1'],2)) + tf.reduce_mean(tf.pow(weights['encoder_h1'],2))


loss            = errorterm + beta*sparseterm + gamma*weightpenalty

## Note this is not the only optimizer and infact we should look into which one we
## should use for our project!!!!! ADAMOptimizer is a well known and utilize option. 
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#
## Initialize the variables (i.e. assign their default value)
init    = tf.global_variables_initializer()
saver   = tf.train.Saver()

###############################################################################
# Start Training
###############################################################################

# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        temp    = np.random.permutation(num_train)
        bat_ind = temp[0:batch_size]
        batch_x = tdata[bat_ind,:]


        # Run optimization op (backprop) and cost op (to get loss value)
        _, l , e, s, h = sess.run([optimizer, loss, errorterm, sparseterm, meanhidact], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f Error: %f Sparse: %f Hidden Act: %f' % (i, l, e, s, h))


    final_W1 = weights['encoder_h1'].eval()
    final_W2 = weights['decoder_h1'].eval()

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = examples_to_show
    k1 = 1
    k2 = 1
    canvas_orig = np.empty((pixel_size * n, pixel_size * n))
    canvas_recon = np.empty((pixel_size * n, pixel_size * n))
    for i in range(n):
   
        g = sess.run(decoder_op, feed_dict={X: batch_x})
        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * pixel_size:(i + 1) * pixel_size, j * pixel_size:(j + 1) * pixel_size] = \
                batch_x[k1].reshape([pixel_size, pixel_size])
            k1 = k1 + 1
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * pixel_size:(i + 1) * pixel_size, j * pixel_size:(j + 1) * pixel_size] = \
                g[k2].reshape([pixel_size, pixel_size])
            k2 = k2 + 1

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

    save_path = saver.save(sess,"./AutoEncoderSparseSigFinal.ckpt")

sess.close()

