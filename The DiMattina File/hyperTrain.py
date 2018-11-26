# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:44:07 2018

@author: cdimattina

Description: This program cycles through a wide variety of hyper-parameter 
             values for training a simple three-layer auto-encoder with RELU 
             hidden units with pre-whitened image patches (10 Field pictures) 
             
             The goal is to study the properties of the linear receptive 
             fields/basis functions which maximize the L1 sparsity of the 
             hidden layer activity, as first applied in Glorot et al. (2011).

"""

###############################################################################
# Import modules 
###############################################################################
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from scipy.io import loadmat 
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
# Set hyper-parameter values (variables to loop through)
###############################################################################
wscale_vec          = [0.01]                
learning_rate_vec   = [0.01]   
bat_size_vec        = [1000]    # [1000]             
nhidden_vec         = [50]      # [25, 50, 100, 200]
sparse_penalty_vec  = [0.1]     # [0, 0.01, 0.1, 1, 10]
w2norm_penalty_vec  = [0.1]     # [0, 0.01, 0.1, 1, 10]
w1norm_penalty_vec  = [0]       # [0]                   # no penalty on W1 norm by default

###############################################################################
# Set default hyper-parameter values (fixed)
###############################################################################
frac_train          = 0.8                       # fraction to use for training                       
display_step        = 500; 
num_steps           = 10000; 
seed_value          = 42;

# Seeding the random number generator ensures we choose the same training set
# each time we run this.  
np.random.seed(seed_value)

###############################################################################
# Get training data 
#
# Currently only supports natural image patches. Will eventually support 
# sparse pixels, cosines and gabors
#
# Currently uses global variables. Re-write in functional form
#
###############################################################################
def getTrainingData(data_type,patch_size,num_patches):
    
    global tdata # training data
    global vdata # test data
    global num_train
    
    num_train    = round(frac_train*num_patches)
    num_test     = num_patches - num_train
    
    if data_type == 'nat':
        
        train_file_name = "train_data_nat" 
        
        Icube    = loadmat('IMAGES.mat')['IMAGES']
        coldim   = np.power(patch_size,2)
        tdata    = np.zeros([num_train, coldim])
        vdata    = np.zeros([num_test,coldim])
        n_images = Icube.shape[2]   # number of images in stack
        
        imsz     = min([Icube.shape[0],Icube.shape[1]])
        maxst    = imsz - patch_size - 1
        
        for i in range(num_train):
            this_pict   = np.random.randint(0,n_images)
            this_rst    = np.random.randint(0,maxst)
            this_cst    = np.random.randint(0,maxst)

            this_patch  = Icube[ (this_rst):(this_rst + patch_size) , \
                                 (this_cst):(this_cst + patch_size) , \
                                  this_pict]
            
            tdata[i,:]  = this_patch.reshape(1,coldim)

        for i in range(num_test):
            this_pict   = np.random.randint(0,n_images)
            this_rst    = np.random.randint(0,maxst)
            this_cst    = np.random.randint(0,maxst)

            this_patch  = Icube[ (this_rst):(this_rst + patch_size) , \
                                 (this_cst):(this_cst + patch_size) , \
                                  this_pict]
            
            vdata[i,:]  = this_patch.reshape(1,coldim)
    
        del Icube
    
    else:
        print('Error! Data type not supported.')
        
    
   
                                      
    print("saving training set : " + train_file_name)
    fileOut = open("./" + train_file_name + ".pkl","wb")
    pickle.dump(tdata,fileOut)
    pickle.dump(vdata,fileOut)
    fileOut.close()

###############################################################################
# Define network
###############################################################################
def defineNetwork(wscale,num_input,num_hidden):
    global weights, biases
    
    weights = {
    'encoder_h1' : tf.Variable(wscale*(-1*tf.ones([num_input, num_hidden]) + 2*tf.random_uniform([num_input, num_hidden]))),
    'decoder_h1' : tf.Variable(wscale*(-1*tf.ones([num_hidden, num_input]) + 2*tf.random_uniform([num_hidden, num_input])))
    }

    biases = {
    'encoder_h1' : tf.Variable(tf.zeros([1, num_hidden])),
    'decoder_h1' : tf.Variable(tf.zeros([1, num_input]))
    } 
## Building the encoder This layer has two layers that are concatinated into a 
## a single layer. This layer should reduce the input vector to a matrix of 
## diffrent deminsions.
def encoder(x):
    # Encoder Hidden layer with rectified linear activation
    layer_1 = tf.nn.relu(tf.matmul(x, weights['encoder_h1']) + biases['encoder_h1'] )
    return layer_1

## Building the decoder This is simular to the process that is happening in the
## encoder code but this layer further reduces the demensionality.
def decoder(x):
    # Decoder Hidden layer 
   layer_1 = tf.matmul(x, weights['decoder_h1'] + biases['decoder_h1'] ) 
   return layer_1    

###############################################################################
# Define TensorFlow graph
###############################################################################
def defineGraph(num_input):
    global X, encoder_op, decoder_op, y_pred, y_true 
    global errorterm, w1penalty, w2penalty, sparsepenalty

    # this placeholder will hold the training batch
    X               = tf.placeholder("float", [None, num_input])
 
    ## Construct model by forming the encoded input and the decoded encoder
    encoder_op      = encoder(X)
    decoder_op      = decoder(encoder_op)
    
    ## Prediction 
    y_pred          = decoder_op
    y_true          = X

    ## Define loss and optimizer, minimize the squared error
    errorterm       = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    w1penalty       = tf.reduce_mean(tf.pow(weights['encoder_h1'],2))
    w2penalty       = tf.reduce_mean(tf.pow(weights['decoder_h1'],2))
    sparsepenalty   = tf.reduce_mean(tf.abs(encoder_op))
    

###############################################################################
# Main functional version of program
#
# Inputs: data_type   : 'nat' - natural images
#                       'pix' - sparse pixels
#                       'cos' - sparse cosines
#                       'gab' - sparse gabors
#       
#         patch_size  : square dimension of patches to sample
#         num_patches : size of training set
#         save_nets   : 1 - save networks
#                       0 - do not save networks
#         verbose     : 1 - output step results
#                       0 - suppress step
###############################################################################
  
def trainNetwork(data_type, patch_size, num_patches,save_nets, verbose):
        
    # Input/output dimensionality
    num_input       = np.power(patch_size,2)   
    
    # Model number
    model_num       = 1
    
    # Get training data
    getTrainingData(data_type,patch_size,num_patches)
    
    # Open PDF for output
    with PdfPages('RESULTS/ModelPlotsRELU10.pdf') as pdf:
    
    # Loop through hyper-parameter values passed into function
        for i_hid in range(len(nhidden_vec)):
            for i_spr in range(len(sparse_penalty_vec)):
                for i_w2n in range(len(w2norm_penalty_vec)):
                    for i_w1n in range(len(w1norm_penalty_vec)):
                        for i_wsc in range(len(wscale_vec)):
                            for i_lrt in range(len(learning_rate_vec)):
                                for i_bat in range(len(bat_size_vec)):
                                    # Create filename for output file
                                    outfile_name = "hyperTrain"               + \
                                    "_hid_" + str(nhidden_vec[i_hid])         + \
                                    "_spr_" + str(sparse_penalty_vec[i_spr])  + \
                                    "_w2n_" + str(w2norm_penalty_vec[i_w2n])  + \
                                    "_w1n_" + str(w1norm_penalty_vec[i_w1n])  + \
                                    "_wsc_" + str(wscale_vec[i_wsc])          + \
                                    "_lrt_" + str(learning_rate_vec[i_lrt])   + \
                                    "_bat_" + str(bat_size_vec[i_bat])
                                
                                    # Define network + graph
                                    defineNetwork(wscale_vec[i_wsc],num_input,nhidden_vec[i_hid])
                                    defineGraph(num_input)
                                    
                                    # Define loss function + optimizer
                                    loss = errorterm + sparse_penalty_vec[i_spr]*sparsepenalty  \
                                                     + w2norm_penalty_vec[i_w2n]*w2penalty      \
                                                     + w1norm_penalty_vec[i_w1n]*w1penalty
                                    optimizer   = tf.train.AdamOptimizer(learning_rate_vec[i_lrt]).minimize(loss)
    
                                    # Define initializer and saver
                                    init        = tf.global_variables_initializer()
    
                                    # Start a new TF session
                                    with tf.Session() as sess:
                                        
                                        # Run the initializer
                                        sess.run(init)
        
                                        # Training
                                        for i in range(1, num_steps+1):
                                            # Prepare Data
                                            temp    = np.random.permutation(num_train)
                                            bat_ind = temp[0:bat_size_vec[i_bat]]
                                            batch_x = tdata[bat_ind,:]
                                            # Run optimization op (backprop) and cost op (to get loss value)
                                            _, l , e, s = sess.run([optimizer, loss, errorterm, sparsepenalty], feed_dict={X: batch_x})
                                            # Display logs per step
                                            if verbose and (i % display_step == 0 or i == 1):
                                                print('Step %i: Minibatch Loss: %f Error: %f Sparse: %f' % (i, l, e, s))
                                        
                                        # Set the final weights     
                                        final_W1  = weights['encoder_h1'].eval()
                                        final_W2  = weights['decoder_h1'].eval()
                                        final_b1  = biases['encoder_h1'].eval()
                                        final_b2  = biases['decoder_h1'].eval()
                                       
                                        print('Trained model : ' + str(model_num))
                                        
                                        if save_nets:
                                            print("saving : " + outfile_name)
                                            fileOut = open("./TRAINEDMODELS/" + outfile_name + ".pkl","wb")
                                            pickle.dump(final_W1,fileOut)
                                            pickle.dump(final_W2,fileOut)
                                            pickle.dump(final_b1,fileOut)
                                            pickle.dump(final_b2,fileOut)
                                            fileOut.close()
                                    
                                    # Close session
                                    sess.close()
                                    
                                    
                                    Z = final_W1
                                    nHid            = Z.shape[1]            # get the number of hidden units
                                    imSz            = int(np.sqrt(Z.shape[0]))   # size of each image patch
                                    plotGap2        = 1                     # half the gap to leave between each image patch
                                    imSzPlot        = int(imSz + 2*plotGap2)    

                                    layoutSqDim     = int(np.ceil(np.sqrt(nHid)))  # find dimensions for layout
                                    W1_plots      = np.zeros((layoutSqDim*imSzPlot,layoutSqDim*imSzPlot))
                                    
                                    k = 0
                                    for i in range(layoutSqDim):
                                        for j in range(layoutSqDim):
                                            if k < nHid:    
                                                littleImPlot = np.zeros((imSzPlot,imSzPlot))
                                                littleImPlot[(plotGap2):(imSzPlot-plotGap2),(plotGap2):(imSzPlot-plotGap2)] = np.reshape(Z[:,k],(imSz,imSz)) 
                                                W1_plots[ (i*imSzPlot ):((i+1)*imSzPlot) , (j*imSzPlot ):((j+1)*imSzPlot) ] = littleImPlot;
                                                k = k + 1
                                                
                                                
                                                
                                    Z = np.transpose(final_W2)
                                    nHid            = Z.shape[1]            # get the number of hidden units
                                    imSz            = int(np.sqrt(Z.shape[0]))   # size of each image patch
                                    plotGap2        = 1                     # half the gap to leave between each image patch
                                    imSzPlot        = int(imSz + 2*plotGap2)    

                                    layoutSqDim     = int(np.ceil(np.sqrt(nHid)))  # find dimensions for layout
                                    W2_plots      = np.zeros((layoutSqDim*imSzPlot,layoutSqDim*imSzPlot))
                                    
                                    k = 0
                                    for i in range(layoutSqDim):
                                        for j in range(layoutSqDim):
                                            if k < nHid:    
                                                littleImPlot = np.zeros((imSzPlot,imSzPlot))
                                                littleImPlot[(plotGap2):(imSzPlot-plotGap2),(plotGap2):(imSzPlot-plotGap2)] = np.reshape(Z[:,k],(imSz,imSz)) 
                                                W2_plots[ (i*imSzPlot ):((i+1)*imSzPlot) , (j*imSzPlot ):((j+1)*imSzPlot) ] = littleImPlot;
                                                k = k + 1

                                                
                                                
        
                                    # Make PDF plots - still in scope of inner loop. Modify Chance's code here. 
                                    
                                    fig = plt.figure(figsize=(35, 7))
                                    txt = "Trained model : %d \n \
                                            num_hidden : %d \n \
                                            wscale : %f \n \
                                            learn_rate : %f \n \
                                            batch_size : %d \n \
                                            sparse_penalty : %f \n \
                                            w2norm_penalty: %f \n \
                                            loss : %f \n \
                                            msq : %f \n \
                                            sparsity : %f \n" %(model_num, nhidden_vec[i_hid], wscale_vec[i_wsc],learning_rate_vec[i_lrt], \
                                                bat_size_vec[i_bat], sparse_penalty_vec[i_spr], w2norm_penalty_vec[i_w2n], l, e, s) 
                                                
                                    fig.suptitle(txt,x = 0.1, y= .65, fontsize=14, fontweight='bold', ha = 'left')
                                    fig.add_subplot(1,3,1).axis('off')
                                    fig.add_subplot(1,3,2)
                                    plt.title('Final_W1 Plots')
                                    plt.imshow(W1_plots, cmap="gray")
                                    fig.add_subplot(1,3,3)
                                    plt.title('Final_W2 Plots')
                                    plt.imshow(W2_plots, cmap="gray")
                                    pdf.savefig(fig)
                                    
                                    
                                    
                                    model_num += 1
        
        
        plt.close()


