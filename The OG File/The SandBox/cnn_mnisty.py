# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:20:35 2018

@author: cjhamilton4176
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

#if __name__ == "__main__":
 # tf.app.run()


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  #############################################################################
  # Input Layer: layers are expected to have the following shape
  # Shape: [batch_size, image_width, image_height, channels]
  # Breakdown:
  #     batch_size      : Size of the subset of examples to use when preforming 
  #                       gradient descent during training.  
  #     image_width     : Width of the emaple images.
  #     image_height    : Heiught of the example images.
  #     channels        : Number of color channels in the example images. 
  #                         Color => 3
  #                         momochrome => 1
  #############################################################################  
  
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

 #############################################################################
  # Convolutional Layer #1 : we want to apply 32 5x5 filters to the input layer
  #                          with a ReLU activation function
  # Breakdown:
  #     inputs          : Input layer see above section
  #     filters         : specifies the number of filters to apply.
  #     kernal_size     : specifies the dimensions of the filters as [w,h] 
  #                       or int value for square filters
  #     padding         : specifies one of two enumerated values
  #                         1) valid (defalt)
  #                         2) same : instructs TensorFlow to add 0 values to 
  #                                   the edges of the input tensor to preserve
  #                                   width and height of 28.
  #     activation      : specifies the activation function to apply to the 
  #                       output of the convolution.
  #############################################################################
  
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  #############################################################################
  # Pooling Layer #1    : Performs max pooling with a 2x2 filter and stride of 2
  #                       (which specifies that pooled regions do not overlap)
  # Breakdown:
  #     inputs          : Input layer see above section
  #     filters         : specifies the number of filters to apply.
  #     kernal_size     : specifies the dimensions of the filters as [w,h] 
  #                       or int value for square filters
  #     padding         : specifies one of two enumerated values
  #                         1) valid (defalt)
  #                         2) same : instructs TensorFlow to add 0 values to 
  #                                   the edges of the input tensor to preserve
  #                                   width and height of 28.
  #     activation      : specifies the activation function to apply to the 
  #                       output of the convolution.
  #############################################################################
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  #############################################################################
  # Convolutional Layer #2 : Applies 64 5x5 filters, with ReLU activation
  #                          function.
  #  
  #############################################################################                       
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  
  #############################################################################
  # Pooling Layer #2    :  Again, performs max pooling with a 2x2 filter and 
  #                        stride of 2.
  #  
  #############################################################################
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  
  #############################################################################
  # Dense Layer #1    :  1,024 neurons, with dropout regularization rate of
  #                      0.4 (probability of 0.4 that any given element will
  #                      be dropped during training).
  #  
  #############################################################################
  
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  #############################################################################
  # Dense Layer #2 (Logits Layer):  10 neurons, one for each digit target 
  #                                 class (0–9).
  #  
  #############################################################################
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  #############################################################################
  # Calculate Loss: define a loss function that measures how closely the 
  #                 model's predictions match the target classes
  #  
  #############################################################################
  

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
