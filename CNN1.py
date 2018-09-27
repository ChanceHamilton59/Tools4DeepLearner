# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:04:55 2018

@author: cjhamilton4176
"""
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    
  # Load training and eval data
  train_data = tf.contrib.learn.datasets.load_dataset("t10k-images-idx3-ubyte")
  
  # Returns np.array
  #train_labels = np.asarray(tf.contrib.learn.datasets.load_dataset("t10k-labels-idx1-ubyte.gz"))
  #eval_data =  tf.contrib.learn.datasets.load_dataset("train-images-idx3-ubyte(1).gz")# Returns np.array
  #eval_labels = np.asarray(tf.contrib.learn.datasets.load_dataset("train-labels-idx1-ubyte.gz"))

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
   model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
  
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

# Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
  mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
       hooks=[logging_hook])

    # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
