# -*- coding: utf-8 -*-
"""
Created on Tue May  8 18:27:21 2018

@author: Vivek
"""

from tensorflow.examples.tutorials.mnist import input_data
mnistdata = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])  #784 is dimentionality of each image
y1 = tf.placeholder(tf.float32, shape=[None, 10])  #there are 10 classes
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #training model
for _ in range(1000):
  batch = mnistdata.train.next_batch(100)   #loading a batch of training samples
  train_step.run(feed_dict={x: batch[0], y1: batch[1]})
  
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y1,1))  #evaluating model
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
print(accuracy.eval(feed_dict={x: mnistdata.test.images, y1: mnistdata.test.labels}))