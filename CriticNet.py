from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import tensorflow as tf


class Critic_Net():

  def __init__(self,size_state,size_act,num_actor_vars):    
          

        self.input = tf.placeholder(shape=[None, size_state], dtype=tf.float32, name="X")
        self.target = tf.placeholder(shape=[None,1], dtype=tf.float32, name="y")
        self.actions = tf.placeholder(shape=[None,size_act], dtype=tf.float32, name="actions")


    	#rete principale critc---------------#
        self.W_fc1 = tf.Variable(tf.truncated_normal([size_state, 400],stddev=0.01))
        self.b_fc1 = tf.Variable(tf.zeros(400))

        self.W_fc2 = tf.Variable(tf.truncated_normal([size_act, 300],stddev=0.01))
        self.b_fc2 = tf.Variable(tf.zeros(300))

        self.W_fc3 = tf.Variable(tf.truncated_normal([400, 300],stddev=0.01))
        self.b_fc3 = tf.Variable(tf.zeros(300))

        self.W_fc4 = tf.Variable(tf.truncated_normal([300, 1],stddev=0.003))
        self.b_fc4 = tf.Variable(tf.zeros(1))


        self.out_fc1 = tf.nn.relu(tf.matmul(self.input, self.W_fc1) + self.b_fc1)
        self.out_fc2 = tf.nn.relu(tf.matmul(self.out_fc1,self.W_fc3) + tf.matmul(self.actions,self.W_fc2) + self.b_fc2)
        self.out_fc3 = tf.matmul(self.out_fc2, self.W_fc4) + self.b_fc4

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        #rete target-----------------------#
        self.W_fc1_target = tf.Variable(tf.truncated_normal([size_state, 400],stddev=0.01))
        self.b_fc1_target = tf.Variable(tf.zeros(400))

        self.W_fc2_target = tf.Variable(tf.truncated_normal([size_act, 300],stddev=0.01))
        self.b_fc2_target = tf.Variable(tf.zeros(300))

        self.W_fc3_target = tf.Variable(tf.truncated_normal([400, 300],stddev=0.01))
        self.b_fc3_target = tf.Variable(tf.zeros(300))

        self.W_fc4_target = tf.Variable(tf.truncated_normal([300, 1],stddev=0.003))
        self.b_fc4_target = tf.Variable(tf.zeros(1))


        self.out_fc1_target = tf.nn.relu(tf.matmul(self.input, self.W_fc1_target) + self.b_fc1_target)
        self.out_fc2_target = tf.nn.relu(tf.matmul(self.out_fc1_target,self.W_fc3_target) + tf.matmul(self.actions,self.W_fc2_target) + self.b_fc2_target)
        self.out_fc3_target = tf.matmul(self.out_fc2_target, self.W_fc4_target) + self.b_fc4_target
       

       	self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]
        #----------------


        #funzione per updare i pesi della rete target e la rete principale - presa online
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], 0.001) +
                                                  tf.multiply(self.target_network_params[i], 1. - 0.001))
                for i in range(len(self.target_network_params))]

        #per la backprop
        self.loss = tf.reduce_mean(tf.square(self.target - self.out_fc3))
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.action_grads = tf.gradients(self.out_fc3, self.actions)