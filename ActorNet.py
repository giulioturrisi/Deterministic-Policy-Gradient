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

class Actor_Net():
  
  def __init__(self,size_state,size_act,bound_act):    
         

        self.input = tf.placeholder(shape=[None, size_state], dtype=tf.float32, name="X")

    	#rete principale actor-----------#
        self.W_fc1 = tf.Variable(tf.truncated_normal([size_state, 400],stddev=0.01))
        self.b_fc1 = tf.Variable(tf.zeros(400))

        self.W_fc2 = tf.Variable(tf.truncated_normal([400, 300],stddev=0.01))
        self.b_fc2 = tf.Variable(tf.zeros(300))

        self.W_fc3 = tf.Variable(tf.truncated_normal([300, 1],stddev=0.003))
        self.b_fc3 = tf.Variable(tf.zeros(1))

        self.network_params = tf.trainable_variables()

        self.out_fc1 = tf.nn.relu(tf.matmul(self.input, self.W_fc1) + self.b_fc1)
        self.out_fc2 = tf.nn.relu(tf.matmul(self.out_fc1, self.W_fc2) + self.b_fc2)
        self.out_fc3 = tf.nn.tanh(tf.matmul(self.out_fc2, self.W_fc3) + self.b_fc3)
 
        self.scaled_out = tf.multiply(self.out_fc3, bound_act)

        #target--------------------------#
        self.W_fc1_target = tf.Variable(tf.truncated_normal([size_state, 400],stddev=0.01))
        self.b_fc1_target = tf.Variable(tf.zeros(400))

        self.W_fc2_target = tf.Variable(tf.truncated_normal([400, 300],stddev=0.01))
        self.b_fc2_target = tf.Variable(tf.zeros(300))

        self.W_fc3_target = tf.Variable(tf.truncated_normal([300, 1],stddev=0.003))
        self.b_fc3_target = tf.Variable(tf.zeros(1))

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        self.out_fc1_target = tf.nn.relu(tf.matmul(self.input, self.W_fc1_target) + self.b_fc1_target)
        self.out_fc2_target = tf.nn.relu(tf.matmul(self.out_fc1_target, self.W_fc2_target) + self.b_fc2_target)
        self.out_fc3_target = tf.nn.tanh(tf.matmul(self.out_fc2_target, self.W_fc3_target) + self.b_fc3_target)
 
        self.scaled_out_target = tf.multiply(self.out_fc3_target, bound_act)


        #------------------------------#
        #funzione per updare i pesi della rete target e la rete principale - presa online
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], 0.001) +
                                                  tf.multiply(self.target_network_params[i], 1. - 0.001))
                for i in range(len(self.target_network_params))]
        #-----------------------------#


        #ricevo il gradiente dal critic
        self.action_gradient = tf.placeholder(tf.float32, [None, size_act])
        #Combino i gradienti
        self.actor_gradients = tf.gradients(self.scaled_out,self.network_params,-self.action_gradient)

        #per la backprop
        self.optimize = tf.train.AdamOptimizer(0.0001).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

  def get_num_trainable_vars(self):
      return self.num_trainable_vars