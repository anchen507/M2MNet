import time
import tensorflow as tf
import numpy as np
from utils import *
import math
import tensorflow.contrib.slim as slim



def spatio_temporal_encoder(input, is_training=True):
    output = tf.layers.conv3d(input, 64, [3,3,3], padding='same', activation=tf.nn.relu, use_bias=True, kernel_initializer= tf.truncated_normal_initializer(stddev=0.01),name ="conv1")
    for layers in range(2, 4 + 1):
        with tf.variable_scope('block%d' % layers):
             output = tf.layers.conv3d(output, 64, [3,3,3],padding='same',activation=None,use_bias=False, kernel_initializer= tf.truncated_normal_initializer(stddev=0.01), name ="conv%d" % layers)
             output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    output = tf.layers.max_pooling3d(output,pool_size=(2,1,1),strides=(2,1,1),padding='SAME')
    for layers in range(5, 10 + 1):
        with tf.variable_scope('block%d' % layers):
             output = tf.layers.conv3d(output, 64, [2,3,3],padding='same',activation=None,use_bias=False, kernel_initializer= tf.truncated_normal_initializer(stddev=0.01), name ="conv%d" % layers)
             output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    output = tf.layers.max_pooling3d(output,pool_size=(2,1,1),strides=(2,1,1),padding='SAME')
    for layers in range(11, 14 + 1):
        with tf.variable_scope('block%d' % layers):
             output = tf.layers.conv3d(output, 64, [1,3,3],padding='same',activation=None,use_bias=False, kernel_initializer= tf.truncated_normal_initializer(stddev=0.01), name ="conv%d" % layers)
             output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    return output

def spatio_encoder(input, is_training=True):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 14 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    return output

def integrated_decoder(input, is_training=True, output_channels=3):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 128, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 14 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 128, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block15'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return output


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
