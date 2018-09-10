# coding=utf-8
import math  
import numpy as np 
import argparse
import tensorflow as tf 


nclasses = 12

def init_weights(shape,name):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01),name = name)
#init weights

def loss(logits,label_batches):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
    cost = tf.reduce_mean(cross_entropy)
    return cost

def get_accuracy(logits,labels):
    acc = tf.nn.in_top_k(logits,labels,1)
    acc = tf.cast(acc,tf.float32)
    acc = tf.reduce_mean(acc)
    return acc


def training(loss,lr):
    train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
    return train_op

def conv2d(x,w,b):
    x = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = "SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def pooling(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")

def norm(x,lsize = 4):
    return tf.nn.lrn(x,depth_radius = lsize,bias = 1,alpha = 0.001/9.0,beta = 0.75)


weights = {
"w1":init_weights([3,3,1,16],"w1"),
"w2":init_weights([3,3,16,128],"w2"),
"w3":init_weights([3,3,128,256],"w3"),
"w4":init_weights([512,512],"w4"),
"wo":init_weights([512,nclasses],"w0")
}

#init biases
biases = {
"b1":init_weights([16],"b1"),
"b2":init_weights([128],"b2"),
"b3":init_weights([256],"b3"),
"b4":init_weights([512],"b4"),
"bo":init_weights([nclasses],"b0")
}

def mmodel(images,dropout = 1.0):

    l1 = conv2d(images,weights["w1"],biases["b1"])
    l2 = pooling(l1)
    l2 = norm(l2)
    l3 = conv2d(l2,weights["w2"],biases["b2"])
    l4 = pooling(l3)
    l4 = norm(l4)
    l5 = conv2d(l4,weights["w3"],biases["b3"])
    #same as the batch size
    l6 = pooling(l5)
    l6 = tf.reshape(l6,[-1,weights["w4"].get_shape().as_list()[0]])
    l7 = tf.nn.relu(tf.matmul(l6,weights["w4"])+biases["b4"])
    l7 = tf.nn.dropout(l7, keep_prob = dropout)
    soft_max = tf.nn.bias_add(tf.matmul(l7,weights["wo"]),biases["bo"])
    return soft_max