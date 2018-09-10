#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: tut1_save.py
#Author: Wang 
#Mail: wang19920419@hotmail.com
#Created Time:2017-08-30 11:04:25
############################

import tensorflow as tf

# prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.Variable(tf.random_normal(shape = [2]), name = 'w1')  # name is very important in restoration
w2 = tf.Variable(tf.random_normal(shape = [2]), name = 'w2')
b1 = tf.Variable(2.0, name = 'bias1')
feed_dict = {w1:[10,3], w2:[5,5]}

# define a test operation that will be restored
w3 = tf.add(w1, w2)  # without name, w3 will not be stored
w4 = tf.multiply(w3, b1, name = "op_to_restore")

#saver = tf.train.Saver()
saver = tf.train.Saver()#max_to_keep = 4, keep_checkpoint_every_n_hours = 1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print sess.run(w1)
#saver.save(sess, 'my_test_model', global_step = 100)
saver.save(sess, 'my_test_model/a.ckpt')
#saver.save(sess, 'my_test_model', global_step = 100, write_meta_graph = False)