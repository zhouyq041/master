#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: tut2_import.py
#Author: Wang 
#Mail: wang19920419@hotmail.com
#Created Time:2017-08-30 14:16:38
############################

import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('my_test_model/a.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('my_test_model/'))
print sess.run('w1:0')