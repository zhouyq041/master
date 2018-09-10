 #coding:utf8

########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow    


# update: 2017-7-30 delphifan
########################################################################################

import tensorflow as tf
import numpy as np
import getdata as gd
from scipy.misc import imread, imresize
from imagenet_classes import class_names
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mnist = input_data.read_data_sets('data', one_hot=True)

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = tf.reshape(imgs,shape=[-1,28,28,1])#imgs
        self.fc_size = 512;
        self.nclasses = 10;
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)  #计算softmax层输出
        self.myout = tf.argmax(tf.nn.softmax(self.fc3l),1)
        if weights is not None and sess is not None:  #载入pre-training的权重
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        # 去RGB均值操作(这里RGB均值为原数据集的均值)
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([0,0,0],#123.68, 116.779, 103.939], 
                dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
    
    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
             # 取出shape中第一个元素后的元素  例如x=[1,2,3] -->x[1:]=[2,3]  
             # np.prod是计算数组的元素乘积 x=[2,3] np.prod(x) = 2*3 = 6  
             # 这里代码可以使用 shape = self.pool5.get_shape()     
             #shape = shape[1].value * shape[2].value * shape[3].value 代替

            shape = int(np.prod(self.pool4.get_shape()[1:]))  
            fc1w = tf.Variable(tf.truncated_normal([shape, self.fc_size],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[self.fc_size], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool4, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([self.fc_size, self.fc_size],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[self.fc_size], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([self.fc_size,self.nclasses ],
                                                         dtype=tf.float32,
                                                         stddev=1e-2), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[self.nclasses], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))

init_para = 0.1
nn_classes = 10
weights ={  
    'wc1':tf.Variable(tf.random_normal([3,3,1,64], dtype=tf.float32, stddev=init_para)),
    'wc2':tf.Variable(tf.random_normal([3,3,64,64], dtype=tf.float32, stddev=init_para)),
    'wc3':tf.Variable(tf.random_normal([3,3,64,128], dtype=tf.float32, stddev=init_para)), 
    'wc4':tf.Variable(tf.random_normal([3,3,128,128], dtype=tf.float32, stddev=init_para)), 
      
    'wc5':tf.Variable(tf.random_normal([3,3,128,256], dtype=tf.float32, stddev=init_para)),  
    'wc6':tf.Variable(tf.random_normal([3,3,256,256], dtype=tf.float32, stddev=init_para)),  
    'wc7':tf.Variable(tf.random_normal([3,3,256,256], dtype=tf.float32, stddev=init_para)), 
    'wc8':tf.Variable(tf.random_normal([3,3,256,256], dtype=tf.float32, stddev=init_para)), 
      
    'wc9':tf.Variable(tf.random_normal([3,3,256,512], dtype=tf.float32, stddev=init_para)), 
    'wc10':tf.Variable(tf.random_normal([3,3,512,512], dtype=tf.float32, stddev=init_para)), 
    'wc11':tf.Variable(tf.random_normal([3,3,512,512], dtype=tf.float32, stddev=init_para)),  
    'wc12':tf.Variable(tf.random_normal([3,3,512,512], dtype=tf.float32, stddev=init_para)),
    'wc13':tf.Variable(tf.random_normal([3,3,512,512], dtype=tf.float32, stddev=init_para)),  
    'wc14':tf.Variable(tf.random_normal([3,3,512,512], dtype=tf.float32, stddev=init_para)), 
    'wc15':tf.Variable(tf.random_normal([3,3,512,512], dtype=tf.float32, stddev=init_para)), 
    'wc16':tf.Variable(tf.random_normal([3,3,512,256], dtype=tf.float32, stddev=init_para)),
      
    'wd1':tf.Variable(tf.random_normal([4096,4096])),  
    'wd2':tf.Variable(tf.random_normal([4096,4096])),  
    'out':tf.Variable(tf.random_normal([4096,nn_classes])),  
}  
  
biases ={  
    'bc1':tf.Variable(tf.zeros([64])),  
    'bc2':tf.Variable(tf.zeros([64])),  
    'bc3':tf.Variable(tf.zeros([128])),  
    'bc4':tf.Variable(tf.zeros([128])),  
    'bc5':tf.Variable(tf.zeros([256])),  
    'bc6':tf.Variable(tf.zeros([256])),  
    'bc7':tf.Variable(tf.zeros([256])),  
    'bc8':tf.Variable(tf.zeros([256])),  
    'bc9':tf.Variable(tf.zeros([512])),  
    'bc10':tf.Variable(tf.zeros([512])),  
    'bc11':tf.Variable(tf.zeros([512])),  
    'bc12':tf.Variable(tf.zeros([512])),  
    'bc13':tf.Variable(tf.zeros([512])),  
    'bc14':tf.Variable(tf.zeros([512])),  
    'bc15':tf.Variable(tf.zeros([512])),  
    'bc16':tf.Variable(tf.zeros([256])),  
      
      
    'bd1':tf.Variable(tf.zeros([4096])),  
    'bd2':tf.Variable(tf.zeros([4096])),  
    'out':tf.Variable(tf.zeros([nn_classes])),  
}  


if __name__ == '__main__':
    learning_rate = 0.001
    max_iters = 100000
    batch_size = 128

    images = tf.placeholder(tf.float32, [None, 784])#224, 224, 3])
    classes = tf.placeholder(tf.float32, [None, 10])

    keep_prob=tf.placeholder(tf.float32)
    dropout=0.8
    
    vgg = vgg16(imgs = images)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = classes,logits = vgg.probs,name = 'entropy_with_logits'))

    opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(vgg.probs,1),tf.argmax(classes,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    with tf.Session() as sess:
        sess.run(init)
        step=1

        while step*batch_size < max_iters:
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(opt,feed_dict = {images:batch_xs,classes:batch_ys})

            if step%10 == 0:
                acc = sess.run(accuracy,feed_dict = {images:batch_xs,classes:batch_ys})
                loss = sess.run(cost,feed_dict = {images:batch_xs,classes:batch_ys})
                print  "iter:"+str(step*batch_size)+"\tacc:"+"{:6f}".format(acc)+"\tloss:"+"{:6f}".format(loss)

            step+=1   

        test_xs, test_ys = mnist.test.images[:1024], mnist.test.labels[:1024]
        print('test acc:%f' % (sess.run(accuracy,feed_dict = {images:batch_xs,classes:batch_ys})))
'''

    print "start run\n"
    #计算VGG16的softmax层输出(返回是列表，每个元素代表一个判别类型的数组)
    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1, img2, img3]})  

    for pro in prob:
        # 源代码使用(np.argsort(prob)[::-1])[0:5]     
        # np.argsort(x)返回的数组值从小到大的索引值  
        #argsort(-x)从大到小排序返回索引值   [::-1]是使用切片将数组从大到小排序  
        #preds = (np.argsort(prob)[::-1])[0:5]  
        preds = (np.argsort(-pro))[0:5]  #取出top5的索引
        for p in preds:
            print class_names[p], pro[p]
        print '\n'

    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)  # 载入预训练好的模型权重

    img1 = imread('img1.jpg', mode='RGB')    #载入需要判别的图片
    img1 = imresize(img1, (224, 224))

    img2 = imread('img2.jpg', mode='RGB')
    img2 = imresize(img2, (224, 224))

    img3 = imread('img3.jpg', mode='RGB')
    img3 = imresize(img3, (224, 224))

    print "start run\n"
    #计算VGG16的softmax层输出(返回是列表，每个元素代表一个判别类型的数组)
    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1, img2, img3]})  

    for pro in prob:
        # 源代码使用(np.argsort(prob)[::-1])[0:5]     
        # np.argsort(x)返回的数组值从小到大的索引值  
        #argsort(-x)从大到小排序返回索引值   [::-1]是使用切片将数组从大到小排序  
        #preds = (np.argsort(prob)[::-1])[0:5]  
        preds = (np.argsort(-pro))[0:5]  #取出top5的索引
        for p in preds:
            print class_names[p], pro[p]
        print '\n'
        '''