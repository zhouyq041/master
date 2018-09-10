# coding=utf-8
import tensorflow as tf
import mydataset as md
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

STEPS = 10000
batch_size = 64

mnist = input_data.read_data_sets('data', one_hot=True)
fc_size = 256


parameters = {
    'w1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, stddev=1e-1), name='w1'),
    'w2': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='w2'),
    'w3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='w3'),
    'w4': tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='w4'),
    'w5': tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='w5'),
    'fc1': tf.Variable(tf.truncated_normal([6272, fc_size], dtype=tf.float32, stddev=1e-2), name='fc1'),
    'fc2': tf.Variable(tf.truncated_normal([fc_size,fc_size], dtype=tf.float32, stddev=1e-2), name='fc2'),
    'softmax': tf.Variable(tf.truncated_normal([fc_size, 10], dtype=tf.float32, stddev=1e-2), name='fc3'),
    'bw1': tf.Variable(tf.random_normal([64])),
    'bw2': tf.Variable(tf.random_normal([64])),
    'bw3': tf.Variable(tf.random_normal([128])),
    'bw4': tf.Variable(tf.random_normal([128])),
    'bw5': tf.Variable(tf.random_normal([256])),
    'bc1': tf.Variable(tf.random_normal([fc_size])),
    'bc2': tf.Variable(tf.random_normal([fc_size])),
    'bs': tf.Variable(tf.random_normal([10]))
}

def conv2d(_x, _w, _b):
    '''
         封装的生成卷积层的函数
         因为NNIST的图片较小,这里采用1,1的步长
    :param _x:  输入
    :param _w:  卷积核
    :param _b:  bias
    :return:    卷积操作
    '''
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b))

def lrn(_x):
    '''
    作局部响应归一化处理
    :param _x:
    :return:
    '''
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def max_pool(_x, f):
    '''
        最大池化处理,因为输入图片尺寸较小,这里取步长固定为1,1,1,1
    :param _x:
    :param f:
    :return:
    '''
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 2, 2, 1], padding='SAME')

def inference(_parameters,_dropout):
    '''
     定义网络结构和训练过程
    :param _parameters:  网络结构参数 
    :param _dropout:     dropout层的keep_prob
    :return: 
    '''

    # 搭建Alex模型
    x = tf.placeholder(tf.float32, [None, 784]) # 输入: MNIST数据图像为展开的向量
    x_ = tf.reshape(x, shape=[-1, 28, 28, 1])   # 将训练数据reshape成单通道图片
    y_ = tf.placeholder(tf.float32, [None, 10]) # 标签值:one-hot标签值

    # 第一卷积层
    conv1 = conv2d(x_, _parameters['w1'], _parameters['bw1'])
    lrn1 = lrn(conv1)
    pool1 = max_pool(lrn1, 2)

    # 第二卷积层
    conv2 = conv2d(pool1, _parameters['w2'], _parameters['bw2'])
    lrn2 = lrn(conv2)
    pool2 = max_pool(lrn2, 2)

    # 第三卷积层
    conv3 = conv2d(pool2, _parameters['w3'], _parameters['bw3'])

    # 第四卷积层
    '''conv4 = conv2d(conv3, _parameters['w4'], _parameters['bw4'])

    # 第五卷积层
    conv5 = conv2d(conv4, _parameters['w5'], _parameters['bw5'])
    pool5 = max_pool(conv5, 2)   '''   

    # FC1层
    pool5 = conv3
    shape = pool5.get_shape() # 获取第五卷基层输出结构,并展开
    reshape = tf.reshape(pool5, [-1, shape[1].value*shape[2].value*shape[3].value])
    fc1 = tf.nn.relu(tf.matmul(reshape, _parameters['fc1']) + _parameters['bc1'])
    fc1_drop = tf.nn.dropout(fc1, keep_prob=_dropout)

    # FC2层
    fc2 = tf.nn.relu(tf.matmul(fc1_drop, _parameters['fc2']) + _parameters['bc2'])
    fc2_drop = tf.nn.dropout(fc2, keep_prob=_dropout)

    # softmax层
    y_conv = tf.nn.softmax(tf.matmul(fc2_drop, _parameters['softmax']) + _parameters['bs'])

    # 定义损失函数和优化器
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))#tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 计算准确率
    correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    print('start train!!')
    with tf.Session() as sess:
        initop = tf.global_variables_initializer()
        sess.run(initop)

        for step in range(STEPS):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            #print batch_ys
            #break
            if step % 10 == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
                loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
                print('step:%5d. --acc:%.6f. -- loss:%.6f.'%(step, acc, loss))

        print('train over!')

        # Test
        test_xs, test_ys = mnist.test.images[:512], mnist.test.labels[:512]

        print('test acc:%f' % (sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys})))

if __name__ == '__main__':
    inference(parameters, 0.9)