# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

STEPS = 500
batch_size = 32

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


parameters = {
    'w1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, stddev=1e-1), name='w1'),
    'w2': tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='w2'),
    'w3': tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='w3'),
    'w4': tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='w4'),
    'w5': tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='w5'),
    'fc1': tf.Variable(tf.truncated_normal([256*28*28, 1024], dtype=tf.float32, stddev=1e-2), name='fc1'),
    'fc2': tf.Variable(tf.truncated_normal([1024, 1024], dtype=tf.float32, stddev=1e-2), name='fc2'),
    'softmax': tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, stddev=1e-2), name='fc3'),
    'bw1': tf.Variable(tf.random_normal([64])),
    'bw2': tf.Variable(tf.random_normal([64])),
    'bw3': tf.Variable(tf.random_normal([128])),
    'bw4': tf.Variable(tf.random_normal([128])),
    'bw5': tf.Variable(tf.random_normal([256])),
    'bc1': tf.Variable(tf.random_normal([1024])),
    'bc2': tf.Variable(tf.random_normal([1024])),
    'bs': tf.Variable(tf.random_normal([10]))
}



def init_weights(shape,name):
    return tf.Variable(tf.random_normal(shape,stddev = 0.02),name = name)
#init weight



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

nclasses = 10
weights = {
"w1":init_weights([3,3,1,16],"w1"),
"w2":init_weights([3,3,16,128],"w2"),
"w3":init_weights([3,3,128,256],"w3"),
"w4":init_weights([4096,4096],"w4"),
"wo":init_weights([4096,nclasses],"w0")
}

#init biases
biases = {
"b1":init_weights([16],"b1"),
"b2":init_weights([128],"b2"),
"b3":init_weights([256],"b3"),
"b4":init_weights([4096],"b4"),
"bo":init_weights([nclasses],"b0")
}

def mmodel(images):

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
    soft_max = tf.add(tf.matmul(l7,weights["wo"]),biases["bo"])
    return soft_max


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
    '''conv1 = conv2d(x_, _parameters['w1'], _parameters['bw1'])
    lrn1 = lrn(conv1)
    pool1 = max_pool(lrn1, 2)

    # 第二卷积层
    conv2 = conv2d(pool1, _parameters['w2'], _parameters['bw2'])
    lrn2 = lrn(conv2)
    pool2 = max_pool(lrn2, 2)

    # 第三卷积层
    conv3 = conv2d(pool2, _parameters['w3'], _parameters['bw3'])

    # 第四卷积层
    conv4 = conv2d(conv3, _parameters['w4'], _parameters['bw4'])

    # 第五卷积层
    conv5 = conv2d(conv4, _parameters['w5'], _parameters['bw5'])
    pool5 = max_pool(conv5, 2)

    # FC1层
    shape = pool5.get_shape() # 获取第五卷基层输出结构,并展开
    reshape = tf.reshape(pool5, [-1, shape[1].value*shape[2].value*shape[3].value])
    fc1 = tf.nn.relu(tf.matmul(reshape, _parameters['fc1']) + _parameters['bc1'])
    fc1_drop = tf.nn.dropout(fc1, keep_prob=_dropout)

    # FC2层
    fc2 = tf.nn.relu(tf.matmul(fc1_drop, _parameters['fc2']) + _parameters['bc2'])
    fc2_drop = tf.nn.dropout(fc2, keep_prob=_dropout)

    # softmax层
    y_conv = tf.nn.softmax(tf.matmul(fc2_drop, _parameters['softmax']) + _parameters['bs'])'''
    y_conv = mmodel(x_)
    # 定义损失函数和优化器
   # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_),logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

    cost = loss(y_conv,tf.argmax(y_,1))
    train_op = training(cost,0.001)

    # 计算准确率
    correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        initop = tf.global_variables_initializer()
        sess.run(initop)

        for step in range(STEPS):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
        
            if step % 50 == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
               # loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
                print('step:%5d. --acc:%.6f. -- loss:%.6f.'%(step, acc, acc))

        print('train over!')

        # Test
        test_xs, test_ys = mnist.test.images[:512], mnist.test.labels[:512]

        print('test acc:%f' % (sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys})))

        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    inference(parameters, 0.9)