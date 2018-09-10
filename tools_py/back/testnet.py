
import os  
import math  
import numpy as np 
import argparse
import tensorflow as tf  
from scipy.misc import imread, imresize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_files(filename):
    class_train = []
    label_train = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename+train_class):
            class_train.append(filename+train_class+'/'+pic)
            label_train.append(train_class)
    temp = np.array([class_train,label_train])
    temp = temp.transpose()
    #shuffle the samples
    np.random.shuffle(temp)
    #after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]


    #print(label_list)
    return  image_list,label_list

def get_batches(image,label,resize_w,resize_h,batch_size,capacity):
    #convert the list of images and labels to tensor
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int64)
    queue = tf.train.slice_input_producer([image,label])
    label = queue[1]

    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c,channels = 3)

   
    #resize
    image = tf.image.resize_image_with_crop_or_pad(image,resize_w,resize_h)
    #(x - mean) / adjusted_stddev
    #image = tf.image.per_image_standardization(image)
    print image
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size = batch_size,
                                             num_threads = 64,
                                             capacity = capacity)#,min_after_dequeue = capacity/2)
    images_batch = tf.cast(image_batch,tf.float32)
    labels_batch = tf.reshape(label_batch,[batch_size])

    return images_batch,labels_batch

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

nclasses = 3
weights = {
"w1":init_weights([3,3,3,16],"w1"),
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




trainname = "data"

def run_training():
    data_dir = trainname+"/"
    image,label = get_files(data_dir)
    image_batches,label_batches = get_batches(image,label,28,28,32,100)
    p = mmodel(image_batches)
    cost = loss(p,label_batches)
    train_op = training(cost,0.001)
    acc = get_accuracy(p,label_batches)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
   
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)

    saver = tf.train.Saver(max_to_keep = 4)
    tf.add_to_collection("predict", p)
    maxstep = 1000
    #-----------------------------------------
    x = tf.placeholder(tf.float32, [None, 784])
    x = tf.reshape(x,shape=[-1,28,28,3])
    y = mmodel(x)
    num = 0
    acc2 = get_accuracy(y,[num,num])
   
    #-----------------------------------------
    try:
        for step in np.arange(maxstep):

            if coord.should_stop():
                break
            _,train_acc,train_loss = sess.run([train_op,acc,cost])
            if step % 50 == 0:
                print(step),
                print("loss:{} accuracy:{}".format(train_loss,train_acc))
                #-----------------------------------------

            if step == maxstep-1:
                saver.save(sess,"mymodel/my_model.ckpt")
        dirname = './'+trainname+'/'+str(num)+'/'
        total_count = 0
        correct_count = 0

        for pic in os.listdir(dirname):
            one_image = imread(dirname+pic, mode='RGB')
            one_image = imresize(one_image, (28, 28))
            pred,accuracy = sess.run([y,acc2],feed_dict = {x:[one_image,one_image]})
            
            ##img1 =  tf.reshape(img1,shape=[28,28,1])
            #pred = sess.run(y,feed_dict = {x:[img2]})
            
            for i in pred:
                print dirname+pic,(np.argsort(-i))[0:6],accuracy
                total_count = total_count+1
                if (np.argsort(-i))[0] == num:
                    correct_count = correct_count+1
            if total_count>30:
                break
            
        print float(correct_count)/float(total_count)

    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    

    
    sess.close()


def load_model():
    x = tf.placeholder(tf.float32, [None, 784])
    x = tf.reshape(x,shape=[-1,28,28,3])

    y = mmodel(x)
    
    num = 0
    acc = get_accuracy(y,[num])
    
    with tf.Session() as sess:
        sess = tf.Session()

        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph('mymodel/my_model.ckpt.meta')
        saver.restore(sess,'mymodel/my_model.ckpt')#tf.train.latest_checkpoint('mymodel/'))
        graph = tf.get_default_graph()  


        img1 = imread('data/0/0.jpg', mode='RGB')    
        img1 = imresize(img1, (28, 28))

        img2 = imread('mnist_dataset/5/train_165.jpg', mode='RGB')
        img2 = imresize(img2, (28, 28))

        img3 = imread('mnist_dataset/7/train_554.jpg', mode='RGB')
        img3 = imresize(img3, (28, 28))
        # pred_y = tf.get_collection("predict")
        #y = graph.get_collection('predict')
       
        dirname = './'+trainname+'/'+str(num)+'/'
        total_count = 0
        correct_count = 0

        for pic in os.listdir(dirname):
            one_image = imread(dirname+pic, mode='RGB')
            one_image = imresize(one_image, (28, 28))
            pred,accuracy = sess.run([y,acc],feed_dict = {x:[one_image]})
            
            ##img1 =  tf.reshape(img1,shape=[28,28,1])
            #pred = sess.run(y,feed_dict = {x:[img2]})
            
            for i in pred:
                print (np.argsort(-i))[0:6],
                print accuracy
                total_count = total_count+1
                if (np.argsort(-i))[0] == num:
                    correct_count = correct_count+1
            
        print float(correct_count)/float(total_count)
        

#        re = sess.run(y,feed_dict={x:xx})

        #print result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    arg = parser.parse_args()
    if arg.mode == '0':
        run_training()
    else:
        load_model()