# coding=utf-8
import os  
import math  
import numpy as np 
import argparse
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import parse_image as pimg
import train_utils as tutils
import model_alex
import model_vgg16
import model_google
import model_resnet
import model_resnet2
import model_my
import model_npn
import cv2
import scipy.misc
import matplotlib
#from scipy.misc import imread, imresize
import cv2 as cv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
slim = tf.contrib.slim
nclasses = 12
hist_file = np.load('pre_setup/hist_20_norm.npz')['arr_0'].tolist()
tut = tutils.my_train_utils()


def get_next_patch(image,label,index,batch_size):
    current_images = image[index:index+batch_size]
    images = []

    for img in current_images:
        images.append(pim.parse_image(img))

    current_labels = label[index:index+batch_size]

    labels = np.zeros([len(current_labels),nclasses])
    i = 0
    for lab in current_labels:
        labels[i][int(lab)] = 1
        i = i+1

    return np.array(images),labels


def get_next_patch_in_mem(datas,index,batch_size):
    ret_data = []
    for data in datas:
        ret_data.append(data[index:index+batch_size])
    return ret_data

def get_batch_x2s(batch_xs):
    ret = []
    for image_path in batch_xs:
        image_path = image_path.replace('traindata','testdata2')
        ret.append(hist_file[image_path])
    return ret


#train dir--------------
#dataset_name1 = "mnist_dataset"
dataset_name2 = "cartoon_dataset"
dataset_name = dataset_name2
trainname = dataset_name+"/traindata/"#"data"
testname = dataset_name+"/testdata/"
testfilename = 'pre_setup/test_1_768'
#train presetup--------------
size_w = 151
size_h = size_w
drop_out = 0.8
channel = 3
pim = pimg.my_parse_image(size_w,size_h,channel)
#-----------------------------

def get_accuracy(logits,labels,k = 1):
    acc = tf.nn.in_top_k(logits,labels,k)
    acc = tf.cast(acc,tf.float32)
    acc = tf.reduce_mean(acc)
    return acc

def run_training():
    #train parameters
    maxstep = 3000
    learning_rate = 1e-3
    batch_size = 32
    netname = 'res101'
    #set xy-------------------------------
    print 'set model'
    x = tf.placeholder(tf.float32, [None, size_w*size_h])
    x = tf.reshape(x,shape=[-1,size_w,size_h,channel])
    y = tf.placeholder(tf.float32, [None, nclasses]) 
    istrain = tf.placeholder(tf.bool, [1]) 
    x2 = tf.placeholder(tf.float32, [None, 3,19])
    #x2 = tf.reshape(x2,shape=[-1,1,3,19])
    #y_result = model_alex.mmodel(x,drop_out)
    #y_result2 = model_vgg16.vgg16(x,drop_out).probs
    #y_result2 = tf.reshape(y_result2,shape=[-1,1,1,512])
    #with slim.arg_scope(model_my.inception_v3_arg_scope()): 
    #y_result2_result,_ = model_resnet2.resnet_v2_50(x, num_classes = 12,is_training=istrain[0])
    #print 'y_result2',y_result
    google_scope = model_my.inception_v3_arg_scope()
    #with slim.arg_scope(google_scope):
    #y_result,_ = model_my.inception_v3(x,inputs2 = None,inputs3=None,num_classes = 12,is_training = istrain[0],scope = 'google_rgb_fin')
    
    #y_result,_ = model_google.inception_v3(x,num_classes = 12,is_training = True)
    
    arg_scope = model_resnet2.resnet_arg_scope()
    #with slim.arg_scope(arg_scope):
    y_result,_ = model_resnet2.resnet_v2_101(x, num_classes = 12)#,is_training = istrain[0])
    #y_result = tf.reshape(y_result,shape=[-1,12])
    
    # 定义损失函数和优化器
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y,1),logits=y_result))
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    train_step2 = tf.train.AdamOptimizer(learning_rate*0.1).minimize(cross_entropy)
    # 计算准确率
    correct_pred = tf.equal(tf.argmax(y_result, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy = get_accuracy(y_result,tf.argmax(y,1),1)
    accuracy3 = get_accuracy(y_result,tf.argmax(y,1),3)
    #train begin
    saver = tf.train.Saver(max_to_keep = 4)
    #tf.add_to_collection("predict", p)
    #train prestep1


    print 'train prestep'
    test_image,test_label,train_image,train_label = tut.get_testfiles(testfilename)
    test_size = len(test_image)
    image_test,label_test = get_next_patch(test_image,test_label,0,test_size)

    image_test_x2s = get_batch_x2s(test_image)
    image_x2s = get_batch_x2s(train_image)

    image,label = train_image,train_label
    #image,label = get_next_patch(train_image,train_label,0,len(train_image))
    #image,label = image.tolist(),label.tolist()
    #[image,label,image_x2s] = tut.shuffle_files([image,label,image_x2s])
    [image,label] = tut.shuffle_files([image,label])
    #train start-----------------------------------------
    print('start train!!')
    index = 0
    epoch = 0
    all_loss = []
    all_acc = []
    all_acc_train = []
    change_learning_rate_flag = True
    #a,b,c,d = get_test_images()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver(tf.global_variables())
        #saver.restore(sess,'mymodel/my_model.ckpt')

        for step in np.arange(maxstep):
            if coord.should_stop():
                break
            #train batch-----------------
            batch_xs, batch_ys = get_next_patch(image,label,index,batch_size)
            batch_x2s = get_batch_x2s(image[index:index+batch_size])
            '''batch_xs = image[index:index+batch_size]
            batch_ys = label[index:index+batch_size]
            batch_x2s = image_x2s[index:index+batch_size]'''
            index = batch_size+index
            if index > len(image)-batch_size:
                index = 0
                epoch += 1
                print 'shuffle:',epoch
                [image,label] = tut.shuffle_files([image,label])
                #[image,label,image_x2s] = tut.shuffle_files([image,label,image_x2s])
            #train batch-----------------
            if epoch > 15:
                break



            if step % 40 == 0:
                print 'test:'
                ii = 0
                cc = 0
                cc3 = 0

                for i in range(test_size/batch_size):
                    loss,acc,acc3 = sess.run([cross_entropy,accuracy,accuracy3], feed_dict={
                        x: image_test[i*batch_size:i*batch_size+batch_size], 
                        y: label_test[i*batch_size:i*batch_size+batch_size],
                        x2:image_test_x2s[i*batch_size:i*batch_size+batch_size],istrain:[False]})
                    ii = ii+1
                    cc = cc+acc
                    cc3 = cc3+acc3

                    loss,acc = sess.run([cross_entropy,accuracy], feed_dict={x: batch_xs, y: batch_ys,x2:batch_x2s,istrain:[True]})
                    all_loss.append(loss)
                    all_acc.append(cc/ii)
                    all_acc_train.append(acc)
                    #print(i),
                    #print("loss:{} accuracy:{} accuracy-top3:{}".format(loss,acc,acc3))
                if ii != 0 and step % 20 == 0:
                    print 'Total accuracy: ',cc/ii,'Total accuracy-top3: ',cc3/ii          
                #test(sess,x,y_result)
                #-----------------------------------------            

            sess.run(train_step,feed_dict = {x:batch_xs,y:batch_ys,x2:batch_x2s,istrain:[True]}) 

            if step % 50 == 0:
                loss,acc = sess.run([cross_entropy,accuracy], feed_dict={x: batch_xs, y: batch_ys,x2:batch_x2s,istrain:[True]})
                
                if change_learning_rate_flag and loss < 3:
                    train_step = train_step2
                    change_learning_rate_flag = False
                    print 'Change learning_rate!'
                print(step),
                print("loss:{} accuracy:{}".format(loss,acc))

                '''loss,acc = sess.run([cross_entropy,accuracy], feed_dict={
                    x: a, 
                    y: d})
                print("loss:{} accuracy:{}".format(loss,acc))'''

           

        #saver.save(sess,"mymodel/my_model.ckpt")
        np.save('results/loss_normal/loss_'+netname+'_norm',all_loss)
        np.save('results/loss_normal/acc1_'+netname+'_norm',all_acc)
        np.save('results/loss_normal/acc_train_'+netname+'_norm',all_acc_train)
        
        coord.request_stop()
        coord.join(threads)
        sess.close()

def run_training2():

    batch_size = 64
    num_net = 3 #network count in npn
    maxstep = 2000
    learning_rate = 1e-1
    max_epoch = 10
    loss_file = 'loss_npn1_res101_64'
    acc_file = loss_file
    #set xy-------------------------------
   
    x = []
    x.append(tf.placeholder(tf.float32, [None, 1,1,2048]))
    x.append(tf.placeholder(tf.float32, [None, 1,1,2048]))
    x.append(tf.placeholder(tf.float32, [None, 1,1,2560]))
    x.append(tf.placeholder(tf.float32, [None, 1,1,2048]))
    for i in range(num_net,len(x)):
        x[i] = tf.placeholder(tf.float32, [None])
    y = tf.placeholder(tf.float32, [None, nclasses]) 

    #load pre_training

    g1 = tf.Graph()
    isess1 = tf.Session(graph = g1)
    g2 = tf.Graph()
    isess2 = tf.Session(graph = g2)
    g3 = tf.Graph()
    isess3 = tf.Session(graph = g3)
    g4 = tf.Graph()
    isess4 = tf.Session(graph = g4)

    with g1.as_default():
        _n1_x = tf.placeholder(tf.float32, [None, size_w*size_h])
        _n1_x = tf.reshape(_n1_x,shape=[-1,size_w,size_h,channel])
        _n1_x2 = tf.placeholder(tf.float32, [None,  3,19])
        #_n1_x2 = tf.reshape(_n1_x2,shape=[-1,1,1,57])
        istrain = tf.placeholder(tf.bool, [1])
        #_n1_y,_n1_end = model_my.inception_v3(_n1_x,inputs2 = None,num_classes = 12,is_training = False)
        _n1_y,_n1_end =  model_resnet2.resnet_v2_101(_n1_x, num_classes = 12)
        saver = tf.train.Saver()#tf.global_variables()
        saver.restore(isess1,'mymodel/saved_all/res_101/my_model.ckpt')

    with g2.as_default():
        _n2_x = tf.placeholder(tf.float32, [None, size_w*size_h])
        _n2_x = tf.reshape(_n2_x,shape=[-1,size_w,size_h,channel])
        _n2_x2 = tf.placeholder(tf.float32, [None,  3,19])
        #_n2_x2 = tf.reshape(_n2_x2,shape=[-1,1,1,57])
        istrain = tf.placeholder(tf.bool, [1])
        _n2_y,_n2_end = model_my.inception_v3(_n2_x,inputs2 = None,num_classes = 12,is_training = False)
        saver = tf.train.Saver()#tf.global_variables()
        saver.restore(isess2,'mymodel/saved_all/google_rgb/my_model.ckpt')

    with g3.as_default():
        _n3_x = tf.placeholder(tf.float32, [None, size_w*size_h])
        _n3_x = tf.reshape(_n3_x,shape=[-1,size_w,size_h,channel])
        _n3_x2 = tf.placeholder(tf.float32, [None,  3,19])
        #_n1_x2 = tf.reshape(_n1_x2,shape=[-1,1,1,57])
        istrain = tf.placeholder(tf.bool, [1])
        _n3_y,_n3_end = model_my.inception_v3(_n3_x,inputs2 = _n3_x2,num_classes = 12,is_training = False)
        saver = tf.train.Saver()#tf.global_variables()
        saver.restore(isess3,'mymodel/saved_all/google_rgb_fin/my_model.ckpt')

    with g4.as_default():
        _n4_x = tf.placeholder(tf.float32, [None, size_w*size_h])
        _n4_x = tf.reshape(_n4_x,shape=[-1,size_w,size_h,channel])
        _n4_x2 = tf.placeholder(tf.float32, [None,  3,19])
        #_n1_x2 = tf.reshape(_n1_x2,shape=[-1,1,1,57])
        istrain = tf.placeholder(tf.bool, [1])
        #_n1_y,_n1_end = model_my.inception_v3(_n1_x,inputs2 = None,num_classes = 12,is_training = False)
        _n4_y,_n4_end =  model_resnet2.resnet_v2_50(_n4_x, num_classes = 12)
        saver = tf.train.Saver()#tf.global_variables()
        saver.restore(isess4,'mymodel/saved_all/res_50/my_model.ckpt')

    print 'set model'

    y_result = model_npn.npn_net(x, num_classes = 12,num_net = num_net)#,is_training = istrain[0])
    y_result = tf.reshape(y_result,shape=[-1,12])    
    
    # 定义损失函数和优化器
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y,1),logits=y_result))
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    train_step2 = tf.train.AdamOptimizer(learning_rate*0.1).minimize(cross_entropy)
    # 计算准确率
    correct_pred = tf.equal(tf.argmax(y_result, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy = get_accuracy(y_result,tf.argmax(y,1),1)
    accuracy3 = get_accuracy(y_result,tf.argmax(y,1),3)
    #train begin
    saver = tf.train.Saver(max_to_keep = 4)

    print 'train prestep'
    test_image,test_label,train_image,train_label = tut.get_testfiles(testfilename)
    test_size = len(test_image)
    image_test,label_test = get_next_patch(test_image,test_label,0,test_size)

    image_test_x2s = get_batch_x2s(test_image)
    image_x2s = get_batch_x2s(train_image)

    image,label = train_image,train_label
    #image,label = get_next_patch(train_image,train_label,0,len(train_image))
    #image,label = image.tolist(),label.tolist()
    #[image,label,image_x2s] = tut.shuffle_files([image,label,image_x2s])
    [image,label] = tut.shuffle_files([image,label])
    #train start-----------------------------------------
    print('start train!!')
    index = 0
    epoch = 0
    all_loss = []
    all_acc = []
    all_acc_train = []
    change_learning_rate_flag = True
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        sess.run(tf.global_variables_initializer())

        for step in np.arange(maxstep):
            if coord.should_stop():
                break
            if step % 20 == 0:
                print 'test:'
                ii = 0
                cc = 0
                cc3 = 0

                for i in range(test_size/batch_size):
                    npn_input = get_npn_input(num_net,[isess1,isess2,isess3,isess4],
                                [g1,g2,g3,g4],
                                [_n1_end,_n2_end,_n3_end,_n4_end],
                                [_n1_x,_n2_x,_n3_x,_n4_x],
                                [_n1_x2,_n2_x2,_n3_x2,_n4_x2],
                                image_test[i*batch_size:i*batch_size+batch_size],
                                image_test_x2s[i*batch_size:i*batch_size+batch_size])

                    loss,acc,acc3 = sess.run([cross_entropy,accuracy,accuracy3], feed_dict={
                        x[0]: npn_input[0],x[1]: npn_input[1],x[2]: npn_input[2],x[3]: npn_input[3],
                        y: label_test[i*batch_size:i*batch_size+batch_size]})
                    ii = ii+1
                    cc = cc+acc
                    cc3 = cc3+acc3
                    #print(i),
                    #print("loss:{} accuracy:{} accuracy-top3:{}".format(loss,acc,acc3))
                if ii != 0:
                    print 'Total accuracy: ',cc/ii,'Total accuracy-top3: ',cc3/ii
                    all_acc.append(cc/ii)
                    all_acc_train.append(acc)      
                #test(sess,x,y_result)
                #-----------------------------------------            
            #train batch-----------------
            batch_xs, batch_ys = get_next_patch(image,label,index,batch_size)
            batch_x2s = get_batch_x2s(image[index:index+batch_size])
            npn_input = get_npn_input(num_net,[isess1,isess2,isess3,isess4],
                                [g1,g2,g3,g4],
                                [_n1_end,_n2_end,_n3_end,_n4_end],
                                [_n1_x,_n2_x,_n3_x,_n4_x],
                                [_n1_x2,_n2_x2,_n3_x2,_n4_x2],
                                batch_xs,
                                batch_x2s)
            '''batch_xs = image[index:index+batch_size]
            batch_ys = label[index:index+batch_size]
            batch_x2s = image_x2s[index:index+batch_size]'''
            index = batch_size+index
            if index > len(image)-batch_size:
                index = 0
                epoch += 1
                print 'shuffle:',epoch
                [image,label] = tut.shuffle_files([image,label])
                #[image,label,image_x2s] = tut.shuffle_files([image,label,image_x2s])
            #train batch-----------------

            sess.run(train_step,feed_dict = {x[0]: npn_input[0],x[1]: npn_input[1],x[2]: npn_input[2],x[3]: npn_input[3],y:batch_ys})
            if step % (32*16/batch_size) == 0:
                loss = sess.run([cross_entropy], 
                    feed_dict = {x[0]: npn_input[0],x[1]: npn_input[1],x[2]: npn_input[2],x[3]: npn_input[3],y:batch_ys})
                all_loss.append(loss)

            if step % 50 == 0:
                loss,acc = sess.run([cross_entropy,accuracy], 
                    feed_dict = {x[0]: npn_input[0],x[1]: npn_input[1],x[2]: npn_input[2],x[3]: npn_input[3],y:batch_ys})
                
                if change_learning_rate_flag and loss < 3:
                    train_step = train_step2
                    change_learning_rate_flag = False
                    print 'Change learning_rate!'
                print(step),
                print("loss:{} accuracy:{}".format(loss,acc))

                '''loss,acc = sess.run([cross_entropy,accuracy], feed_dict={
                    x: a, 
                    y: d})
                print("loss:{} accuracy:{}".format(loss,acc))'''

           
            if epoch >= max_epoch or step == maxstep-1:
                #saver.save(sess,"mymodel/my_model.ckpt")
                np.save('results_npn/acc_train_'+loss_file,all_acc_train)
                np.save('results_npn/acc_'+acc_file,all_acc)
                break

        
        
        coord.request_stop()
        coord.join(threads)
        sess.close()

def get_npn_input(num_net,sesss,gs,ys,xs,x2s,x,x2):
    ends = []
    for i in range(num_net):
        with gs[i].as_default():
            ends.append((sesss[i].run(ys[i],feed_dict = {xs[i]:x,x2s[i]:x2}))['PreLogits'])

    for i in range(num_net,len(sesss)):
         ends.append([None]*len(x))
    return ends

def load_model():
    maxstep = 3000
    learning_rate = 1e-3
    batch_size = 64
    test_image,test_label,train_image,train_label = tut.get_testfiles(testfilename)
    test_size = len(test_image)
    image_test,label_test = get_next_patch(test_image,test_label,0,test_size)

    image_test_x2s = get_batch_x2s(test_image)
    image_x2s = get_batch_x2s(train_image)

    image,label = train_image,train_label
    #image,label = get_next_patch(train_image,train_label,0,len(train_image))
    #image,label = image.tolist(),label.tolist()
    #[image,label,image_x2s] = tut.shuffle_files([image,label,image_x2s])
    [image,label] = tut.shuffle_files([image,label])
     
    x = tf.placeholder(tf.float32, [None, size_w*size_h])
    x = tf.reshape(x,shape=[-1,size_w,size_h,channel])
    y = tf.placeholder(tf.float32, [None, nclasses]) 
    #y = model_alex.mmodel(x)
    #y = model_vgg16.vgg16(x).probs
    #with slim.arg_scope(model_my.inception_v3_arg_scope()):
    #x2 = tf.placeholder(tf.float32, [None, 57])
    x2 = tf.placeholder(tf.float32, [None, 3,19])
    istrain = tf.placeholder(tf.bool, [1])

    #y,_ = model_my.inception_v3(x,inputs2=x2,num_classes = 12,is_training = False)  
    #arg_scope = model_resnet.resnet_arg_scope()
    #with slim.arg_scope(arg_scope):
    y_result,_ = model_resnet2.resnet_v2_101(x, num_classes = 12)
    #y = tf.reshape(y,shape=[-1,12])
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y,1),logits=y_result))
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    train_step2 = tf.train.AdamOptimizer(learning_rate*0.1).minimize(cross_entropy)
    # 计算准确率
    correct_pred = tf.equal(tf.argmax(y_result, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy = get_accuracy(y_result,tf.argmax(y,1),1)
    accuracy3 = get_accuracy(y_result,tf.argmax(y,1),3)
    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()#tf.global_variables()
        saver.restore(sess,'mymodel/saved_all/res_101/my_model.ckpt')
        while True:
            print 'test:'
            ii = 0
            cc = 0
            cc3 = 0

            for i in range(test_size/batch_size):
                loss,acc,acc3 = sess.run([cross_entropy,accuracy,accuracy3], feed_dict={
                    x: image_test[i*batch_size:i*batch_size+batch_size], 
                    y: label_test[i*batch_size:i*batch_size+batch_size],
                    x2:image_test_x2s[i*batch_size:i*batch_size+batch_size],istrain:[False]})
                ii = ii+1
                cc = cc+acc
                cc3 = cc3+acc3
                #print(i),
                #print("loss:{} accuracy:{} accuracy-top3:{}".format(loss,acc,acc3))
            if ii != 0:
                print 'Total accuracy: ',cc/ii,'Total accuracy-top3: ',cc3/ii     

def load_model_pre_logits():
    image_id = '4'
    x = tf.placeholder(tf.float32, [None, size_w*size_h])
    x = tf.reshape(x,shape=[-1,size_w,size_h,channel])
    x2 = tf.placeholder(tf.float32, [None, 3,19])
    y,end = model_my.inception_v3(x,inputs2=None,num_classes = 12,dropout_keep_prob = 1,is_training = False) 
    #y,end = model_resnet2.resnet_v2_101(x,num_classes = 12) 

    test_image,test_label,train_image,train_label = tut.get_testfiles(testfilename)
    test_size = len(test_image)
    image_test,label_test = get_next_patch(test_image,test_label,0,test_size)

    image_test_x2s = get_batch_x2s(test_image)
    image_x2s = get_batch_x2s(train_image)

    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()#tf.global_variables()
        saver.restore(sess,'mymodel/saved_all/google_rgb/my_model.ckpt')

        one_image = pim.parse_image('test_prelogits/'+image_id+'.jpg')
        batch_image = [one_image]*1
        image_test[0] = one_image

        h = pim.get_hist(one_image,20,top_level=220)
        image_x2s[0] = h
            #

        start_time = time.time()
        for i in range(1000):
            _y,_end = sess.run([y,end],feed_dict = {x:image_test[0:1],x2:image_x2s[0:1]})
        end_time = time.time()
        print end_time-start_time
        #np.save('test_prelogits/'+image_id+'_g_fin',_end['PreLogits'][0])



    '''pre = np.load('test_prelogits/'+image_id+'_g_fin.npy')
    pre = pre.reshape(2560)
    pre = pre[0:2048]
    pre = pre.reshape(32,2048/32)
    pre = pre * 255/pre.max()

    cv2.imwrite('test_prelogits/'+image_id+'_g_fin.jpg',pre)'''
    #matplotlib.image.imsave('test_prelogits/1_g_fin.jpg',aa)
    #scipy.misc.imsave('test_prelogits/1_g_fin.jpg',aa)
    #print pre.max()


def test(sess,x,y,istrain = None):
    dirname = testname
   
    whole_correct_count = 0.0
    whole_num = 0
    
    for pic_class in os.listdir(dirname):
        total_count = 0
        correct_count = 0

        for pic in os.listdir(dirname+'/'+pic_class):
            total_count = total_count+1

            one_image = pim.parse_image(dirname+pic_class+'/'+pic)
            batch_image = [one_image]

            pred = sess.run(y,feed_dict = {x:batch_image,istrain:[False]})
            #pred = pred[0][0]
            for i in pred:
                if pic_class == str((np.argsort(-i))[0]):
                    correct_count = correct_count+1
                print pic_class+'/'+pic ,' ',(np.argsort(-i))[0:6],i[(np.argsort(-i))[0]],i[(np.argsort(-i))[1]],\
                i[(np.argsort(-i))[2]]

        if total_count != 0:
            whole_correct_count  = whole_correct_count+float(correct_count)/float(total_count)
            whole_num = whole_num+1
            print pic_class,' ',float(correct_count)/float(total_count)
        
    
    if whole_num != 0:
        print 'whole accuracy:', float(whole_correct_count)/float(whole_num)

def get_test_images():
    images = []
    image_names = []
    labels = []
    labels_for_train = []
    dirname = testname
   
    whole_correct_count = 0.0
    whole_num = 0
    
    for pic_class in os.listdir(dirname):
        total_count = 0
        correct_count = 0

        for pic in os.listdir(dirname+'/'+pic_class):
            total_count = total_count+1
            one_image = pim.parse_image(dirname+pic_class+'/'+pic)
            images.append(one_image)
            image_names.append(pic_class+'/'+pic)
            labels.append(int(pic_class))

    labels_for_train = np.zeros([len(labels),nclasses])
    i = 0
    for lab in labels:
        labels_for_train[i][lab] = 1
        i = i+1

    return images,labels,image_names,labels_for_train

def test2(sess,x,y,istrain = None):   
        #batch_image,batch_label,image_names,labels_for_train = get_test_images()
        test_image,test_label,train_image,train_label = tut.get_testfiles(testfilename)
        test_size = len(test_image)
        image_test,label_test = get_next_patch(test_image,test_label,0,test_size)

        batch_size = 768/16
        #test_size = len(batch_image)
        index = 0
        count = 0
        for i in range(test_size/batch_size):
            pred = sess.run(y,feed_dict = {x:image_test[i*batch_size:i*batch_size+batch_size]})

            for p in pred:
       
                if (np.argsort(-p))[0]== test_label[index]:
                    count += 1
                
                #print p[0][0]
                print (np.argsort(-p))[0:6],test_label[index]
                index += 1
        print count      
            #index = 0
            #for index in range(len(batch_image)):
             #   print batch_label[index],' ',(np.argsort(-pred[index]))[0:6]#(np.argsort(-i))[0:6]#,i[(np.argsort(-i))[0]]
          


def test_other():
    #image = pim.parse_image('tools_py/76.png',channels=3)
    #h = pim.get_hist(image,20,top_level=220)

    '''dic = {} 
    dirname = testname
    for pic_class in os.listdir(dirname):
        for pic in os.listdir(dirname+'/'+pic_class):
            one_image = pim.parse_image(dirname+pic_class+'/'+pic,channels=3)
            h = pim.get_hist(one_image,20,top_level=220)
            dic[dirname+pic_class+'/'+pic] = h
            print dirname+pic_class+'/'+pic
    np.savez('pre_setup/hist_20_norm',dic)
    keys = hist_file.keys()

    index = 0
    start_time = time.time()
    for i in range(1000):
        index = 0

        #testh = hist_file[keys[np.random.randint(len(keys))]]
    end_time = time.time()
    print end_time-start_time
    a = [1,2,3,4,5,6]
    np.save('results/loss_1',a)'''
    
    #image,label = tut.get_testfiles('pre_setup/test_1_768')

    #print label[0:100]
    tut.generate_testfiles(trainname,48,rate = 0.2,save_path = 'pre_setup/test_')
    
    #print ddd.keys()

    #np.savez('pre_setup/npsavez',dic)

    #start_time = time.time()
    #end_time = time.time()
    #print end_time-start_time

    #image = parse_image('tools_py/191.jpg',size_w,size_h)
    #h = tf.summary.histogram('activations', img)
    #b = cv.split(image)[0]
    #g = cv.split(image)[1]
    #r = cv.split(image)[2]
  
    #h = plt.hist(b,facecolor='b')
   # h = plt.hist(g,facecolor='g')
    #n,bins,patches = plt.hist(r,bins = 10,facecolor='r')
    #print bins
    #print h.shape

    #print h
    print 'end'
    #plt.imshow(h)
    plt.show()


if __name__ == '__main__':
    #exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    arg = parser.parse_args()
    if arg.mode == '0':
        run_training()
    elif arg.mode == '1':
        load_model()
    elif arg.mode == '2':
        run_training2()
    elif arg.mode == '3':
        load_model_pre_logits()
    else:
        test_other()