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
        labels[i][lab] = 1
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
    batch_size = 48   
    #set xy-------------------------------
    print 'set model'
    x = tf.placeholder(tf.float32, [None, size_w*size_h])
    x = tf.reshape(x,shape=[-1,size_w,size_h,channel])
    y = tf.placeholder(tf.float32, [None, nclasses]) 
    x2 = tf.placeholder(tf.float32, [None, 3,19])
    #x2 = tf.reshape(x2,shape=[-1,1,3,19])
    #y_result = model_alex.mmodel(x,drop_out)
    #y_result = model_vgg16.vgg16(x,drop_out).probs
    #with slim.arg_scope(model_my.inception_v3_arg_scope()): 
    y_result,_ = model_my.inception_v3(x,inputs2 = x2,num_classes = 12,is_training = True)
    
    #y_result,_ = model_google.inception_v3(x,num_classes = 12,is_training = True)
    
    #arg_scope = model_resnet.resnet_arg_scope(is_training = True)
    #with slim.arg_scope(arg_scope):
    #    y_result,_ = model_resnet.resnet_v2_101(x, num_classes = 12)
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
    #tf.add_to_collection("predict", p)
    #train prestep
    print 'train prestep'
    test_image,test_label,train_image,train_label = tut.get_testfiles(testfilename)
    test_size = len(test_image)
    image_test,label_test = get_next_patch(test_image,test_label,0,test_size)

    image_test_x2s = get_batch_x2s(test_image)
    image_x2s = get_batch_x2s(train_image)

    image,label = get_next_patch(train_image,train_label,0,len(train_image))
    image,label = image.tolist(),label.tolist()
    [image,label,image_x2s] = tut.shuffle_files([image,label,image_x2s])
    #train start-----------------------------------------
    print('start train!!')
    index = 0
    epoch = 0
    all_loss = []
    change_learning_rate_flag = True
    #a,b,c,d = get_test_images()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess,coord = coord)
        sess.run(tf.global_variables_initializer())


        for step in np.arange(maxstep):
            if coord.should_stop():
                break


            #train batch-----------------
            #[batch_xs, batch_ys,batch_x2s] = get_next_patch_in_mem([image,label,image_x2s],index,batch_size)
            batch_xs = image[index:index+batch_size]
            batch_ys = label[index:index+batch_size]
            batch_x2s = image_x2s[index:index+batch_size]
            index = batch_size+index
            if index > len(image)-batch_size:
                index = 0
                epoch += 1
                print 'shuffle:',epoch
                [image,label,image_x2s] = tut.shuffle_files([image,label,image_x2s])
            #train batch-----------------

            sess.run(train_step,feed_dict = {x:batch_xs,y:batch_ys,x2:batch_x2s})
            
            if step % 50 == 0:
                loss,acc = sess.run([cross_entropy,accuracy], feed_dict={x: batch_xs, y: batch_ys,x2:batch_x2s})
                all_loss.append(loss)
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

            if step % 200 == 0:
                print 'test:'
                ii = 0
                cc = 0
                cc3 = 0
                for i in range(test_size/batch_size):
                    loss,acc,acc3 = sess.run([cross_entropy,accuracy,accuracy3], feed_dict={
                        x: image_test[i*batch_size:i*batch_size+batch_size], 
                        y: label_test[i*batch_size:i*batch_size+batch_size],
                        x2:image_test_x2s[i*batch_size:i*batch_size+batch_size]})
                    ii = ii+1
                    cc = cc+acc
                    cc3 = cc3+acc3
                    #print(i),
                    #print("loss:{} accuracy:{} accuracy-top3:{}".format(loss,acc,acc3))
                if ii != 0:
                    print 'Total accuracy: ',cc/ii,'Total accuracy-top3: ',cc3/ii          
                #test(sess,x,y_result)
                #-----------------------------------------

            if step == maxstep-1:
                saver.save(sess,"mymodel/my_model.ckpt")
                np.save('results/loss_rgb_fin',all_loss)

        
        
        coord.request_stop()
        coord.join(threads)
        sess.close()



def load_model():

    x = tf.placeholder(tf.float32, [None, size_w*size_h])
    x = tf.reshape(x,shape=[-1,size_w,size_h,channel])
    y = tf.placeholder(tf.float32, [None, nclasses]) 
    #y = model_alex.mmodel(x)
    #y = model_vgg16.vgg16(x).probs
    #with slim.arg_scope(model_my.inception_v3_arg_scope()):
    x2 = tf.placeholder(tf.float32, [None, 3,19])

    y,_ = model_my.inception_v3(x,inputs2=x2,num_classes = 12,is_training = False)
    #y,_ = model_google.inception_v3(x,num_classes = 12,is_training = False)

       
        #arg_scope = model_resnet.resnet_arg_scope(is_training = True)
        #with slim.arg_scope(arg_scope):
        #    y,_ = model_resnet.resnet_v2_101(x, num_classes = 12)
    #y = tf.reshape(y,shape=[-1,12])
    with tf.Session() as sess:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        #saver = tf.train.import_meta_graph('mymodel/my_model.ckpt.meta')
        #saver.restore(sess,'mymodel/google_good/my_model.ckpt')
        saver.restore(sess,'mymodel/my_model.ckpt')#tf.train.latest_checkpoint('mymodel/'))
        #graph = tf.get_default_graph() 

        test(sess,x,y,x2)

def test(sess,x,y,x2 = None):
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
            #print dirname+pic_class+'/'+pic
            pred = sess.run(y,feed_dict = {x:batch_image,x2:[pim.get_hist(one_image,20,top_level=220)]})

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

def test2(sess,x,y):   
        batch_image,batch_label,image_names,labels_for_train = get_test_images()

        pred = sess.run(y,feed_dict = {x:batch_image})

        index = 0
        for index in range(len(batch_image)):
            print batch_label[index],' ',(np.argsort(-pred[index]))[0:6]#(np.argsort(-i))[0:6]#,i[(np.argsort(-i))[0]]
          


def test_other():
    #image = pim.parse_image('tools_py/76.png',channels=3)
    #h = pim.get_hist(image,20,top_level=220)

    dic = {} 
    dirname = testname
    for pic_class in os.listdir(dirname):
        for pic in os.listdir(dirname+'/'+pic_class):
            one_image = pim.parse_image(dirname+pic_class+'/'+pic,channels=3)
            h = pim.get_hist(one_image,20,top_level=220)
            dic[dirname+pic_class+'/'+pic] = h
            print dirname+pic_class+'/'+pic
    np.savez('pre_setup/hist_20_norm',dic)
    '''keys = hist_file.keys()

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
    #tut.generate_testfiles(trainname,48,rate = 0.2,save_path = 'pre_setup/test_1')
    
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
    else:
        test_other()