import os  
import math  
import numpy as np  
import tensorflow as tf  
from scipy.misc import imread, imresize


train_dir = "./data"#"~/zhouyq/data"



def get_files(file_dir):
	#step1 prepare list
	parameters = {
		'n01':'bee',
		'n02':'panda',
		'n03':'penguin',
	}

	parameters = {
		'n01':'0',
		'n02':'1',
		'n03':'2',
	}
	image_01 = []
	label_01 = []
	image_02 = []
	label_02 = []
	image_03 = []
	label_03 = []
	for file in os.listdir(file_dir+'/'+parameters['n01']):
		image_01.append(file_dir +'/'+parameters['n01']+'/'+ file)
		label_01.append(0) 
	for file in os.listdir(file_dir+'/'+parameters['n02']):  
		image_02.append(file_dir +'/'+parameters['n02']+'/'+file)  
		label_02.append(1)  
	#for file in os.listdir(file_dir+'/'+parameters['n03']):  
	#	image_03.append(file_dir +'/'+parameters['n03']+'/'+ file)   
	#	label_03.append(2)  
	image_list = np.hstack((image_01, image_02))  
	label_list = np.hstack((label_01, label_02))

	#step2 shuffle
	temp = np.array([image_list, label_list])  
	temp = temp.transpose()  
	np.random.shuffle(temp)

	image_list = list(temp[:, 0])  
	label_list = list(temp[:, 1])  
	label_list = [int(i) for i in label_list]  

	return image_list, label_list

def get_batch(image,label,resize_w,resize_h,batch_size,capacity,channels = 3):
    #convert the list of images and labels to tensor
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int64)
    queue = tf.train.slice_input_producer([image,label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c,channels)
    #resize
    image = tf.image.resize_image_with_crop_or_pad(image,resize_w,resize_h)
    #(x - mean) / adjusted_stddev
    image = tf.image.per_image_standardization(image)
    
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size = batch_size,
                                             num_threads = 64,
                                             capacity = capacity)
    images_batch = tf.cast(image_batch,tf.float32)
    labels_batch = tf.reshape(label_batch,[batch_size])
    return images_batch,labels_batch



if __name__ == '__main__':
	image_list, label_list = get_files(train_dir)
	images_batch,labels_batch = get_batch(image_list, label_list,32,32,16,20)
	print images_batch.get_shape()[:2]