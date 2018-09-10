import numpy as np
import os

class my_train_utils:
	def __init__(self):
		return

	def get_files(self,filename):
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


	def shuffle_files(self,data):
		temp = np.array(data)
		temp = temp.transpose()
		#shuffle the samples
		np.random.shuffle(temp)
		#after transpose, images is in dimension 0 and label in dimension 1
		ret_data = []
		for data_i in range(len(data)):
			ret_data.append(list(temp[:,data_i]))
		#label_list = [int(i) for i in label_list]
		return  ret_data


	def get_next_patch(self,image,label,index,batch_size,resize_w,resize_h):
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

	def generate_testfiles(self,dataset_path,batch_size,rate = 0.2,save_path = None):
		all_image_name,all_label = self.get_files(dataset_path)
		test_size = int(len(all_image_name)*rate/batch_size)*batch_size

		test_image_name = all_image_name[0:test_size]
		test_label = all_label[0:test_size]
		train_image_name = all_image_name[test_size:]
		train_label = all_label[test_size:]
		test_dict = {}
		test_dict['test_image'] = test_image_name
		test_dict['test_label'] = test_label
		test_dict['train_image'] = train_image_name
		test_dict['train_label'] = train_label
		if save_path != None:
			np.savez(save_path+'_'+str(len(test_image_name)),test_dict)
			print 'save testfiles:',save_path+'_'+str(len(test_image_name)),'train nums:',len(train_image_name)

	def get_testfiles(self,data_path):
		test_files = np.load(data_path+'.npz')['arr_0'].tolist()
		return test_files['test_image'],test_files['test_label'],\
				test_files['train_image'],test_files['train_label']





'''def get_batches(image,label,resize_w,resize_h,batch_size,capacity):
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
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size = batch_size,
                                             num_threads = 64,
                                             capacity = capacity)#,min_after_dequeue = capacity/2)
    images_batch = tf.cast(image_batch,tf.float32)
    labels_batch = tf.reshape(label_batch,[batch_size])

    return images_batch,labels_batch'''