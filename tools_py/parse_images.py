#coding:utf-8
from scipy.misc import imread, imresize,imsave
import os 

filename = "../data2/"
savename = "../data/"
count = 0
for train_class in os.listdir(filename):
	for pic in os.listdir(filename+train_class):
		count = count+1	
		try:
			im = imread(filename+train_class+'/'+pic, mode='RGB')
			imsave(savename+train_class+'/'+pic,im,'jpeg')
		except tf.errors.OutOfRangeError:
        	print "Done!!!" 
   		finally:
			print count


			