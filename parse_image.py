# coding=utf-8
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
class my_parse_image:
	def __init__(self, resize_w, resize_h, channels):
		self.w = resize_w
		self.h = resize_h
		self.c = channels

	def parse_image(self,image_path,channels = None):
	    '''image_tmp = tf.read_file(tf.cast(img,tf.string))
	    image_tmp = tf.image.decode_jpeg(image_tmp,channels = 3)
	    image_tmp = tf.image.resize_image_with_crop_or_pad(image_tmp,resize_w,resize_h)'''
	    c = self.c
	    if channels != None:
	    	c = channels
	    if c == 1:
	    	image_tmp = cv.imread(image_path,0)
	    else:
	    	image_tmp = cv.imread(image_path)#, mode='RGB')
	    image_tmp = cv.resize(image_tmp, (self.w, self.h))
	    #image_tmp = image_tmp/255.0

	    #image_tmp = cv.Canny(image_tmp , 50 , 200)
	    image_tmp = image_tmp.reshape(self.w,self.h,c)
	    
	    return image_tmp

	def get_hist(self,image,bins,top_level = 255,ifnorm = True):
		if bins < 1 or bins > top_level:
			return []

		retb = [0]*(bins+3)
		retg = [0]*(bins+3)
		retr = [0]*(bins+3)

		b = cv.split(image)[0]
		g = cv.split(image)[1]
		r = cv.split(image)[2]

		batch_size = 255/bins
		
		for row in b:
			for color in row:
				if color < top_level:
					retb[int(color/batch_size)] += 1
		for row in g:
			for color in row:
				if color < top_level:
					retg[int(color/batch_size)] += 1
		for row in r:
			for color in row:
				if color < top_level:
					retr[int(color/batch_size)] += 1

		top_index = top_level/batch_size+1
		if ifnorm:
			def norm(array):
				Max = array.max()
				Min = array.min()
				return (array-Min+0.0)/(Max-Min)

			a1 = norm(np.array(retb[0:top_index]))
			a2 = norm(np.array(retg[0:top_index]))
			a3 = norm(np.array(retr[0:top_index]))
			h = np.array([a1,a2,a3])
		else:
			h = np.array([retb[0:top_index],retg[0:top_index],retr[0:top_index]])
		#h = h.reshape(1,1,h.size)
		return h
