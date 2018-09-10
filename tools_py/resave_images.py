import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imread, imresize,imsave


filename = "../cartoon_dataset/traindata/"
for train_class in os.listdir(filename):
	for pic in os.listdir(filename+train_class):
		image_name = filename+train_class+'/'+pic
		print image_name

		image = cv2.imread(image_name,cv2.IMREAD_UNCHANGED)
		#plt.imshow(image)
		#plt.show()
		#image = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB) 
		channels = len(cv2.split(image))
		if channels == 4:
			b = cv2.split(image)[0]
			g = cv2.split(image)[1]
			r = cv2.split(image)[2]
			a = cv2.split(image)[3]
			#for i in range(20):
				#print a[i]
			a = (255-a)/255*255

			image = cv2.merge([b+a,g+a,r+a])


			cv2.imwrite(image_name,image)
		elif channels == 0:
			image = imread(image_name)
			imsave(image_name,image)
