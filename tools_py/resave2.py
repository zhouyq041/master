import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt



resize_w = 151
resize_h = 151
filename = "../cartoon_dataset/traindata/"
for train_class in os.listdir(filename):
	for pic in os.listdir(filename+train_class):
		image_name = filename+train_class+'/'+pic
		image = cv.imread(image_name)
#plt.imshow(image)
#plt.show()
		cv.imwrite(image_name,image)
#image_tmp = cv.resize(image, (resize_w, resize_h))
    #image_tmp = cv.Canny(image_tmp , 50 , 200)
#image_tmp = image_tmp.reshape(resize_w,resize_h,3)
