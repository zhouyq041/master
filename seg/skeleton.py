#coding:utf-8

from skimage import morphology,data,color
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('77.png',0)
#image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = color.rgb2gray(image)
image =(image <= 220)*1.0

#image = color.rgb2gray(data.horse())


#ret,image = cv2.threshold(image,230,255,cv2.THRESH_BINARY)

image = 1-image #反相
#实施骨架算法
skeleton =morphology.skeletonize(image)

#显示结果
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ax1.imshow(image, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original', fontsize=20)

ax2.imshow(skeleton, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('skeleton', fontsize=20)

fig.tight_layout()
plt.show()