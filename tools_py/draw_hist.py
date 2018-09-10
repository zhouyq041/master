# coding=utf-8
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def get_hist(image,bins,top_level = 255,ifnorm = True):
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
			h = np.array([retb[1:top_index],retg[1:top_index],retr[1:top_index]])
		#h = h.reshape(1,1,h.size)
		return h




#ax = plt.gca()
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
bins = 20
size = 10
image = cv.imread('real.jpg')
h = get_hist(image,bins,top_level=220,ifnorm = False)

[b,g,r] = h/(h.max()+0.0)
x = range(0,(260/len(b))*len(b),260/len(b))
#plt.figure()


plt.subplot(331)
plt.bar(x,b,size,color='b')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)
plt.subplot(332)
plt.bar(x,g,size,color='g')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)
plt.subplot(333)
plt.bar(x,r,size,color='r')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)

image = cv.imread('IUS_b.png')
h = get_hist(image,bins,top_level=220,ifnorm = False)

[b,g,r] = h/(h.max()+0.0)
print cv.split(image)[0]
x = range(0,(260/len(b))*len(b),260/len(b))
#b = b/(b.max()+0.0)
#g = g/(g.max()+0.0)
#r = r/(r.max()+0.0)
plt.subplot(334)
plt.bar(x,b,size,color='b')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)
plt.subplot(335)
plt.bar(x,g,size,color='g')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)
plt.subplot(336)
plt.bar(x,r,size,color='r')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)

image = cv.imread('IUS.png')
h = get_hist(image,bins,top_level=220,ifnorm = False)

[b,g,r] = h/(h.max()+0.0)
print cv.split(image)[0]
x = range(0,(260/len(b))*len(b),260/len(b))
#b = b/(b.max()+0.0)
#g = g/(g.max()+0.0)
#r = r/(r.max()+0.0)
plt.subplot(337)
plt.bar(x,b,size,color='b')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)
plt.subplot(338)
plt.bar(x,g,size,color='g')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)
plt.subplot(339)
plt.bar(x,r,size,color='r')
plt.ylim(0,1)
plt.grid(True,linestyle = '--',linewidth = 0.5)
#plt.hist(y,10,facecolor='b')
#plt.title()
#plt.grid(True,linestyle = '--',linewidth = 1)
plt.show()