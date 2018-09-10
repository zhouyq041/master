import numpy as np
import matplotlib.pyplot as plt

def draw_loss(inputs):
	plt.figure(figsize = (6,4))

	for i in inputs:	
		yy = np.load(i[0])
		y = []
		min_acc = 10
		tmp = 10
		for yyy in yy:
			if yyy < tmp+0.1:
				y.append(yyy)
				tmp = min_acc
				min_acc = yyy
		if len(y)<60:
			y = yy
		

		size = len(y)
		size = size - size%60
		y = y[0:size]
		x = range(0,60)
		x = np.array(x) / 4.0
		
		if size>60:
			y = y[0:size:size/60]
		print size
		plt.plot(x,y,label = i[1],color = i[2])
		#plt.plot(x,y)
	plt.xticks([2,4,6,8,10,12,14])
	plt.grid(True,linestyle = '--',linewidth = 0.5)
	plt.legend()
	plt.show()


if __name__ == '__main__':
	draw_loss([
		#['loss_google_res_npn.npy','loss_google_res_npn','red'],
		#['loss_gray_fin.npy','gray fin','blue'],
		['loss_normal/loss_google_fin.npy','Loss of G-v3','#FF0000',0.9,3],
		#['loss_normal/loss_google_norm.npy','loss_rgb_norm','green',0.9,20],
		#['loss_normal/acc1_google_fin.npy','loss_rgb_norm','green',0.9,20],
		#['loss_normal/acc1_google_norm.npy','loss_rgb_norm','red',0.9,30],
		#['loss_normal/acc_train_google_fin.npy','loss_rgb_norm','blue',0.9,20],
		#['loss_normal/acc_train_google_norm.npy','loss_rgb_norm','yellow',0.9,20],
		#['../results_npn/acc_loss_npn1_res101_64.npy','loss_rgb_norm','blue',0.9,100],
		['../results_npn/loss_loss_npn1_res101_64.npy','Loss of NPN-3','yellow',0.9,200]
		#['loss_rgb_fin.npy','loss_rgb_fin','yellow']
		])