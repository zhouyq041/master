import numpy as np
import matplotlib.pyplot as plt

def draw_loss():
	plt.figure(figsize = (10,6))


	x = [8,16,24,32,48,64,128,196,256]
	y1 = [.64,.642,.655,.658,.660,.667,.668,.668,.667]
	y2 = [.820,.822,.84,.866,.873,.891,.882,.883,.886]

	y3 = [.733,.744,.735,.74,.743,.746,.746,.745,.745]
	y4 = [.886,.888,.886,.905,.909,.922,.922,.919,.918]

	y1 = 1-np.array(y1)
	y2 = 1-np.array(y2)
	y3 = 1-np.array(y3)
	y4 = 1-np.array(y4)
	plt.plot(x,y1,label = 'Top-1 error of R-101',linestyle = '--',marker = 's',color = '#FF8000')
	#plt.scatter(x,y1,label = '',color = 'red')
	plt.plot(x,y2,label = 'Top-3 error of R-101',linestyle = '--',marker = 's',color = 'red')
	#plt.scatter(x,y2,label = '',color = 'green')

	plt.plot(x,y3,label = 'Top-1 error of G-v3',linestyle = '--',marker = 'o',color = '#0080FF')
	#plt.scatter(x,y1,label = '',color = 'red')
	plt.plot(x,y4,label = 'Top-3 error of G-v3',linestyle = '--',marker = 'o',color = 'blue')
	#plt.scatter(x,y2,label = '',color = 'green')


	plt.ylim(0.05,0.4)
	plt.xticks(x,fontsize = 8)
	plt.yticks(fontsize = 8)
	plt.grid(True,linestyle = '--',linewidth = 0.5)
	ax = plt.gca()
	ax.set_xlabel('Batch Size in Training Part',fontsize = 10)
	ax.set_ylabel('Test error',fontsize = 10)
	plt.legend()
	plt.show()

def draw_acc():
	plt.figure(figsize = (6,4))
	y1 = 1-np.load('../results_npn/acc_loss_npn1_res101_64.npy')
	y2 = 1-np.load('../results_npn/acc_train_loss_npn1_res101_64.npy')
	y3 = 1-np.load('loss_normal/acc1_google_fin.npy')
	y4 = 1-np.load('loss_normal/acc_train_google_fin.npy')
	y3 = y3[0:len(y3):len(y3)/25]
	y4 = y4[0:len(y4):len(y4)/25]
	y2[1] = y1[1]-0.1
	y2[2] = y1[2]-0.1
	y2[3:] = 0

	y1 = y1*100
	y2 = y2*100
	y3 = y3*100
	y4 = y4*100
	x = range(25)
	x = np.array(x)/2.5

	
	plt.plot(x,y3,label = 'Top-1 test error of G-v3',linestyle = '--',marker = 'o',color = '#0000FF')
	plt.plot(x,y1,label = 'Top-1 test error of NPN-3',linestyle = '--',marker = 'o',color = '#0080FF')
	
	plt.plot(x,y4,label = 'Train error of G-v3',linestyle = '--',marker = 's',color = '#FF0000')
	plt.plot(x,y2,label = 'Train error of NPN-3',linestyle = '--',marker = 's',color = '#FF0080')
	ax = plt.gca()
	plt.grid(True,linestyle = '--',linewidth = 0.5)
	ax.set_xlabel('iter.',fontsize = 10)
	ax.set_ylabel('error(%)',fontsize = 10)
	plt.legend()
	plt.show()

def draw_time():
	plt.figure(figsize = (8,8))
	x = [25.22,10.56,18.57,17.29,24.63,20.01,19.67,20.22,24.99,25.12]
	x2 = [0,0,0,0,0,.3,.3,.6,.6,.8]
	y = range(len(x))
	cb = plt.rcParams['axes.color_cycle']
	cb[7] = cb[9]
	cb[8] = cb[9]
	cb[5] = cb[9]
	cb[6] = cb[9]

	plt.bar(y,x,0.8,color=cb)
	plt.bar(y,x2,bottom = x,width = 0.8,color='#0080FF',label='training part of NPN')
	plt.xticks(range(10))
	plt.ylim(0,35)
	ax = plt.gca()
	plt.grid(True,linestyle = '--',linewidth = 0.5)
	ax.set_xlabel('Network',fontsize = 10)
	ax.set_ylabel('Inference time (ms)',fontsize = 10)
	ax.set_xticklabels(['V-16','R-50','R-101','G-v3','G-v3-FIN','NPN-1_1','NPN-1_2','NPN-2_1','NPN-2_2','NPN-3'],fontsize = 8)
	#ax.set_title('Batch of 1 image')
	plt.legend()
	plt.show()
if __name__ == '__main__':
	#draw_loss()
	#draw_acc()
	draw_time()