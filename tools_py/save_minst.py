#--coding:utf-8 --
import struct
import numpy as np
#import  matplotlib.pyplot as plt
import Image

filename = './_data/train-images.idx3-ubyte'
filename_label = './_data/train-labels.idx1-ubyte'
binfile = open(filename,'rb')
buf = binfile.read()
binfile_label = open(filename_label,'rb')
buf_label = binfile_label.read()

index = 0;
index_label = 0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index += struct.calcsize('>IIII')
index_label += struct.calcsize('>II')



for image in range(0,numImages):
    
 
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')
   #这里注意 Image对象的dtype是uint8，需要转换
    im=np.array(im,dtype='uint8')
    im=im.reshape(28,28)
   # fig=plt.figure()
   # plotwindow=fig.add_subplot(111)
   # plt.imshow(im,cmap='gray')
   # plt.show()
    im=Image.fromarray(im)
    im_label = struct.unpack_from('>1B',buf_label,index_label)
    index_label += struct.calcsize('>1B')
    im_label = np.array(im_label,dtype='uint8')

    im.save('./train_mnist/'+str(im_label[0])+'/train_%s.jpg'%image,'jpeg')

  
  
