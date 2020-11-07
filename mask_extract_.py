from keras import backend as K
import os
import cv2
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import random
import tensorflow as tf
from tensorflow import keras
import matplotlib.image as mpimg
path = "tt/"
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed
image_size=256
dest_path2="C:/Users/Abbas/Desktop/TH1/tt/M"
train_ids = next(os.walk(path))[1]
images_folder = os.path.join(path, train_ids[0]) 
mask_folder = os.path.join(path, train_ids[1])
image_id = next(os.walk(images_folder))[2]
mask_id = next(os.walk(mask_folder))[2]
C1=np.zeros((256,256))
C2=np.zeros((256,256))
C3=np.zeros((256,256))
C4=np.zeros((256,256))
C5=np.zeros((256,256))
C6=np.zeros((256,256))
C7=np.zeros((256,256))

mask_path = os.path.join(mask_folder,mask_id[1])
mask = cv2.imread(mask_path,0)


#mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#gray=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
plt.figure(0)
plt.imshow(mask, cmap = 'gray')
plt.show()
##layer_111

C1[np.where(gray==61)]=1
C1[np.where(gray==62)]=1
C1[np.where(gray==63)]=1


kernel = np.ones((2,2),np.uint8)
kernel1 = np.ones((1,4),np.uint8)
#C1 = cv2.erode(C1,kernel,iterations =1)
C1 = cv2.dilate(C1,kernel,iterations = 2)
plt.figure(1)
plt.imshow(C1, cmap = 'gray', interpolation = 'bicubic')
plt.show()

C1 = cv2.convertScaleAbs(C1, alpha=(255.0))
cv2.imwrite(os.path.join(dest_path2 , str(0)+"_1"+".png"),C1)
#leyer_2222
C2[np.where(gray==116)]=1
C2[np.where(gray==117)]=1

C2 = cv2.erode(C2,kernel,iterations =1)
C2 = cv2.dilate(C2,kernel,iterations = 1)


C2 = cv2.convertScaleAbs(C2, alpha=(255.0))
cv2.imwrite(os.path.join(dest_path2 , str(0)+"_2"+".png"),C2)
#layer_333
C3[np.where(gray==150)]=1
C3[np.where(gray==151)]=1
C3[np.where(gray==149)]=1

C3 = cv2.erode(C3,kernel,iterations =1)
C3 = cv2.dilate(C3,kernel,iterations = 1)

C3 = cv2.convertScaleAbs(C3, alpha=(255.0))
cv2.imwrite(os.path.join(dest_path2 , str(0)+"_3"+".png"),C3)

#LAYER_444
C4[np.where(gray==105)]=1
C4[np.where(gray==106)]=1
C4[np.where(gray==107)]=1

C4 = cv2.erode(C4,kernel,iterations =1)
C4 = cv2.dilate(C4,kernel,iterations = 1)


C4 = cv2.convertScaleAbs(C4, alpha=(255.0))
cv2.imwrite(os.path.join(dest_path2 , str(0)+"_4"+".png"),C4)

#LAYER_5555
C5[np.where(gray==163)]=1
C5[np.where(gray==164)]=1
C5[np.where(gray==165)]=1
C5 = cv2.erode(C5,kernel,iterations =1)
C5 = cv2.dilate(C5,kernel,iterations = 1)


C5 = cv2.convertScaleAbs(C5, alpha=(255.0))
cv2.imwrite(os.path.join(dest_path2 , str(0)+"_5"+".png"),C5)

#LAYER_6666
C6[np.where(gray==114)]=1
C6[np.where(gray==115)]=1
C6 = cv2.erode(C6,kernel,iterations =1)
C6 = cv2.dilate(C6,kernel,iterations = 1)


C6 = cv2.convertScaleAbs(C6, alpha=(255.0))
cv2.imwrite(os.path.join(dest_path2 , str(0)+"_6"+".png"),C6)

#LAYER_7777
C7[np.where(gray==182)]=1
C7[np.where(gray==183)]=1
C7[np.where(gray==184)]=1

C7 = cv2.erode(C7,kernel,iterations =1)
C7 = cv2.dilate(C7,kernel,iterations = 1)
C7 = cv2.convertScaleAbs(C7, alpha=(255.0))
cv2.imwrite(os.path.join(dest_path2 , str(0)+"_7"+".png"),C7)
