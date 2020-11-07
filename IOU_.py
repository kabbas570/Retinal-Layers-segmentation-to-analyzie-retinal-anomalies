from tensorflow.python.keras import backend as K
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
path = "DATA/"
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed
image_size=256
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
C8=np.zeros((256,256))
C9=np.zeros((256,256))
mask_path = os.path.join(mask_folder,mask_id[5])
mask = cv2.imread(mask_path)
#mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

plt.axis()
plt.figure(0)
plt.imshow(mask, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()

gray=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
'''C1[np.where(gray==62)]=1
plt.axis()
plt.figure(1)
plt.imshow(C1, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()

C2[np.where(gray==116)]=1
plt.axis()
plt.figure(2)
plt.imshow(C2, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()

C3[np.where(gray==150)]=1
plt.axis()
plt.figure(3)
plt.imshow(C3, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()'''

C4[np.where(gray==106)]=1
plt.axis()
plt.figure(0)
plt.imshow(C4, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()
'''
C5[np.where(gray==164)]=1
plt.axis()
plt.figure(5)
plt.imshow(C5, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()

C6[np.where(gray==115)]=1
C6=np.array(C6)
plt.axis()
plt.figure(6)
plt.imshow(C6, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()

C7[np.where(gray==183)]=1
plt.axis()
plt.figure(7)
plt.imshow(C7, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()

C8[np.where(gray==140)]=1
plt.axis()
plt.figure(8)
plt.imshow(C8, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()'''

print(C4[78][222])
for i in range(255):
    for j in range(255):
        if(C4[i][j]*C4[i][j+1]*C4[i][j-1]==1):
            C9[i][j]=1
plt.axis()
plt.figure(1)
plt.imshow(C9, cmap = 'gray', interpolation = 'bicubic')
plt.title('Predicted by Model')
plt.show()
