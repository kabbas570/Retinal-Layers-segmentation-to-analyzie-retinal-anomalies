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
from sklearn.preprocessing import OneHotEncoder
path = "tt/"
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed
maskV_  = []
image_size=256
dest_path2="C:/Users/Abbas/Desktop/TH1/tt/M"
train_ids = next(os.walk(path))[1]
images_folder = os.path.join(path, train_ids[0]) 
mask_folder = os.path.join(path, train_ids[1])
image_id = next(os.walk(images_folder))[2]
mask_id = next(os.walk(mask_folder))[2]


for i in range(1):
    mask_path = os.path.join(mask_folder,mask_id[i])
    mask = cv2.imread(mask_path)
    C1=np.zeros((256,256))
    mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #mask[np.where(mask==115)]=0
    #mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #gray=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    '''plt.figure(0)
    plt.imshow(mask, cmap = 'gray')
    plt.show()'''
    ##layer_111
    C1[np.where(mask==62)]=1
    C1[np.where(mask==63)]=1
    C1[np.where(mask==116)]=2
    C1[np.where(mask==117)]=2
    C1[np.where(mask==149)]=3
    C1[np.where(mask==150)]=3
    C1[np.where(mask==151)]=3
    C1[np.where(mask==105)]=4
    C1[np.where(mask==106)]=4
    C1[np.where(mask==107)]=4
    C1[np.where(mask==163)]=5
    C1[np.where(mask==164)]=5
    C1[np.where(mask==165)]=5
    C1[np.where(mask==115)]=6
    C1[np.where(mask==114)]=6
    C1[np.where(mask==183)]=7
    C1[np.where(mask==182)]=7
    '''plt.figure(1)
    plt.imshow(C1, cmap = 'gray')
    plt.show()'''
    R=np.array(C1)
    R=np.reshape(R,[256,256])
    h=0
    while(h<=256-2):
        w=0
        while(w<=256-2):
            dumy=R[h:h+2,w:w+2]
            m=np.amax(dumy)
            if (dumy[0][0]+dumy[0][1]+dumy[1][0]+dumy[1][1]!=0):
             dumy[np.where(dumy==0)]=m
             #R[h:h+2,w:w+2]=dumy
            w=w+2
        h=h+2 #this+1 is stride
    '''plt.figure(1)
    plt.imshow(R, cmap = 'gray')
    plt.show()'''
    R=np.reshape(R,[65536,1])
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(R)
    target=np.reshape(onehot_encoded,[256,256,8])
    maskV_.append(target)
Y=target[:,:,0]
    