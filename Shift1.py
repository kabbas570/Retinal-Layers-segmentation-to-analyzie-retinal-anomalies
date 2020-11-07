import os
import cv2
from scipy.ndimage.interpolation import shift
import numpy as np
import random
import tensorflow as tf
path = "tt/"
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
dest_path1="C:/Users/Abbas/Desktop/augmentation/DATA/images"
dest_path2="C:/Users/Abbas/Desktop/augmentation/DATA/masks"
for i in range(106):
    image_path = os.path.join(images_folder,image_id[i])
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image,(image_size, image_size))
    shifted_imageU = shift(image, [-10,0,0],mode='nearest')
    shifted_imageU = shifted_imageU.reshape((256, 256,3))
    shifted_imageB = shift(image, [20,0,0],mode='nearest')
    shifted_imageB = shifted_imageB.reshape((256, 256,3))  
    cv2.imwrite(os.path.join(dest_path1 , str(i)+"SU"+".png"), shifted_imageU)
    cv2.imwrite(os.path.join(dest_path1 , str(i)+"SB"+".png"), shifted_imageB)
    mask_path = os.path.join(mask_folder,mask_id[i])
    mask = cv2.imread(mask_path, 1)
    mask = cv2.resize(mask,(image_size, image_size))
    shifted_maskU = shift(mask, [-10,0,0],mode='nearest')
    shifted_maskU = shifted_maskU.reshape((256, 256,3))
    shifted_maskB = shift(mask, [20,0,0],mode='nearest')
    shifted_maskB = shifted_maskB.reshape((256, 256,3))
    cv2.imwrite(os.path.join(dest_path2 , str(i)+"SMU"+".png"), shifted_maskU)
    cv2.imwrite(os.path.join(dest_path2 , str(i)+"SMB"+".png"), shifted_maskB)




