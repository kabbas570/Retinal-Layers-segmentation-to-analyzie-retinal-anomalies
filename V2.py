import os
import cv2
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import random
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import matplotlib.image as mpimg
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
def DataGenV():
    img_ = []
    mask_  = []
    for i in range(1):
        C1=np.zeros((256,256))
        image_path = os.path.join(images_folder,image_id[i])
        image = cv2.imread(image_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image=image/255
        mask_path = os.path.join(mask_folder,mask_id[i])
        mask = cv2.imread(mask_path)
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
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
        R=np.reshape(R,[65536,1])
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit_transform(R)
        target=np.reshape(onehot_encoded,[256,256,8])
        img_.append(image)
        mask_.append(target)
    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_
def DataGen():
    imageV_ = []
    maskV_  = []
    for j in range(10,11):
        C2=np.zeros((256,256))
        image_path = os.path.join(images_folder,image_id[j])
        image = cv2.imread(image_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
        image=image/255
        mask_path = os.path.join(mask_folder,mask_id[j])
        mask = cv2.imread(mask_path)
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
        C2[np.where(mask==62)]=1
        C2[np.where(mask==63)]=1
        C2[np.where(mask==116)]=2
        C2[np.where(mask==117)]=2
        C2[np.where(mask==149)]=3
        C2[np.where(mask==150)]=3
        C2[np.where(mask==151)]=3
        C2[np.where(mask==105)]=4
        C2[np.where(mask==106)]=4
        C2[np.where(mask==107)]=4
        C2[np.where(mask==163)]=5
        C2[np.where(mask==164)]=5
        C2[np.where(mask==165)]=5
        C2[np.where(mask==115)]=6
        C2[np.where(mask==114)]=6
        C2[np.where(mask==183)]=7
        C2[np.where(mask==182)]=7
        R=np.array(C2)
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
        R=np.reshape(R,[65536,1])
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded = onehot_encoder.fit_transform(R)
        target=np.reshape(onehot_encoded,[256,256,8])
        imageV_.append(image)
        maskV_.append(target)
    imageV_ = np.array(imageV_)
    maskV_ = np.array(maskV_)
    return imageV_,maskV_
epochs=1
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c
def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    outputs = keras.layers.Conv2D(8, (1, 1), padding="same", activation="softmax")(u4)
    outputs = keras.layers.Reshape([256*256,8])(outputs)
    i1=outputs[ ... ,8]
    model = keras.models.Model(inputs, outputs)
    return model,i1
model,layer = UNet()
l=layer
print(l)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()
valid_genx, valid_geny = DataGenV()
train_genx, train_geny = DataGen()

#l1=valid_genx[0]
#l2=valid_geny[0]
#p=l2.T
batch_size=1
train_steps = 1//batch_size
valid_steps = 1//batch_size
model.fit(train_genx,train_geny, validation_data=(valid_genx, valid_geny),steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=epochs)
## Save the Weights
model.save_weights("UNetW.h5")
I=train_geny[0]
'''
## Dataset for prediction

#g=train_genx[0,:,:,:]
#g = np.expand_dims(g, axis=0)
result = model.predict(train_genx)
L=result[0]
f1=L[:,:,0]
f2=L[:,:,1]
f3=L[:,:,2]
f4=L[:,:,3]
f5=L[:,:,4]
f6=L[:,:,5]
f7=L[:,:,6]
f8=L[:,:,7]

prediction=np.zeros([256,256])
prediction[np.where(f1==1)]=200
prediction[np.where(f2==1)]=150

prediction[np.where(f3==1)]=100
prediction[np.where(f4==1)]=120

prediction[np.where(f5==1)]=180
prediction[np.where(f6==1)]=240

prediction[np.where(f7==1)]=50
prediction[np.where(f8==1)]=10


F1=np.reshape(f1,[65536,1])
F2=np.reshape(f2,[65536,1])
F3=np.reshape(f3,[65536,1])
F4=np.reshape(f4,[65536,1])
F5=np.reshape(f5,[65536,1])
F6=np.reshape(f6,[65536,1])
F7=np.reshape(f7,[65536,1])
F8=np.reshape(f8,[65536,1])

I=train_geny[0]
background=I[:,:,0]
l1=I[:,:,1]
l2=I[:,:,2]
l3=I[:,:,3]
l4=I[:,:,4]
l5=I[:,:,5]
l6=I[:,:,6]
l7=I[:,:,7]
backgrounD=np.reshape(background,[256*256,1])
L1=np.reshape(l1,[256*256,1])
L2=np.reshape(l2,[256*256,1])
L3=np.reshape(l3,[256*256,1])
L4=np.reshape(l4,[256*256,1])
L5=np.reshape(l5,[256*256,1])
L6=np.reshape(l6,[256*256,1])
L7=np.reshape(l7,[256*256,1])
#R = np.stack((background,background,background), axis=0)
ground_truth=np.zeros([256,256])
ground_truth[np.where(background==1)]=200
ground_truth[np.where(l1==1)]=150

ground_truth[np.where(l2==1)]=100
ground_truth[np.where(l3==1)]=120

ground_truth[np.where(l4==1)]=180
ground_truth[np.where(l5==1)]=240

ground_truth[np.where(l6==1)]=50
ground_truth[np.where(l7==1)]=10

plt.figure(1)
plt.imshow(ground_truth, cmap = 'gray')
plt.show()'''