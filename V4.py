import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras import backend as K
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
'''def calculate_iou(y_true , y_pred):
    #L1=y_pred[ ... ,0]
    #L2=y_true[ ... ,0]
    #P=keras.layers.Flatten()(L1)
    #T=keras.layers.Flatten()(L2)
    print('this_T SHAPE',y_true)
    print('this_P SHAPE',y_pred)
    P=y_true+y_true

    #tn, fp, fn, tp = confusion_matrix(T,P).ravel()
    for t in range(8):
        tn, fp, fn, tp = confusion_matrix(K.reshape(L[ ... ,t],[65536,1]),K.reshape(L1[ ... ,t],[65536,1])).ravel()
        IOU1=tp/(tp+fp+fn)
        IOU=IOU1+IOU
    return P
def iou_metric( y_true , y_pred ):
    return calculate_iou( y_true , y_pred ) 
def custom_loss( y_true , y_pred ):
    iou = calculate_iou( y_true , y_pred ) 
    return  1 - iou''' 
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
    u3 = up_block(u2, c2, f[1]) #32 -> 64sw
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    outputs = keras.layers.Conv2D(8, (1, 1), padding="same", activation="softmax")(u4)
   
    #outputs = keras.layers.Reshape([8,256*256])(outputs)
    model = keras.models.Model(inputs, outputs)
    return model, outputs
model, outputs = UNet()
valid_genx, valid_geny = DataGenV()
train_genx, train_geny = DataGen()
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=8)])
model.summary()
batch_size=1
train_steps = 1//batch_size
valid_steps = 1//batch_size
model.fit(train_genx,train_geny, validation_data=(valid_genx, valid_geny),steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=epochs)
## Save the Weights
model.save_weights("UNetW.h5")
result = model.predict(train_genx)
evol=model.evaluate(train_genx,train_geny)
L=result[0]
T=train_geny[0]
plt.figure(0)
plt.axis("off")
plt.imshow(L)
plt.title('Predicted')
plt.show()
plt.figure(1)
plt.axis("off")
plt.imshow(T)
plt.title('Ground Truth')
plt.show()
mask=cv2.cvtColor(L, cv2.COLOR_RGB2GRAY) 
mask=mask*255
plt.figure(2)
plt.axis("off")
plt.imshow(mask)
plt.title('Ground Truth')
plt.show()