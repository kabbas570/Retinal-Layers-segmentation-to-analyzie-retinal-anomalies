import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
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
data1= tf.placeholder(tf.float32, name = "data_placeholder_a")

print(data1)
def DataGenV():
    img_ = []
    mask_  = []
    for i in range(1):
        image_path = os.path.join(images_folder,image_id[i])
        image = cv2.imread(image_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image=image/255
        mask_path = os.path.join(mask_folder,mask_id[i])
        mask = cv2.imread(mask_path)
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
        mask=mask/255
        #mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        img_.append(image)
        mask_.append(mask)

    img_ = np.array(img_)
    mask_  = np.array(mask_)
    return img_,mask_

valid_gen = DataGenV()

def DataGen():
    imageV_ = []
    maskV_  = []
    for j in range(20,21):
        image_path = os.path.join(images_folder,image_id[j])
        image = cv2.imread(image_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        

        image=image/255
        mask_path = os.path.join(mask_folder,mask_id[j])
        mask = cv2.imread(mask_path)
        mask=cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) 
        mask=mask/255
        
        #plt.imshow(mask)
        #plt.title('Ground Truth')
        #plt.show()
            
        imageV_.append(image)
        maskV_.append(mask)
        

    imageV_ = np.array(imageV_)
    maskV_  = np.array(maskV_)
    return imageV_,maskV_
train_genx, train_geny = DataGen()


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
    outputs = keras.layers.Conv2D(3, (1, 1), padding="same", activation="softmax")(u4)
    '''y_true_f = K.flatten(outputs)
    print("this is abbas kahnsd",y_true_f)
    with tf.Session() as sess:    
        init = tf.global_variables_initializer()
        sess.run(init)
        print( sess.run(y_true_f))
        #print(y_true_f.eval())
        print(y_true_f.shape)'''
    model = keras.models.Model(inputs, outputs)
    return model
model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()

valid_genx, valid_geny = DataGenV()
train_genx, train_geny = DataGen()
 

batch_size=1
train_steps = 1//batch_size
valid_steps = 1//batch_size

model.fit(train_genx,train_geny, validation_data=(valid_genx, valid_geny),steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=epochs)
## Save the Weights


## Dataset for prediction
#x, y = valid_gen.__getitem__(0)
'''
g=train_genx[0,:,:,:]
g = np.expand_dims(g, axis=0)
result = model.predict(train_genx)
L=result[2]
plt.axis("off")
plt.imshow(L)
plt.title('Predicted by Model')
plt.show()
plt.axis("off")
plt.imshow(train_geny[0])
plt.title('Ground Truth')
plt.show()'''
'''
model.load_weights('UNetW11.h5')
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
plt.show()'''

'''
from keras.models import load_model
model = load_model('UNetW.h5')
result = model.predict(train_genx)
L=result[0]'''
'''
plt.show()
plt.axis("off")
plt.imshow(L)
plt.title('Ground Truth')
plt.show()'''
with tf.Session() as sess:    
        init = tf.global_variables_initializer()
        sess.run(init)
        print(data1.eval())
