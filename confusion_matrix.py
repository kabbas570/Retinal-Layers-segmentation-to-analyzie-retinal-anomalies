from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
P=np.array([[0,1],[0,1]])
T=np.array([[0,1],[0,1]])
tn, fp, fn, tp = confusion_matrix(T,P)

import numpy as np
from sklearn.metrics import confusion_matrix
def IoU(result,GT):
    Y=np.reshape(GT,(result.shape[0]*result.shape[2]*result.shape[2],1))
    Y=Y.astype(int)
    P=np.reshape(result,(result.shape[0]*result.shape[2]*result.shape[2],1))
    P=P.astype(int)
    tn, fp, fn, tp=confusion_matrix(Y, P,labels=[0,1]).ravel()
    iou=tp/(tp+fn+fp)  
    return   print( 'IoU is:   ',iou)