from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
P=np.array([[0,1],[0,1]])
T=np.array([[0,1],[0,1]])
tn, fp, fn, tp = confusion_matrix(T,P)
