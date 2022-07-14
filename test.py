# from tabnanny import check
# import torch
# from keras import backend as K 
# from keras.layers import multiply
# import tensorflow
# import sys
# import torch.utils.data as Data
# import keras.losses 
# import tensorflow as tf
import numpy as np

# x = K.constant([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]])

# c = K.gather(x,[0,1,2])
# c_val = K.eval(c)
# print("ok")


batch = []

for b in range(int(128 / 128)):
    batch.append([i+(b*128) for i in range(128)])

print("OK")