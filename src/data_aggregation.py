import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import joblib
import glob
import random

'''
paths = sorted(os.listdir("collapsed_data/train"), key=lambda x: random.random())
assert (len(paths) == 10000)

for i in range(10000):
    try:
        img = cv2.imread("collapsed_data/train/" + paths[i])
        img = cv2.resize(img, (125, 125))
        img = img / 255.0
        np.save("arr_data_batchsize_1/arr_" + str(i), img)
    except:
        print(paths[i])

for b in range(200):
    batch = np.zeros((50, 125, 125, 3))
    for i in range(b * 50, (b + 1) * 50):
        local_i = i - (b * 50)
        batch[local_i, :, :, :] = np.load("arr_data_batchsize_1/arr_" + str(i) + ".npy")
    np.save("arr_data_batchsize_50/arr_" + str(b), batch)
'''