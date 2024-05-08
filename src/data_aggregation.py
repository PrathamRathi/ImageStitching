import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random

'''
train_paths = sorted(os.listdir("collapsed_data/train"), key=lambda x: random.random())
assert (len(train_paths) == 10000)
validation_paths = sorted(os.listdir("collapsed_data/validation"), key=lambda x: random.random())
assert (len(validation_paths) == 1500)
test_paths = sorted(os.listdir("collapsed_data/test"), key=lambda x: random.random())
assert (len(test_paths) == 500)

# training data preprocessing

for i in range(10000):
    try:
        img = cv2.imread("collapsed_data/train/" + train_paths[i])
        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        np.save("train_arr_data_batchsize_1/arr_" + str(i), img)
    except:
        print(train_paths[i])

for b in range(200):
    batch = np.zeros((50, 150, 150, 3))
    for i in range(b * 50, (b + 1) * 50):
        local_i = i - (b * 50)
        batch[local_i, :, :, :] = np.load("train_arr_data_batchsize_1/arr_" + str(i) + ".npy")
    np.save("train_arr_data_batchsize_50/arr_" + str(b), batch)

# validation data preprocessing

for i in range(1500):
    try:
        img = cv2.imread("collapsed_data/validation/" + validation_paths[i])
        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        np.save("validation_arr_data_batchsize_1/arr_" + str(i), img)
    except:
        print(validation_paths[i])

for b in range(30):
    batch = np.zeros((50, 150, 150, 3))
    for i in range(b * 50, (b + 1) * 50):
        local_i = i - (b * 50)
        batch[local_i, :, :, :] = np.load("validation_arr_data_batchsize_1/arr_" + str(i) + ".npy")
    np.save("validation_arr_data_batchsize_50/arr_" + str(b), batch)

# testing data preprocessing

for i in range(500):
    try:
        img = cv2.imread("collapsed_data/test/" + test_paths[i])
        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        np.save("test_arr_data_batchsize_1/arr_" + str(i), img)
    except:
        print(test_paths[i])

for b in range(10):
    batch = np.zeros((50, 150, 150, 3))
    for i in range(b * 50, (b + 1) * 50):
        local_i = i - (b * 50)
        batch[local_i, :, :, :] = np.load("test_arr_data_batchsize_1/arr_" + str(i) + ".npy")
    np.save("test_arr_data_batchsize_50/arr_" + str(b), batch)
'''
