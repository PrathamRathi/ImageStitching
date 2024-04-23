import tensorflow as tf
import numpy as np
import cv2 as cv
import os

data_dir = './collapsed_data'

def get_data(sub_dir, size=(256,256)):
    dir = os.path.join(data_dir, sub_dir)
    files = os.listdir(dir)
    x = []
    y = []
    for f in files:
        path = os.path.join(dir, f)
        try:
            img = cv.imread(path)
            img = cv.resize(img, size)
            y.append(img)
        except:
            print(path)
            continue
    y = np.array(y, dtype=np.float32)
    x = np.copy(y)
    h,w = size
    rec_w = w//5
    # Make middle black
    x[:,:,rec_w*2:rec_w*3,:]=0
    y=y/255
    x = x/255
    print('done')
    return x,y


