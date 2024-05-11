import tensorflow as tf
import numpy as np
import cv2 as cv
import os


data_dir = './archive'

def get_data(sub_dir, size=(256,256)):
    # dir = os.path.join(data_dir, sub_dir)
    files = os.listdir(sub_dir)
    count = 0
    for f in files:
        path = os.path.join(sub_dir, f)
        try:
            img = cv.imread(path)
            img = cv.resize(img, size)
            cv.imwrite('archive-resize/new' + str(count) + '.jpeg', img)
            count +=1
        except Exception as e:
            print(e)
            continue
    # y = np.array(y, dtype=np.float32)
    # x = np.copy(y)
    # h,w = size
    # rec_w = w//5
    # # Make middle black
    # x[:,:,rec_w*2:rec_w*3,:]=0
    # y=y/255
    # x = x/255
    # print('done')
    # return x,y

# img = cv.imread('collapsed_data/train/Coast-Train (22).jpeg')
# img = cv.resize(img, (256,256))
# cv.imwrite('result.jpeg', img)

get_data(data_dir)


