from net.autoencoder import Autoencoder, custom_loss
from src.data_generator import MaskedImageDataGenerator
import tensorflow as tf
import keras
import argparse
import cv2
import os
import sys
import numpy as np

isColab = "google.colab" in sys.modules
data_dir = 'collapsed_data'
results_dir = 'results'
# this also works:
# isColab = "COLAB_GPU" in os.environ

if isColab:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)

    data_dir = ("/content/drive/MyDrive/collapsed_data")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, help = "file of model to visualize", default= "model.keras", required=True)
    return parser.parse_args()

def get_masked_data(dir, mask_denom=5,target_size=(256,256)):
    # dir = os.path.join(data_dir, sub_dir)
    files = os.listdir(dir)
    x = []
    y = []
    for f in files:
        path = os.path.join(dir, f)
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, target_size)
            y.append(img)
        except:
            print(path)
            continue
    y = np.array(y, dtype=np.float32)
    x = np.copy(y)
    h,w = target_size
    sec_w = w//mask_denom
    # Make middle black
    offset = (mask_denom // 2)
    x[:,:,sec_w * offset:sec_w * (offset + 1),:]=0
    y=y/255
    x = x/255
    return x,y

def write_reconstructions(model, x,y, name='recon'):
    pred = model.predict(x)
    n = x.shape[0]
    for i in range(n):
        path = results_dir + '/' + name + str(i) 
        cv2.imshow('test', y[i])
        cv2.waitKeyEx()
        cv2.imwrite(path + '-truth.jpeg', 255*y[i])
        cv2.imwrite(path + '-pred.jpeg', 255*pred[i])

def write_stitchings(model, x, name='stitch',size=(256,256), denom=5):
    h,w=size
    rec_w = w//denom
    n = x.shape[0]
    modified = []
    for i in range(n-1):
        img1 = x[i]
        img2 = x[i+1]
        new_img = img1.copy()
        new_img[:,rec_w*3:rec_w*5,:]=img2[:,rec_w*3:rec_w*5,:]
        modified.append(new_img)
    modified = np.array(modified)
    pred = model.predict(modified)
    for i in range(n-1):
        mod_img = modified[i]
        path = results_dir + '/' + name + str(i) 
        cv2.imwrite(path + '-mod.jpeg', 255*mod_img)
        cv2.imwrite(path + '-pred.jpeg', 255*pred[i])

if __name__ == "__main__":
    args = parse_arguments()
    model_path = 'models/' + args.name
    model = keras.models.load_model(model_path,custom_objects={'Autoencoder': Autoencoder, 'custom_loss': custom_loss})

    x_test,y_test = get_masked_data(data_dir + '/test')
    x=x_test[:5]
    y=y_test[:5]
    write_reconstructions(model,x,y)
    write_stitchings(model,x)
