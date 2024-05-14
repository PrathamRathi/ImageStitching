from net.autoencoder import Autoencoder, custom_loss
from net.vae import VAE
from src.data_generator import MaskedImageDataGenerator
import tensorflow as tf
import keras
import argparse
import cv2
import os
import numpy as np

data_dir = 'collapsed_data'
results_dir = 'results'
panorama_dir = 'panoramas'
stitch_dir = 'stitches'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, help = "file of model to visualize", default= "model.keras", required=True)
    parser.add_argument("-model", type=str, help = "type of model: ae, vae", default= "ae", )
    return parser.parse_args()

def get_masked_data(dir, denom=5,target_size=(256,256)):
    """
        Returns masked data from given directory
        dir: directory to get data from
        denom: denominator of masked portion
        target_size: size of images
    """
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
    sec_w = w//denom
    # Make middle black
    offset = (denom // 2)
    x[:,:,sec_w * offset:sec_w * (offset + 1),:]=0
    y=y/255
    x = x/255
    return x,y

def write_reconstructions(model, x,y, is_vae=False, name='recon'):
    """
        Given a model, write its reconstructions on data to directory
        x: masked images to be reconstructed
        y: unmasked, true images
        is_vae: specifies if the model is a vae or regular ae
        name: name of files
    """
    pred=None
    if is_vae:
        pred = model(x)[0]
        pred = pred.numpy()
    else:
        pred = model.predict(x)

    n = x.shape[0]
    for i in range(n):
        path = results_dir + '/' + name + str(i) 
        # cv2.imshow('test', y[i])
        # cv2.waitKeyEx()
        print(pred[i])
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

def make_panorama(model, file1, file2, name, denom=5, size=(256,256)):
    img1 = cv2.imread(file1)/255
    img2 = cv2.imread(file2)/255
    h,w = size
    sec_w = w//denom
    offset = (denom // 2)
    new_img = img1.copy()

    new_img[:,sec_w*0:sec_w*2,:]=img1[:,sec_w*3:sec_w*5,:]
    new_img[:,sec_w * offset:sec_w * (offset + 1),:]=0
    new_img[:,sec_w*3:sec_w*5,:]=img2[:,sec_w*0:sec_w*2,:]

    batched = np.expand_dims(new_img, axis=0)
    pred = model.predict(batched)[0]

    file_name = name + '.jpeg'
    cv2.imwrite(panorama_dir + '/pred' + file_name, pred*255)
    transition = pred[:,sec_w * offset:sec_w * (offset + 1),:]
    concat = np.hstack([img1, img2])
    cv2.imwrite(panorama_dir + '/concat-' + file_name, concat * 255)

    trans_width = transition.shape[1]
    start = (w*2 - trans_width)//2
    end = start + trans_width
    concat[:, start:end,:] = transition
    cv2.imwrite(panorama_dir + '/blend-' + file_name, concat * 255)
    panorama = np.hstack([np.hstack([img1, transition]), img2]) * 255
    cv2.imwrite(panorama_dir + '/pano-' + file_name, panorama)

def model_visual_test(model, is_vae=False):
    x_test,y_test = get_masked_data(data_dir + '/test', denom=5)
    x=x_test[:5]
    y=y_test[:5]
    write_reconstructions(model,x,y, is_vae=is_vae)
    if not is_vae:
        write_stitchings(model,x,denom=5)


def stitch_images(model, file1, file2, name, denom=5, size=(256,256)):
    img1 = cv2.imread(file1)/255
    img2 = cv2.imread(file2)/255
    h,w = size
    rec_w = w//denom
    offset = (denom // 2)
    new_img = img1.copy()
    new_img[:,rec_w*3:rec_w*5,:]=img2[:,rec_w*3:rec_w*5,:]
    new_img[:,rec_w * offset:rec_w * (offset + 1),:]=0

    file_name = name + '.jpeg'
    cv2.imwrite(stitch_dir + '/orig-' + file_name, 255*new_img)
    batched = np.expand_dims(new_img, axis=0)
    pred = model.predict(batched)[0]
    cv2.imwrite(stitch_dir + '/stitched-' + file_name, 255*pred)
    

if __name__ == "__main__":
    args = parse_arguments()
    model_path = 'models/' + args.name
    model = None
    if args.model == 'ae':
        model = keras.models.load_model(model_path,custom_objects={'custom_loss': custom_loss})
    if args.model == 'vae':
        model = keras.models.load_model(model_path)
    
    model_visual_test(model, is_vae=False)
    img1 = 'testing/Coast/Coast-Test (105).jpeg'
    img2 = 'testing/Coast/Coast-Test (106).jpeg'
    
    make_panorama(model,img1,img2, 'coast_test')
    stitch_images(model,img1,img2, 'coast_test')

