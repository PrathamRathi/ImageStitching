from net.vae import VAE, fit
from src.data_generator import MaskedImageDataGenerator
import tensorflow as tf
import keras
import argparse
import json

TRAIN_DIR = 'collapsed_data/train'
MODEL_DIR = 'models/'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=10, help = "epochs")
    parser.add_argument("-name", type=str, help = "name of final model", default= "model.keras", required=True)
    parser.add_argument("-size", type=int, help = "size of input images", default= 256)
    parser.add_argument("-batch", type=int, help="batch size", default=100)   
    return parser.parse_args()

def get_model_name(args):
    return args.name + '-e' + str(args.epochs) + '-size' + str(args.size)

if __name__ == "__main__":
    args = parse_arguments()
    input_shape = (1,args.size,args.size,3)
    model = VAE(.001,img_size=(256,256), latent_size=256, hidden_dim=512)
    model.build(input_shape = input_shape)   ## Required to see architecture summary
    model.compile(
        optimizer   = model.optimizer,
    )
    
    print('-------------------------------- Model Summaries --------------------------------')
    model.encoder.summary()
    model.decoder.summary()
    model.summary()

    print('-------------------------------- Training Model --------------------------------')
    devices = tf.config.list_physical_devices()
    device = '/device:CPU:0'
    for device in devices:
        if 'GPU' in device.name:
            device =  '/device:GPU:0'
    with tf.device(device):
        fit(model, args.epochs, TRAIN_DIR, batch_size=args.batch)

    print('-------------------------------- Saving Model --------------------------------')
    model_name = args.name
    model.save(MODEL_DIR + model_name)