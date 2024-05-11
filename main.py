from net.autoencoder import Autoencoder, custom_loss
from src.data_generator import MaskedImageDataGenerator
import tensorflow as tf
import keras
import argparse
import json

TRAIN_DIR = 'collapsed_data/train'
VALIDATION_DIR = 'collapsed_data/validation'
TEST_DIR  = 'collapsed_data/test'
MODEL_DIR = 'models/'
HISTORY_DIR = 'models/history/'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=10, help = "epochs")
    parser.add_argument("-name", type=str, help = "name of final model", default= "model.keras", required=True)
    parser.add_argument("-size", type=int, help = "size of input images", default= 256)
    parser.add_argument("-model", type=str, help = "type of model to use (dense, conv, etc.)") 
    parser.add_argument("-batch", type=int, help="batch size", default=100)   
    return parser.parse_args()

def get_model_name(args):
    return args.name + '-e' + str(args.epochs) + '-size' + str(args.size)

if __name__ == "__main__":
    args = parse_arguments()
    input_shape = (1,args.size,args.size,3)
    model = Autoencoder()
    model.build(input_shape = input_shape)   ## Required to see architecture summary
    model.summary()
    model.compile(
        optimizer   = keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss        = custom_loss,
        metrics     = [
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanAbsoluteError()
            #tf.keras.metrics.BinaryCrossentropy(),
        ]
    )
    
    training_generator = MaskedImageDataGenerator(TRAIN_DIR, mask_denom=5, target_size=(args.size, args.size), batch_size=args.batch)
    validation_generator = MaskedImageDataGenerator(TEST_DIR, mask_denom=5, target_size=(args.size, args.size), batch_size=args.batch)

    devices = tf.config.list_physical_devices()
    device = '/device:CPU:0'
    for device in devices:
        if 'GPU' in device.name:
            device =  '/device:GPU:0'
    with tf.device(device):
        history = model.fit(training_generator, epochs=args.epochs, validation_data=validation_generator)

    print('Evaluating model')
    test_generator = MaskedImageDataGenerator(TEST_DIR, mask_denom=7, target_size=(args.size, args.size), batch_size=args.batch)
    model.evaluate(test_generator)

    print('----------------------------------------------------------------')

    print('Saving model')
    model_name = args.name #get_model_name(args)
    model.save(MODEL_DIR + model_name + '.keras')
    out_file = open(HISTORY_DIR + model_name + '.json', "w") 
    json.dump(history.history, out_file)
