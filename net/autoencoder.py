import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv2DTranspose
import os
import cv2
import numpy as np

conv_kwargs = {
    "padding"             : "SAME",
    "activation"          : keras.layers.LeakyReLU(alpha=0.2),
    "kernel_initializer"  : tf.random_normal_initializer(stddev=.1)
}
class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = keras.Sequential([
            Conv2D(16, 8, 2, **conv_kwargs),
            Conv2D(16, 8, 1, **conv_kwargs),
            keras.layers.MaxPooling2D(),
            Conv2D(16, 4, 1, **conv_kwargs),
        ], name="ae_encoder")

        self.decoder = keras.Sequential([
            Conv2DTranspose(32, 4, 1, **conv_kwargs),
            Conv2DTranspose(16, 8, 2, **conv_kwargs),
            Conv2DTranspose(3, 8, 2, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=.1), activation='sigmoid')
        ], name='ae_decoder')

    def call(self, inputs):
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
        return inputs


def custom_loss(y_true, y_pred):
    mse_loss = keras.losses.MeanSquaredError()
    mae_loss = keras.losses.MeanAbsoluteError()
    bce_loss = keras.losses.BinaryCrossentropy()
    mse = mse_loss(y_true, y_pred)
    mae = mae_loss(y_true, y_pred)
    bce = bce_loss(y_true, y_pred)
    loss =  .3*mse + .7*mae
    #loss = mae
    return loss

# class Autoencoder(tf.keras.Model):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = keras.Sequential([
#             Conv2D(16, 8, 2, **conv_kwargs),
#             Conv2D(16, 8, 2, **conv_kwargs),
#             keras.layers.MaxPooling2D(),
#             Conv2D(64, 4, 1, **conv_kwargs),
#         ], name="ae_encoder")

#         self.decoder = keras.Sequential([
#         Conv2DTranspose(64, 4, 1, **conv_kwargs),
#         Conv2DTranspose(16, 8, 2, **conv_kwargs),
#         Conv2DTranspose(3, 8, 4, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=.1))
#     ], name='ae_decoder')

#     def call(self, inputs):
#         inputs = self.encoder(inputs)
#         inputs = self.decoder(inputs)
#         return inputs
