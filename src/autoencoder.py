import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv2DTranspose
from imageViz import ImageVisualizer
from preprocess import get_data
import cv2 as cv

class Autoencoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = keras.Sequential([
             Conv2D(16, 8, 4, **conv_kwargs),
            Conv2D(16, 8, 4, **conv_kwargs),
            keras.layers.MaxPooling2D(),
            Conv2D(8, 4, 2, **conv_kwargs),
            Conv2D(8, 4, 2, **conv_kwargs)
        ], name="ae_encoder")

        self.decoder = keras.Sequential([
             Conv2DTranspose(8, 4, 2, **conv_kwargs),
        Conv2DTranspose(8, 4, 2, **conv_kwargs),
        Conv2D(16, 8, 4, **conv_kwargs),
        Conv2DTranspose(3, 8, 4, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=.1))
    ], name='ae_decoder')

    def call(self, inputs):
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
        return inputs


def mse_bce_loss(*args, **kwargs):
    mse_loss = keras.losses.MeanSquaredError()     ## HINTS
    bce_loss = keras.losses.BinaryCrossentropy()
    mse = mse_loss(*args, **kwargs)
    bce = bce_loss(*args, **kwargs)
    return .8 * mse + .2 * bce

conv_kwargs = {
    "padding"             : "SAME",
    "activation"          : keras.layers.LeakyReLU(alpha=0.2),
    "kernel_initializer"  : tf.random_normal_initializer(stddev=.1)
}

ae_model = Autoencoder(name='autoencoder')

ae_model.build(input_shape = (1,256,256,3))   ## Required to see architecture summary
initial_weights = ae_model.get_weights() ## Just so we can reset out autoencoder

ae_model.summary()
ae_model.encoder.summary()
ae_model.decoder.summary()

ae_model.compile(
    optimizer   = keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss        = mse_bce_loss,
    metrics     = [
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.BinaryCrossentropy()
    ]
)

x_train,y_train = get_data('train')
x_valid,y_valid = get_data('validation')

# Train the model
print('Fitting model on training data')
ae_model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_valid,y_valid))
print('------------------------')

print('Evaluating model on testing data')
x_test,y_test = get_data('test')
ae_model.evaluate(x_test, y_test, batch_size=32)

x_show = x_test[:-10]
y_show = y_test[:-10]

for i in range(10):
    x = x_show[i]
    pred = ae_model.predict(x)
    pred = pred.numpy()
    x *= 255
    pred *= 255
    cv.imshow()
    cv.imshow('Image', pred)

    
