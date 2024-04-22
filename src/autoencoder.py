import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose

class Autoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
        return inputs

## Some common keyword arguments you way want to use. HINT: func(**kwargs)
conv_kwargs = {
    "padding"             : "SAME",
    "activation"          : keras.layers.LeakyReLU(alpha=0.2),
    "kernel_initializer"  : tf.random_normal_initializer(stddev=.1)
}

## TODO: Make encoder and decoder sub-models
ae_model = Autoencoder(
    encoder = tf.keras.Sequential([
        ## TODO: Implement encoder
        Conv2D(8, 8, 4, **conv_kwargs),
        Conv2D(8, 8, 2, **conv_kwargs)
    ], name="ae_encoder"),
    decoder = tf.keras.Sequential([
        Conv2DTranspose(8, 8, 2, **conv_kwargs),
        Conv2DTranspose(1, 8, 4, **conv_kwargs)
    ], name='ae_decoder')
, name='autoencoder')

ae_model.build(input_shape = X0.shape)   ## Required to see architecture summary
initial_weights = ae_model.get_weights() ## Just so we can reset out autoencoder

ae_model.summary()
ae_model.encoder.summary()
ae_model.decoder.summary()