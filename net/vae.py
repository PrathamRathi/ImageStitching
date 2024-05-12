import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv2DTranspose
import numpy as np
import os
import math
from keras.preprocessing.image import load_img, img_to_array
import random

conv_kwargs = {
    "padding"             : "SAME",
    "activation"          : tf.keras.layers.LeakyReLU(alpha=0.2),
    "kernel_initializer"  : tf.random_normal_initializer(stddev=.1)
}
class VAE(tf.keras.Model):
    def __init__(self, learning_rate,img_size=(256,256),latent_size=256, hidden_dim=512):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim    
        self.img_size = img_size
        self.encoder = keras.Sequential([
           Conv2D(16, 8, 2, **conv_kwargs),
            Conv2D(16, 8, 1, **conv_kwargs),
            keras.layers.MaxPooling2D(),
            Conv2D(8, 4, 2, **conv_kwargs),
            Conv2D(4, 4, 2, **conv_kwargs),
            Conv2D(4, 4, 1, **conv_kwargs),
            keras.layers.Flatten(),
            keras.layers.Dense(512)
        ], name="ae_encoder")

        self.decoder = keras.Sequential([
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Reshape((16,16,2)),
            Conv2DTranspose(32, 4, 2, **conv_kwargs),
            Conv2DTranspose(16, 8, 2, **conv_kwargs),
            Conv2DTranspose(3, 8, 2, **conv_kwargs),
            Conv2DTranspose(3, 8, 2, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=.1), activation='sigmoid')
        ], name='ae_decoder')


        self.mu_layer = keras.layers.Dense(latent_size)
        self.logvar_layer = keras.layers.Dense(latent_size)
        self.optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon. loss')
        self.kld_loss_tracker = keras.metrics.Mean(name='kld loss')


    def get_latent_encoding(self, x):
        """
        Returns latent encoding of input
        Inputs:
        - x: a batch of input images
        Returns:
        - z: batch of latent encodings of input created by encoder and reparameterization
        """
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = self.reparametrize(mu, logvar)
        return z
    

    def call(self, x):
        """    
        Runs a forward pass of the entire vae
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)    
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z
    

    def predict(self, x):
        """
        Runs a forward pass on the data but only returns reconstructions
        Inputs:
        - x: Batch of input images of shape (N, 3, H, W)    
        Returns:
        - x_hat: Reconstruced input data of shape (N,3,H,W)
        """
        latent = self.encoder(x)
        mu = self.mu_layer(latent)
        logvar = self.logvar_layer(latent)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat
    

    def reparametrize(self, mu, logvar):
        """
        Differentiably sample random Gaussian data with specified mean and variance using the
        reparameterization trick.

        Inputs:
        - mu: Tensor of shape (N, Z) giving means
        - logvar: Tensor of shape (N, Z) giving log-variances

        Returns: 
        - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
            mean mu[i, j] and log-variance logvar[i, j].
        """
        std_dev = tf.math.sqrt(tf.math.exp(logvar))
        z = mu + tf.random.normal(shape=tf.shape(std_dev)) * std_dev
        return z


    def bce_function(self,x, x_hat):
        """
        Computes the reconstruction loss of the VAE.
        
        Inputs:
        - x_hat: Reconstructed input data of shape (N, 3, H, W)
        - x: Input data for this timestep of shape (N, 3, H, W)
        
        Returns:
        - reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
        """
        bce_fn = keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=keras.losses.Reduction.SUM,
        )
        reconstruction_loss = bce_fn(x, x_hat) * x.shape[
            -1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
        self.recon_loss_tracker.update_state(reconstruction_loss/x.shape[0])
        return reconstruction_loss

    def mse(self, x, x_hat):
        mse = keras.losses.MeanSquaredError()
        loss = mse(x, x_hat)
        self.recon_loss_tracker.update_state(loss)
        return loss

    def loss_function(self,x_hat, x, mu, logvar):
        """
        Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
        Returned loss is the average loss per sample in the current batch.

        Inputs:
        - x_hat: Reconstructed input data of shape (N, 3, H, W)
        - x: Input data for this timestep of shape (N, 3, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
        
        Returns:
        - loss: Tensor containing the scalar loss for the negative variational lowerbound
        """
        variance = tf.math.exp(logvar)
        kl_loss = -.5 * tf.math.reduce_sum((1 + logvar - tf.square(mu) - variance))
        self.kld_loss_tracker.update_state(kl_loss)
        #loss = self.bce_function(x, x_hat) + kl_loss
        loss = self.mse(x, x_hat) + kl_loss/x.shape[0]
        #loss /= x.shape[0]
        return loss
    
    def train_step(self, data):
        x = data[0]
        with tf.GradientTape() as tape:
            x_hat, mu, logvar, _ = self(x)
            loss = self.loss_function(x_hat, x, mu, logvar)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {'loss':self.loss_tracker.result(),
                'recon. loss':self.recon_loss_tracker.result(),
                'kl loss':self.kld_loss_tracker.result()
                }
    

def fit(model, epochs, train_dir, batch_size):
    for e in range(epochs):
        print(f'----------- Starting epoch {e} -----------')
        filenames = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
        random.shuffle(filenames)
        num_batches = int(math.ceil(len(filenames) / batch_size))
        losses = None
        for i in range(num_batches):
            batch_filenames = filenames[i * batch_size : (i + 1) * batch_size]
            x = []
            for filename in batch_filenames:
                # Load image, resize, and convert to array
                img = load_img(os.path.join(train_dir, filename), target_size=(256,256))
                image = img_to_array(img)
                x.append(image)
            x = np.array(x, dtype=np.float32)/255
            losses = model.train_step(x)
        
        print(f"Training losses at epoch {e}: {losses}")

