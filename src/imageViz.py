import matplotlib.pyplot as plt
from PIL import Image   ## Python Image Library. Very Useful
import io
import numpy as np
import tensorflow as tf

class ImageVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_inputs, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.sample_inputs = sample_inputs
        self.imgs = []

    def on_epoch_end(self, epoch, logs=None):
        encoding   = self.model.encoder(self.sample_inputs)
        prediction = self.model.decoder(encoding)  ## = self.model(self.sample_inputs)
        enc_pic = tf.image.resize(encoding, [32, 32], method=tf.image.ResizeMethod.MITCHELLCUBIC)

        self.add_to_imgs(
            tf.concat([
                self.sample_inputs,
                enc_pic[:,:,:,:1],
                prediction
            ], axis=0),
            epoch = epoch
        )

    def add_to_imgs(self, outputs, epoch, nrows=3, ncols=8, figsize=(18, 8)):
        '''
        Plot the image samples in outputs in a pyplot figure and add the image
        to the 'imgs' list. Used to later generate a gif.
        '''
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs[0][0].set_title(f'Epoch {epoch+1}')
        axs[0][3].set_title('Inputs')
        axs[1][3].set_title('Encoding')
        axs[1][4].set_title('(Channel 0, upscaled)')
        axs[2][3].set_title('Decoding')
        for i, ax in enumerate(axs.reshape(-1)):
            out_numpy = np.squeeze(outputs[i].numpy(), -1)
            ax.imshow(out_numpy, cmap='gray')
        self.imgs += [self.fig2img(fig)]
        plt.close(fig)

    @staticmethod
    def fig2img(fig):
        """
        Convert a Matplotlib figure to a PIL Image and return it
        https://stackoverflow.com/a/61754995/5003309
        """
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return Image.open(buf)

    def save_gif(self, filename='mnist_recon', loop=True, duration=500):
        imgs = self.imgs
        self.imgs[0].save(
            filename+'.gif', save_all=True, append_images=self.imgs[1:],
            loop=loop, duration=duration)