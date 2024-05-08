from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import keras
import os
import math
import numpy as np

class MaskedImageDataGenerator(keras.utils.Sequence):
  def __init__(self, directory, mask_denom=5, target_size=(256, 256), batch_size=32):
    """
    Initializes the data generator.

    Args:
      directory: Path to the directory containing images.
      mask_center: Tuple of floats (x, y) representing the center of the mask relative to the image (0.5, 0.5 is the center).
      mask_size: Tuple of ints (height, width) representing the size of the square mask.
      target_size: Tuple of ints (height, width) representing the desired output size of the images.
      batch_size: Integer representing the number of images per batch.
    """
    self.directory = directory
    self.mask_denom = mask_denom
    self.target_size = target_size
    self.batch_size = batch_size
    self.filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

  def __len__(self):
    """
    Returns the number of batches in the dataset.
    """
    return int(math.ceil(len(self.filenames) / self.batch_size))


  def __getitem__(self, idx):
    """
    Generates a batch of masked images and their originals as x and y

    Args:
      idx: Integer representing the batch index.

    Returns:
      A tuple containing a numpy array of masked images and a numpy array of original images.
    """
    
    batch_filenames = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
    y = []
    for filename in batch_filenames:
      # Load image, resize, and convert to array
      img = load_img(os.path.join(self.directory, filename), target_size=self.target_size)
      image = img_to_array(img)
      y.append(image)
    y = np.array(y, dtype=np.float32)
    x = np.copy(y)
    h,w = self.target_size
    sec_w = w//self.mask_denom
    # Make middle black
    offset = (self.mask_denom // 2)
    x[:,:,sec_w * offset:sec_w * (offset + 1),:]=0
    y=y/255
    x = x/255
    return x,y