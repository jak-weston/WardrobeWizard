#%%
import dataloader
import torch
import torch.nn as nn
from config import config_gshape
from scipy import ndimage
import numpy as np


# Example usage (similar to OpenCV example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        pass

    def forward(self, noise, down_sampled, design_encoding):
        pass


# Load Data

segmented_images = dataloader.load_segmented_images()
text = dataloader.get_text_data()

# Bi-cubic down sample on segmented image
def bicubic_downsample(image, scale=[1,1, 1/16, 1/16]):
  """
  Downsamples an image using bicubic interpolation with SciPy.

  Args:
    image: NumPy array representing the image (HxWxC).
    scale: Downsampling factor (float).

  Returns:
    Downsampled image (HxW'xC') where W' = W // scale.
  """
  # Get image shape
  shape = image.shape
  # Resize the image using bicubic interpolation
  resized_image = ndimage.zoom(image, scale, mode='constant', order=3)

  return resized_image
# Downsampled images
down_sampled_images = bicubic_downsample(segmented_images)

print(down_sampled_images.shape)
# Train Loop

#%%