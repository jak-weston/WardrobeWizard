#%%
import h5py
import scipy.io
import dataloader
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G2 = h5py.File('./data/G2.h5', 'r')

#%%
import matplotlib.pyplot as plt

print(G2.keys())
print(G2['ih'][0])
img = G2['ih'][0] + G2['ih_mean']
plt.imshow(img.T, cmap='jet')
#%%
text = dataloader.get_text_data()

segmented_images = dataloader.load_segmented_images()
#%%

# Plot first segmented image with legend
import matplotlib.pyplot as plt
import numpy as np
plt.imshow(segmented_images[0][0].T, cmap='jet')
plt.colorbar()
plt.show()
#%%
segment_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 4,
    4: 4,
    5: 4,
    6: 4
}# %%

# apply the segment map to the first segmented image
down_sampled = torch.tensor([segment_map[x] for x in segmented_images[0].flatten()]).reshape(segmented_images[0].shape)
plt.imshow(down_sampled[0].T, cmap='jet')
plt.colorbar()
plt.show()

# %%
G2 = h5py.File('./data/G2.h5', 'r')

#%%
#PET DATASET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
meta = pd.read_csv('./data/list.txt', sep=' ')

meta.head()

pet_class = meta['class'].values
#one hot encode pet_class
pet_class = pd.get_dummies(pet_class)
#convert to numpy array
pet_class = pet_class.values
print(pet_class)
#%%
image_path = './data/images'
segmentation_path = './data/segmentation'
image_0 = meta['image'][0]

image = Image.open(f'{image_path}/{image_0}.jpg')
segmentation = np.load(f'{segmentation_path}/{image_0}.npy')

def plot_image(image, segmentation):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(segmentation)
    plt.show()

plot_image(image, segmentation)
# %%

image_paths = [f'{image_path}/{x}.jpg' for x in meta['image']]
segmentation_paths = [f'{segmentation_path}/{x}.npy' for x in meta['image']]


images = []
segmentations = []
for i, img_path in enumerate(image_paths):
    img = Image.open(img_path)
    seg = np.load(segmentation_paths[i])
    #Resize segmentation to 128x128
    seg = Image.fromarray(seg)
    seg = seg.resize((128, 128))
    img = img.resize((128, 128))
    #Convert to numpy array
    seg = np.array(seg)
    img = np.array(img)
    images.append(img)
    segmentations.append(seg)


plot_image(images[0], segmentations[0])
#%%

for i, img in enumerate(images):
    if img.shape != (128, 128, 3):
        #Get rid of 4th channel
        img = img[:,:,:3]
        images[i] = img

for img in images:
    if img.shape != (128, 128, 3):
        print(img.shape)


img_np = np.array(images)
#%%
seg_np = np.array(segmentations)
#%%
print(img_np.shape)
print(seg_np.shape)
#%%
np.save('./data/images.npy', img_np)
np.save('./data/segmentations.npy', seg_np)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_np = np.load('./data/images.npy')
seg_np = np.load('./data/segmentations.npy')
def plot_image(image, segmentation):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(segmentation)
    plt.show()

plot_image(img_np[0], seg_np[0])
plot_image(img_np[1], seg_np[1])
#Convert images to black and white
img_bw = np.mean(img_np, axis=3)

print(img_bw.shape)
print(seg_np.shape)
#only keep where segmentation is 1
img_bw = np.where(seg_np != 2 , img_bw, 0)
seg_np = np.where(seg_np != 2, 0, 1)

#Fill in the holes
from scipy.ndimage import binary_fill_holes
for i in range(len(seg_np)):
    seg_np[i] = binary_fill_holes(seg_np[i])
print(img_bw.shape)
print(seg_np.shape)

plot_image(img_bw[0], seg_np[0])
plot_image(img_bw[1], seg_np[1])
#%%
