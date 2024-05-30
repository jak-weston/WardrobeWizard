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