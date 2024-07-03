#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the trimap image
trimap_path = '../../../../Downloads/the-oxfordiiit-pet-dataset/annotations/annotations/trimaps/pug_2.png'
trimap_image = Image.open(trimap_path)

# Convert the trimap image to a NumPy array
trimap_array = np.array(trimap_image)

# Define the color map
colors = ['indigo', 'teal', 'yellow']  # Assuming 0 is not used in your trimap
cmap = ListedColormap(colors)

# Normalize the values to fit the colormap
normed_trimap = trimap_array - 1  # Shift values to start at 0

# Save the colorized trimap image
colorized_image_path = 'colorized_trimap.png'
plt.imsave(colorized_image_path, normed_trimap, cmap=cmap)

# Display the colorized trimap image
plt.imshow(normed_trimap, cmap=cmap)
plt.title('Colorized Trimap Image')
plt.colorbar(ticks=[0, 1, 2], format=plt.FuncFormatter(lambda val, loc: ['1', '2', '3'][int(val)]))
plt.show()
# %%
import os
import numpy as np
from PIL import Image

# Define the paths
trimaps_dir = '../../../../Downloads/the-oxfordiiit-pet-dataset/annotations/annotations/trimaps'
segmentation_dir = '../../../../Downloads/the-oxfordiiit-pet-dataset/annotations/annotations/segmentation'

# Create the segmentation directory if it does not exist
os.makedirs(segmentation_dir, exist_ok=True)

# Iterate through all files in the trimaps directory
for filename in os.listdir(trimaps_dir):
    if filename.endswith('.png') and not filename.startswith('._'):  # Process all valid PNG files, skip `._` files
        # Construct the full path to the trimap image
        trimap_path = os.path.join(trimaps_dir, filename)
        print(f"Processing file: {trimap_path}")

        # Load the trimap image
        trimap_image = Image.open(trimap_path)
        print(f"Successfully opened image: {trimap_path}")

        # Convert the trimap image to a NumPy array
        trimap_array = np.array(trimap_image)

        # Construct the full path to save the NumPy array
        save_path = os.path.join(segmentation_dir, filename.replace('.png', '.npy'))

        # Save the NumPy array to the segmentation directory
        np.save(save_path, trimap_array)
        print(f"Processed and saved: {save_path}")

print("All images have been processed and saved.")