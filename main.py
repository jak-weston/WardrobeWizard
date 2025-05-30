#%%
## File that accepts sentence, image as input and changes it's style

import torch
import torch.nn as nn
import numpy as np
import h5py
import dataloader 
import matplotlib.pyplot as plt

def plot_image(image):
    plt.imshow(image)
    # get rid of the axis
    plt.axis('off')
    # plt.colorbar()
    plt.show()
#%%
load_number = 10000
# sentence = 12400 + 1199
sentence_num = 16000
G2 = h5py.File('./data/G2.h5', 'r')
image = (torch.tensor(np.array(G2['ih'][load_number])) + dataloader.get_image_mean()).permute(2, 1, 0)
sentence = dataloader.get_raw_text_data()[sentence_num][0]
segmented_image = torch.tensor(np.array(G2['b_'][load_number])).permute(2, 1, 0)
embedded_sentence = dataloader.get_text_data()[sentence_num]
embedded_sentence = embedded_sentence.unsqueeze(0).unsqueeze(2).unsqueeze(3).float()

print(sentence)
print(embedded_sentence.shape)
print(segmented_image.shape )
plot_image(segmented_image)
plot_image(image)
# %%

#Preprocess given segmentation
condition = torch.where(segmented_image >= 3 , 3, segmented_image).float() # 1 x 128 x 128
condition = nn.functional.one_hot(condition.to(torch.int64), num_classes=4).float().squeeze(2) # 4 x 128 x 128
print(condition.shape)
plot_image(condition.argmax(-1)) # Before interpolation
condition = nn.functional.interpolate(condition.permute(2,0,1).unsqueeze(0), size=(8, 8), mode='bicubic').squeeze(0).permute(1,2,0) # 8 x 8 x 4
plot_image(condition.argmax(-1)) # After interpolation
condition = condition.permute(2,1,0).unsqueeze(0) # 1 x 4 x 8 x 8
#%%

#Generate segmentation

from G_shape_model import Generator

device = "cpu"

netG = Generator().to(device)

netG.load_state_dict(torch.load('fashionGAN_results/G_shape_results_0.9/netG_81500.pth', map_location=device))

noise = torch.randn(1, 80, 1, 1).to(device)
embedded_sentence = embedded_sentence.to(device)
condition = condition.to(device)

fake_seg = netG(noise, embedded_sentence, condition)



fake_segment = fake_seg[0].argmax(dim=0).permute(1,0)
plot_image(fake_segment.detach().cpu().numpy())


#%%

# Generate image

from G_image_model import Generator

seg = fake_segment.unsqueeze(0).unsqueeze(1).float()
seg = nn.functional.one_hot(seg.to(torch.int64), num_classes=7).float().squeeze(0).permute(0,3,2,1)
print(seg.shape)
plot_image(seg.argmax(1)[0].detach().cpu().numpy())

netG = Generator().to(device)

netG.load_state_dict(torch.load('fashionGAN_results/G_image_results_0.8/netG_98500.pth', map_location=device))

noise = torch.randn(1, 80, 1, 1).to(device)

fake_image = netG(noise, embedded_sentence, seg)

print(fake_image.shape)
mean = dataloader.get_image_mean()

print(mean.shape)

fake_image = fake_image[0].cpu() + mean

hair_label = 1
face_label = 2

print(fake_segment.shape)
fake_segment = fake_segment.permute(1,0).cpu()
fake_image = fake_image.cpu()
image = image.permute(2,1,0).cpu()
segmented_image = segmented_image.squeeze(2).cpu().T
print(segmented_image.shape)
print(fake_image.shape)

hair_mask =  (segmented_image == hair_label) #| (fake_segment == hair_label) 
face_mask =  (segmented_image == face_label) #fake_segment == face_label) |
#Remove face and hair, and replace with original image
fake_image[0][hair_mask] = image[0][hair_mask]
fake_image[1][hair_mask] = image[1][hair_mask]
fake_image[2][hair_mask] = image[2][hair_mask]

fake_image[0][face_mask] = image[0][face_mask]
fake_image[1][face_mask] = image[1][face_mask]
fake_image[2][face_mask] = image[2][face_mask]


print(fake_image.shape)
plot_image(fake_image.detach().numpy().transpose(2,1,0))

#%%