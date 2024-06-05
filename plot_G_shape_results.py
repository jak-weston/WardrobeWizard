#%%
import torch
import os
# Read results

results = torch.load('G_shape_results_0.9_real_0.5/training_data.pth')

G_losses = results['G_losses']
D_losses = results['D_losses']
img_list = results['img_list']

#%%
import matplotlib.pyplot as plt
import numpy as np

# Plot the training losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
img_length = len(img_list)
plot_index = range(len(img_list))
print(plot_index)
# Plot the generated images
for i in plot_index:
    img = img_list[i]
    img = img.argmax(dim=1)
    plt.figure(figsize=(10,10))
    plt.axis("off")
    plt.title("Generated Images at iteration {}".format(i * 500))
    img = img_list[i]
    img = img.argmax(dim=1)
    plt.imshow(np.transpose(img[0]))
    plt.show()
# %%
print(len(img_list ))
# plot last couple of images
for i in range(5):
    img = img_list[-i]
    img = img.argmax(dim=1)
    for x in range(len(img)):
        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.title("Generated Images at iteration {}".format((i) * 500))
        plt.imshow(np.transpose(img[x]))
        plt.show()

#%%
import dataloader
seg = dataloader.load_segmented_images()
sentence = dataloader.get_text_data()
# Attempt with specific sentence
#%%

import torch.nn as nn
from config import config
nt = config['nt']
nz = config['nz']
nt_input = config['nt_input']

nc = config['n_map_all']
ncondition = config['n_condition']

win_size = config['win_size']
lr_win_size = config['lr_win_size']

ngf = 64
ndf = 64
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode_conv1 = nn.Conv2d(nt_input, nt, 1)
        self.lerelu = nn.LeakyReLU(0.2)

        self.g1_deconv1 = nn.ConvTranspose2d(nz + nt, ngf * 16, 4)
        self.g1_bn1 = nn.BatchNorm2d(ngf * 16)
        self.relu = nn.ReLU()
        
        self.g_extra_deconv1 = nn.ConvTranspose2d(ngf * 16, ngf * 8, 4,2,1)
        self.g_extra_bn1 = nn.BatchNorm2d(ngf * 8)

        self.condition_conv1 = nn.Conv2d(ncondition, ngf,3,1,1)
        self.condition_bn1 = nn.BatchNorm2d(ngf)

        self.f1_conv1 = nn.Conv2d(ngf, ngf*2, 3, 1, 1)
        self.f1_bn1 = nn.BatchNorm2d(ngf*2)

        self.g2_deconv1 = nn.ConvTranspose2d(ngf * 10, ngf * 4, 4, 2, 1)
        self.g2_bn1 = nn.BatchNorm2d(ngf * 4)

        self.g2_conv1 = nn.Conv2d(ngf * 4, ngf * 8, 3, 1, 1)
        self.g2_bn2 = nn.BatchNorm2d(ngf * 8)

        self.m1_conv1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.m1_bn1 = nn.BatchNorm2d(ngf * 8)

        self.m2_conv1 = nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1)
        self.m2_bn1 = nn.BatchNorm2d(ngf * 4)

        self.m3_deconv1 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1)
        self.m3_bn1 = nn.BatchNorm2d(ngf * 2)

        self.g3_deconv1 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1)
        self.g3_bn1 = nn.BatchNorm2d(ngf)

        self.g4_decov1 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1)

        self.softmax = nn.Softmax2d()


    def forward(self, input_data, input_encode, input_condition):
        #Handle the input_encode and input_data into g_extra
        input_encode = self.encode_conv1(input_encode)
        input_encode = self.lerelu(input_encode)
        input_data_encode = torch.cat((input_data, input_encode), 1)

        g1 = self.g1_deconv1(input_data_encode)
        g1 = self.g1_bn1(g1)
        g1 = self.relu(g1)
        
        g_extra = self.g_extra_deconv1(g1)
        g_extra = self.g_extra_bn1(g_extra)
        g_extra = self.relu(g_extra)

        #handle the input_condition into f_extra
        f1 = self.condition_conv1(input_condition)
        f1 = self.condition_bn1(f1)
        f1 = self.lerelu(f1)

        f1 = self.f1_conv1(f1)
        f1 = self.f1_bn1(f1)
        f_extra = self.lerelu(f1)

        g2 = torch.cat((g_extra, f_extra), 1)
        g2 = self.g2_deconv1(g2)
        g2 = self.g2_bn1(g2)
        g2 = self.relu(g2)

        g2 = self.g2_conv1(g2)
        g2 = self.g2_bn2(g2)
        m1 = self.lerelu(g2)

        m1 = self.m1_conv1(m1)
        m1 = self.m1_bn1(m1)
        m2 = self.lerelu(m1)

        m2 = self.m2_conv1(m2)
        m2 = self.m2_bn1(m2)
        m3 = self.lerelu(m2)

        m3 = self.m3_deconv1(m3)
        m3 = self.m3_bn1(m3)
        g3 = self.relu(m3)

        g3 = self.g3_deconv1(g3)
        g3 = self.g3_bn1(g3)
        g4 = self.relu(g3)

        g5 = self.g4_decov1(g4)
        g5 = self.softmax(g5)
        return g5


netG = Generator()
netG.load_state_dict(torch.load('G_shape_results/netG.pth'))
netG.eval()
noise = torch.randn(1, nz, 1, 1)

segmented_images_batch = seg[0].unsqueeze(0)
condition = torch.where(segmented_images_batch >= 3 , 3, segmented_images_batch).float() # batch_size x 1 x 128 x 128
condition = nn.functional.one_hot(condition.to(torch.int64), num_classes=4).permute(0,4,1,2,3).float().squeeze(2) # batch_size x 4 x 128 x 128
condition = nn.functional.interpolate(condition, size=(8, 8), mode='bicubic') # batch_size x 4 x 8 x 8

sen = sentence[10].unsqueeze(0).unsqueeze(2).unsqueeze(3).float()
print(noise.shape)
print(sen.shape)
print(condition.shape)

# Generate an image
fake = netG(noise, sen, condition)[0]
fake = fake.argmax(dim=0)

# Plot the generated image
plt.figure(figsize=(10,10))
plt.axis("off")
plt.title("Generated Image")
plt.imshow(np.transpose(fake.detach().numpy()))
plt.show()
#%%
