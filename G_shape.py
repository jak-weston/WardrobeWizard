#%%
import dataloader
import torch
import torch.nn as nn
from config import config
from scipy import ndimage
import numpy as np

# Example usage (similar to OpenCV example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nt = config['nt']
nz = config['nz']
nt_input = config['nt_input']

nc = config['n_map_all']
ncondition = config['n_condition']

win_size = config['win_size']
lr_win_size = config['lr_win_size']

ngf = 64
ndf = 64


segmented_images = dataloader.load_segmented_images().to(device)
text = dataloader.get_text_data().to(device)

# Train .lua Notes: noise_vis = 80 dim noise vector, encode_vis = 100 dim text encoding, condition_vis = downsampled segmented image

# Net_graph_sr1 Notes: input_data = noise_vis, input_encode = encode_vis, input_condition = condition_vis
#%%
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode_conv1 = nn.Conv2d(nt_input, nt, 1)
        self.lerelu = nn.LeakyReLU(0.2, inplace=True)

        self.g1_deconv1 = nn.ConvTranspose2d(nz + nt, ngf * 16, 4)
        self.g1_bn1 = nn.BatchNorm2d(ngf * 16)
        self.relu = nn.ReLU(True)
        
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

        return self.softmax(g5)

class Descriminator(nn.Module):
    def __init__(self):
        super(Descriminator, self).__init__()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

        self.output_data_conv1 = nn.Conv2d(nc, ndf, 4, 2, 1)


    #output data = 
    def forward(self, output_data, output_encode, output_condition):
        pass

        


# Load Data as torch tensors
#%%
batch_size = config['batchSize']
for i in range(0, segmented_images.shape[0], batch_size):
    segmented_images_batch = segmented_images[i:i+batch_size]
    text_batch = text[i:i+batch_size]
    condition = torch.where(segmented_images_batch >= 3 , 3, segmented_images_batch).float() # batch_size x 1 x 128 x 128
    condition = nn.functional.one_hot(condition.to(torch.int64), num_classes=4).permute(0,4,1,2,3).float().squeeze(2) # batch_size x 4 x 128 x 128
    condition = nn.functional.interpolate(condition, size=(lr_win_size, lr_win_size), mode='bicubic') # batch_size x 4 x 8 x 8

    noise = torch.randn(batch_size, nz, 1, 1).to(device)
    target = nn.functional.one_hot(segmented_images_batch.to(torch.int64), num_classes=nc).permute(0,4,1,2,3).float().squeeze(2) # batch_size x 7 x 128 x 128
    text_batch = text_batch.unsqueeze(2).unsqueeze(3).float()
    print("text shape:", text_batch.shape)
    print("condition shape:", condition.shape)
    print("noise shape:", noise.shape)
    print("target shape:", target.shape)

    #Sanity check model
    model = Generator().to(device)
    output = model(noise, text_batch, condition)
    print("output shape:", output.shape)
    break

#%%