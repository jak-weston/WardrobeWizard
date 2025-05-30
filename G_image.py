#%%
print("Starting imports")

import torch
import torch.optim as optim
import torch.nn as nn
from config import config
from dataloader import get_text_data, load_segmented_images, load_full_images, get_image_mean
from G_image_model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Done with imports")
# Load data 
input_data = get_text_data().to(device)
input_segments = load_segmented_images().to(device)
input_images = load_full_images().to(device)

image_mean = get_image_mean()

print("Done with loading Data")
l = 100
lambda_real = config['lambda_real']
lambda_fake = config['lambda_fake']
lambda_mismatch = config['lambda_mismatch']

#check results directory exists
import os
if not os.path.exists(f'G_image_results_{lambda_fake}_real_{lambda_real}'):
    os.makedirs(f'G_image_results_{lambda_fake}_real_{lambda_real}')

# Initialize models
netG = Generator().to(device)
netD = Discriminator().to(device)

# Loss and optimizer
criterion = nn.BCELoss()

optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

nz = config['nz']
batchsize = config['batchSize']


#%%
print("Starting Training Loop...")
num_epochs = 10

img_list = []
G_losses = []
D_losses = []

iters = 0

for epoch in range(num_epochs):
    for i in range(0, len(input_data), batchsize):
        # Prepare batch data
        batch_encode = input_data[i:i + batchsize].float()
        batch_data = input_images[i:i + batchsize].float()
        batch_condition = input_segments[i:i + batchsize].float()
        batch_encode = batch_encode.unsqueeze(2).unsqueeze(3)
        if batch_data.shape[0] != batchsize:
            break

        #one hot encode condition
        batch_condition = nn.functional.one_hot(batch_condition.to(torch.int64), num_classes=7).float().squeeze(1).permute(0, 3, 1, 2)

        # Create labels
        real_label = torch.ones(len(batch_data), 1, 1, 1).to(device)
        fake_label = torch.zeros(len(batch_data), 1, 1, 1).to(device)

        ############################
        # (1) Update D network
        ###########################
        netD.zero_grad()

        # Train with real
        output = netD(batch_data, batch_encode, batch_condition)
        errD_real = criterion(output, real_label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake
        noise = torch.randn(batchsize, nz, 1, 1).to(device)
        fake = netG(noise, batch_encode, batch_condition)
        #Add noise to fake
        # fake = fake + torch.randn_like(fake) * 0.1
        output = netD(fake.detach(), batch_encode, batch_condition)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()
        optimizerD.step()
        errD = errD_real + errD_fake
        D_G_z1 = output.mean().item()

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        output = netD(fake,  batch_encode, batch_condition)
        errG = criterion(output, real_label)
        errG.backward()
        optimizerG.step()
        D_G_z2 = output.mean().item()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(input_data),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(input_data)-1)):
            with torch.no_grad():
                img_list.append(fake)
                torch.save(netG.state_dict(), f'G_image_results_{lambda_fake}_real_{lambda_real}/netG_{iters}.pth')
                torch.save(netD.state_dict(), f'G_image_results_{lambda_fake}_real_{lambda_real}/netD.pth')
                torch.save({
                    'G_losses': G_losses,
                    'D_losses': D_losses,
                    'img_list': img_list
                }, f'G_image_results_{lambda_fake}_real_{lambda_real}/training_data.pth')
            
        iters += 1

#%%