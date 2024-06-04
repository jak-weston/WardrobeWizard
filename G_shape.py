#%%
import dataloader
import torch
import torch.nn as nn
from config import config
from scipy import ndimage
import numpy as np
from G_shape_model import Generator, Descriminator

# Example usage (similar to OpenCV example)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "mps"

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
# Load Data as torch tensors
#%%
batch_size = config['batchSize']

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

netG = Generator().to(device)
netD = Descriminator().to(device)

optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(batch_size, nz, 1, 1).to(device)

img_list = []
G_losses = []
D_losses = []
D_guess_fake = []
D_guess_real = []

iters = 0

lambda_fake = config['lambda_fake']
lambda_mismatch = config['lambda_mismatch']
#check results directory exists
import os
if not os.path.exists(f'G_shape_results_{lambda_fake}'):
    os.makedirs(f'G_shape_results_{lambda_fake}')
print("Starting Training Loop...")

num_epochs = 5
for epoch in range(num_epochs):
  for i in range(0, segmented_images.shape[0], batch_size):
    segmented_images_batch = segmented_images[i:i+batch_size]
    text_batch = text[i:i+batch_size]
    if segmented_images_batch.shape[0] != batch_size:
        break
    condition = torch.where(segmented_images_batch >= 3 , 3, segmented_images_batch).float() # batch_size x 1 x 128 x 128
    condition = nn.functional.one_hot(condition.to(torch.int64), num_classes=4).permute(0,4,1,2,3).float().squeeze(2) # batch_size x 4 x 128 x 128
    condition = nn.functional.interpolate(condition, size=(lr_win_size, lr_win_size), mode='bicubic') # batch_size x 4 x 8 x 8

    target = nn.functional.one_hot(segmented_images_batch.to(torch.int64), num_classes=nc).permute(0,4,1,2,3).float().squeeze(2) # batch_size x 7 x 128 x 128
    text_batch = text_batch.unsqueeze(2).unsqueeze(3).float()

    # Train Descriminator

    #Train Real Images
    netD.zero_grad()

    output = netD(target, text_batch, condition)
    label = torch.full(output.shape, real_label, device=device).float()
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    #Train Fake Images
    
    noise = torch.randn(batch_size, nz, 1, 1).to(device)
    # Generate fake image
    fake = netG(noise, text_batch, condition)
    label.fill_(fake_label)
    output = netD(fake.detach(), text_batch, condition)
    errD_fake = lambda_mismatch * criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()

    # Train Generator
    netG.zero_grad()
    label.fill_(real_label)

    output = netD(fake, text_batch, condition)

    errG = lambda_fake * criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()

    if i % 50 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(segmented_images),
                  errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    G_losses.append(errG.item())
    D_losses.append(errD.item())
    if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(segmented_images)-1)):
        with torch.no_grad():
            fake = netG(fixed_noise, text_batch, condition).detach().cpu()
            img_list.append(fake)
            torch.save(netG.state_dict(), f'G_shape_results_{lambda_fake}/netG_{iters}.pth')
            torch.save(netD.state_dict(), f'G_shape_results_{lambda_fake}/netD.pth')
            torch.save({
                'G_losses': G_losses,
                'D_losses': D_losses,
                'img_list': img_list
            }, f'G_shape_results_{lambda_fake}/training_data.pth')
        
    iters += 1
       

# Save everything

torch.save(netG.state_dict(), 'G_shape_results/netG.pth')
torch.save(netD.state_dict(), 'G_shape_results/netD.pth')

torch.save({
    'G_losses': G_losses,
    'D_losses': D_losses,
    'img_list': img_list
}, f'./G_shape_results_{lambda_fake}/training_data.pth')
#%%