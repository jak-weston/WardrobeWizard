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

class Descriminator(nn.Module):
    def __init__(self):
      super(Descriminator, self).__init__()
      self.leakyrelu = nn.LeakyReLU(0.2)

      self.output_data_conv1 = nn.Conv2d(nc, ndf, 4, 2, 1)

      self.d1_conv1 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
      self.d1_bn1 = nn.BatchNorm2d(ndf * 2)

      self.d2_conv1 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
      self.d2_bn1 = nn.BatchNorm2d(ndf * 4)

      self.d3_conv1 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)
      self.d3_bn1 = nn.BatchNorm2d(ndf * 8)

      self.mid1_conv1 = nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1)
      self.mid1_bn1 = nn.BatchNorm2d(ndf * 8)

      self.mid2_conv1 = nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1)
      self.mid2_bn1 = nn.BatchNorm2d(ndf * 4)

      self.mid3_conv1 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1)
      self.mid3_bn1 = nn.BatchNorm2d(ndf * 8)

      self.output_condition_conv1 = nn.Conv2d(ncondition, ndf, 3, 1, 1)

      self.c1_conv1 = nn.Conv2d(ndf, ndf * 2, 3, 1, 1)
      self.c1_bn1 = nn.BatchNorm2d(ndf * 2)

      self.d_extra_conv1 = nn.Conv2d(ndf * 10, ndf * 8, 4, 2, 1)
      self.d_extra_bn1 = nn.BatchNorm2d(ndf * 8)

      self.output_encde_conv1 = nn.Conv2d(nt_input, nt, 1)
      self.output_encode_bn1 = nn.BatchNorm2d(nt)

      self.d_extra_b1_conv1 = nn.Conv2d(ndf * 8 + nt, ndf*8, 1)
      self.d_extra_b1_bn1 = nn.BatchNorm2d(ndf * 8)

      self.d_extra_b1_conv2 = nn.Conv2d(ndf * 8, 1, 4)
      self.sigmoid = nn.Sigmoid()



    #output data = 
    def forward(self, output_data, output_encode, output_condition):
        #Handle output_data 
        output_data = self.output_data_conv1(output_data)
        d1 = self.leakyrelu(output_data)

        d1 = self.d1_conv1(d1)
        d1 = self.d1_bn1(d1)
        d2 = self.leakyrelu(d1)

        d2 = self.d2_conv1(d2)
        d2 = self.d2_bn1(d2)
        d3 = self.leakyrelu(d2)

        d3 = self.d3_conv1(d3)
        d3 = self.d3_bn1(d3)
        mid1 = self.leakyrelu(d3)

        mid1 = self.mid1_conv1(mid1)
        mid1 = self.mid1_bn1(mid1)
        mid2 = self.leakyrelu(mid1)

        mid2 = self.mid2_conv1(mid2)
        mid2 = self.mid2_bn1(mid2)
        mid3 = self.leakyrelu(mid2)

        mid3 = self.mid3_conv1(mid3)
        mid3 = self.mid3_bn1(mid3)
        d4 = self.leakyrelu(mid3)

        #handle output_condition
        output_condition = self.output_condition_conv1(output_condition)
        c1 = self.leakyrelu(output_condition)

        c1 = self.c1_conv1(c1)
        c1 = self.c1_bn1(c1)
        c2 = self.leakyrelu(c1)

        d_extra = torch.cat((d4, c2), 1)
        d_extra = self.d_extra_conv1(d_extra)
        d_extra = self.d_extra_bn1(d_extra)
        d_extra = self.leakyrelu(d_extra)

        #Handle output_encode
        output_encode = self.output_encde_conv1(output_encode)
        output_encode = self.output_encode_bn1(output_encode)
        b1 = self.leakyrelu(output_encode)

        #Replicate b1 dimensions to match d_extra
        b1 = b1.repeat(1, 1, d_extra.shape[2], d_extra.shape[3])

        d_extra_b1 = torch.cat((d_extra, b1), 1)
        d_extra_b1 = self.d_extra_b1_conv1(d_extra_b1)
        d_extra_b1 = self.d_extra_b1_bn1(d_extra_b1)
        d_extra_b1 = self.leakyrelu(d_extra_b1)

        d_extra_b1 = self.d_extra_b1_conv2(d_extra_b1)
        return self.sigmoid(d_extra_b1)



        


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

iters = 0

print("Starting Training Loop...")
torch.autograd.set_detect_anomaly(True)
num_epochs = 1
for epoch in range(num_epochs):
  for i in range(0, segmented_images.shape[0], batch_size):
    segmented_images_batch = segmented_images[i:i+batch_size]
    text_batch = text[i:i+batch_size]
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
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake
    optimizerD.step()

    # Train Generator
    netG.zero_grad()
    label.fill_(real_label)

    output = netD(fake, text_batch, condition)

    errG = criterion(output, label)
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
    iters += 1

# Save everything

torch.save(netG.state_dict(), 'netG.pth')
torch.save(netD.state_dict(), 'netD.pth')

torch.save({
    'G_losses': G_losses,
    'D_losses': D_losses,
    'img_list': img_list
}, 'training_data.pth')
#%%