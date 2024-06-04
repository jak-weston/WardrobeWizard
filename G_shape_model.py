import torch
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

