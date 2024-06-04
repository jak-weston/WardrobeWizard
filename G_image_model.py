import torch
import torch.nn as nn
from config import config

nc = config['n_c']
ncondition = nc
nz = config['nz']
nt_input = config['nt_input']
nt = config['nt']

ndf = 64
ngf = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode_conv1 = nn.Conv2d(nt_input,nt, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        
        self.g1_deconv1 = nn.ConvTranspose2d(nz + nt, ngf * 16, 4)
        self.g1_bn1 = nn.BatchNorm2d(ngf * 16)
        
        self.g_extra_deconv1 = nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1)
        self.g_extra_bn1 = nn.BatchNorm2d(ngf * 8)

        self.g2_deconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.g2_bn1 = nn.BatchNorm2d(ngf * 8)
        
        self.condition_conv1 = nn.Conv2d(7, ngf, 4, 2, 1)
        self.condition_bn1 = nn.BatchNorm2d(ngf)
        
        self.f1_conv1 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.f1_bn1 = nn.BatchNorm2d(ngf * 2)

        self.f_extra_conv1 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.f_extra_bn1 = nn.BatchNorm2d(ngf * 4)

        self.g3_deconv1 = nn.ConvTranspose2d(ngf * 12, ngf * 8, 4, 2, 1)
        self.g3_bn1 = nn.BatchNorm2d(ngf * 8)

        self.g3_conv1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.g3_bn1 = nn.BatchNorm2d(ngf * 8)
        
        self.mid1_conv1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.mid1_bn1 = nn.BatchNorm2d(ngf * 8)
        
        self.mid2_conv1 = nn.Conv2d(ngf * 8, ngf * 8, 3, 1, 1)
        self.mid2_bn1 = nn.BatchNorm2d(ngf * 8)

        self.mid3_conv1 = nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1)
        self.mid3_bn1 = nn.BatchNorm2d(ngf * 4)

        self.mid4_conv1 = nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1)
        self.mid4_bn1 = nn.BatchNorm2d(ngf * 2)
        
        self.mid5_deconv1 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1)
        self.mid5_bn1 = nn.BatchNorm2d(ngf)

        self.g4_deconv1 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1)
        self.g5_tanh = nn.Tanh()


    def forward(self, input_data, input_encode, input_condition):
        input_encode = self.encode_conv1(input_encode)
        h1 = self.leaky_relu(input_encode)

        input_data_encode = torch.cat((input_data, h1), 1)

        g1 = self.g1_deconv1(input_data_encode)
        g1 = self.g1_bn1(g1)
        g1 = self.relu(g1)
        
        g_extra = self.g_extra_deconv1(g1)
        g_extra = self.g_extra_bn1(g_extra)
        g_extra = self.relu(g_extra)

        g2 = self.g2_deconv1(g_extra)
        g2 = self.g2_bn1(g2)
        g2 = self.relu(g2)

        f1 = self.condition_conv1(input_condition)
        f1 = self.condition_bn1(f1)
        f1 = self.leaky_relu(f1)

        f1 = self.f1_conv1(f1)
        f1 = self.f1_bn1(f1)
        f_extra = self.leaky_relu(f1)

        f_extra = self.f_extra_conv1(f_extra)
        f_extra = self.f_extra_bn1(f_extra)
        f2 = self.leaky_relu(f_extra)

        g3 = torch.cat((g2, f2), 1)

        g3 = self.g3_deconv1(g3)
        g3 = self.g3_bn1(g3)
        g3 = self.relu(g3)

        g3 = self.g3_conv1(g3)
        g3 = self.g3_bn1(g3)
        mid1 = self.leaky_relu(g3)

        mid1 = self.mid1_conv1(mid1)
        mid1 = self.mid1_bn1(mid1)
        mid2 = self.leaky_relu(mid1)

        mid2 = self.mid2_conv1(mid2)
        mid2 = self.mid2_bn1(mid2)
        mid3 = self.leaky_relu(mid2)

        mid3 = self.mid3_conv1(mid3)
        mid3 = self.mid3_bn1(mid3)
        mid4 = self.leaky_relu(mid3)

        mid4 = self.mid4_conv1(mid4)
        mid4 = self.mid4_bn1(mid4)
        mid5 = self.leaky_relu(mid4)

        mid5 = self.mid5_deconv1(mid5)
        mid5 = self.mid5_bn1(mid5)
        g4 = self.relu(mid5)

        g5 = self.g4_deconv1(g4)
        g5 = self.g5_tanh(g5)
        
        return g5

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.output_data_conv1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        
        self.d1_conv1 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.d1_bn1 = nn.BatchNorm2d(ndf * 2)
        
        self.d2_conv1 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.d2_bn1 = nn.BatchNorm2d(ndf * 4)
        
        self.output_condition_conv1 = nn.Conv2d(7, ndf, 4, 2, 1)

        self.c1_conv1 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.c1_bn1 = nn.BatchNorm2d(ndf * 2)

        self.c2_conv1 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.c2_bn1 = nn.BatchNorm2d(ndf * 4)

        self.m1_conv1 = nn.Conv2d(ndf * 8, ndf * 8, 3,1,1)
        self.m1_bn1 = nn.BatchNorm2d(ndf * 8)

        self.m2_conv1 = nn.Conv2d(ndf * 8, ndf * 8, 3,1,1)
        self.m2_bn1 = nn.BatchNorm2d(ndf * 8)

        self.m3_conv1 = nn.Conv2d(ndf * 8, ndf * 8, 3,1,1)
        self.m3_bn1 = nn.BatchNorm2d(ndf * 8)

        self.m4_conv1 = nn.Conv2d(ndf * 8, ndf * 8, 3,1,1)
        self.m4_bn1 = nn.BatchNorm2d(ndf * 8)

        self.m5_conv1 = nn.Conv2d(ndf * 8, ndf * 8, 4,2,1)
        self.m5_bn1 = nn.BatchNorm2d(ndf * 8)

        self.d4_conv1 = nn.Conv2d(ndf * 8, ndf*16, 4,2,1)
        self.d4_bn1 = nn.BatchNorm2d(ndf*16)

        self.output_encode_conv1 = nn.Conv2d(nt_input, nt, 1)
        self.output_encode_bn1 = nn.BatchNorm2d(nt)

        self.d_extra_b1_conv1 = nn.Conv2d(ndf * 16 + nt, ndf * 16, 1)
        self.d_extra_b1_bn1 = nn.BatchNorm2d(ndf * 16)

        self.d_extra_b1_conv2 = nn.Conv2d(ndf * 16, 1, 4)
        self.sigmoid = nn.Sigmoid()

        
       
        
    def forward(self,output_data, output_encode, output_condition):

        output_data = self.output_data_conv1(output_data)
        d1 = self.leaky_relu(output_data)

        d1 = self.d1_conv1(d1)
        d1 = self.d1_bn1(d1)
        d2 = self.leaky_relu(d1)

        d2 = self.d2_conv1(d2)
        d2 = self.d2_bn1(d2)
        d3 = self.leaky_relu(d2)

        output_condition = self.output_condition_conv1(output_condition)
        c1 = self.leaky_relu(output_condition)

        c1 = self.c1_conv1(c1)
        c1 = self.c1_bn1(c1)
        c2 = self.leaky_relu(c1)

        c2 = self.c2_conv1(c2)
        c2 = self.c2_bn1(c2)
        c3 = self.leaky_relu(c2)

        m1 = torch.cat((d3, c3), 1)
        m1 = self.m1_conv1(m1)
        m1 = self.m1_bn1(m1)
        m2 = self.leaky_relu(m1)

        m2 = self.m2_conv1(m2)
        m2 = self.m2_bn1(m2)
        m3 = self.leaky_relu(m2)

        m3 = self.m3_conv1(m3)
        m3 = self.m3_bn1(m3)
        m4 = self.leaky_relu(m3)

        m4 = self.m4_conv1(m4)
        m4 = self.m4_bn1(m4)
        m5 = self.leaky_relu(m4)

        m5 = self.m5_conv1(m5)
        m5 = self.m5_bn1(m5)
        d4 = self.leaky_relu(m5)

        d4 = self.d4_conv1(d4)
        d4 = self.d4_bn1(d4)
        d_extra = self.leaky_relu(d4)

        output_encode = self.output_encode_conv1(output_encode)
        output_encode = self.output_encode_bn1(output_encode)
        b1 = self.leaky_relu(output_encode)

        b1 = b1.repeat(1, 1, d_extra.shape[2], d_extra.shape[3])

        d_extra_b1 = torch.cat((d_extra, b1), 1)

        d_extra_b1 = self.d_extra_b1_conv1(d_extra_b1)
        d_extra_b1 = self.d_extra_b1_bn1(d_extra_b1)
        d_extra_b1 = self.leaky_relu(d_extra_b1)

        d_extra_b1 = self.d_extra_b1_conv2(d_extra_b1)
        output = self.sigmoid(d_extra_b1)
       
        return output
