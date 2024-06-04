import torch
import torch.nn as nn
from config import config 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode_conv1 = nn.Conv2d(config['nt_input'], config['nt'], 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        
        self.g1_deconv1 = nn.ConvTranspose2d(config['nz'] + config['nt'], 64 * 16, 4)
        self.g1_bn1 = nn.BatchNorm2d(64 * 16)
        
        self.g_extra_deconv1 = nn.ConvTranspose2d(64 * 16, 64 * 8, 4, 2, 1)
        self.g_extra_bn1 = nn.BatchNorm2d(64 * 8)
        
        self.condition_conv1 = nn.Conv2d(config['n_condition'], 64, 3, 1, 1)
        self.condition_bn1 = nn.BatchNorm2d(64)
        
        self.f1_conv1 = nn.Conv2d(64, 64 * 2, 3, 1, 1)
        self.f1_bn1 = nn.BatchNorm2d(64 * 2)
        
        self.g2_deconv1 = nn.ConvTranspose2d(64 * 10, 64 * 4, 4, 2, 1)
        self.g2_bn1 = nn.BatchNorm2d(64 * 4)
        
        self.g2_conv1 = nn.Conv2d(64 * 4, 64 * 8, 3, 1, 1)
        self.g2_bn2 = nn.BatchNorm2d(64 * 8)
        
        self.m1_conv1 = nn.Conv2d(64 * 8, 64 * 8, 3, 1, 1)
        self.m1_bn1 = nn.BatchNorm2d(64 * 8)
        
        self.m2_conv1 = nn.Conv2d(64 * 8, 64 * 4, 3, 1, 1)
        self.m2_bn1 = nn.BatchNorm2d(64 * 4)
        
        self.m3_deconv1 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1)
        self.m3_bn1 = nn.BatchNorm2d(64 * 2)
        
        self.g3_deconv1 = nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1)
        self.g3_bn1 = nn.BatchNorm2d(64)
        
        self.g4_decov1 = nn.ConvTranspose2d(64, config['n_map_all'], 4, 2, 1)
        self.softmax = nn.Softmax2d()

    def forward(self, input_data, input_encode, input_condition):
        input_encode = self.encode_conv1(input_encode)
        input_encode = self.leaky_relu(input_encode)
        input_data_encode = torch.cat((input_data, input_encode), 1)

        g1 = self.g1_deconv1(input_data_encode)
        g1 = self.g1_bn1(g1)
        g1 = self.relu(g1)
        
        g_extra = self.g_extra_deconv1(g1)
        g_extra = self.g_extra_bn1(g_extra)
        g_extra = self.relu(g_extra)

        f1 = self.condition_conv1(input_condition)
        f1 = self.condition_bn1(f1)
        f1 = self.leaky_relu(f1)

        f1 = self.f1_conv1(f1)
        f1 = self.f1_bn1(f1)
        f_extra = self.leaky_relu(f1)

        g2 = torch.cat((g_extra, f_extra), 1)
        g2 = self.g2_deconv1(g2)
        g2 = self.g2_bn1(g2)
        g2 = self.relu(g2)

        g2 = self.g2_conv1(g2)
        g2 = self.g2_bn2(g2)
        m1 = self.leaky_relu(g2)

        m1 = self.m1_conv1(m1)
        m1 = self.m1_bn1(m1)
        m2 = self.leaky_relu(m1)

        m2 = self.m2_conv1(m2)
        m2 = self.m2_bn1(m2)
        m3 = self.leaky_relu(m2)

        m3 = self.m3_deconv1(m3)
        m3 = self.m3_bn1(m3)
        g3 = self.relu(m3)

        g3 = self.g3_deconv1(g3)
        g3 = self.g3_bn1(g3)
        g4 = self.relu(g3)

        g5 = self.g4_decov1(g4)
        g5 = self.softmax(g5)
        return g5

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.output_data_conv1 = nn.Conv2d(config['n_map_all'], 64, 4, 2, 1)
        
        self.d1_conv1 = nn.Conv2d(64, 64 * 2, 4, 2, 1)
        self.d1_bn1 = nn.BatchNorm2d(64 * 2)
        
        self.d2_conv1 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1)
        self.d2_bn1 = nn.BatchNorm2d(64 * 4)
        
        self.d3_conv1 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1)
        self.d3_bn1 = nn.BatchNorm2d(64 * 8)
        
        self.mid1_conv1 = nn.Conv2d(64 * 8, 64 * 8, 3, 1, 1)
        self.mid1_bn1 = nn.BatchNorm2d(64 * 8)
        
        self.mid2_conv1 = nn.Conv2d(64 * 8, 64 * 4, 3, 1, 1)
        self.mid2_bn1 = nn.BatchNorm2d(64 * 4)
        
        self.mid3_conv1 = nn.Conv2d(64 * 4, 64 * 8, 3, 1, 1)
        self.mid3_bn1 = nn.BatchNorm2d(64 * 8)
        
        self.output_condition_conv1 = nn.Conv2d(config['n_condition'], 64, 3, 1, 1)
        
        self.c1_conv1 = nn.Conv2d(64, 64 * 2, 3, 1, 1)
        self.c1_bn1 = nn.BatchNorm2d(64 * 2)
        
        self.output_fake = nn.Conv2d(64 * 10, 1, 4)
        
    def forward(self, output_data, condition_data):
        output_data = self.output_data_conv1(output_data)
        output_data = self.leaky_relu(output_data)

        d1 = self.d1_conv1(output_data)
        d1 = self.d1_bn1(d1)
        d1 = self.leaky_relu(d1)

        d2 = self.d2_conv1(d1)
        d2 = self.d2_bn1(d2)
        d2 = self.leaky_relu(d2)

        d3 = self.d3_conv1(d2)
        d3 = self.d3_bn1(d3)
        mid1 = self.leaky_relu(d3)

        mid1 = self.mid1_conv1(mid1)
        mid1 = self.mid1_bn1(mid1)
        mid2 = self.leaky_relu(mid1)

        mid2 = self.mid2_conv1(mid2)
        mid2 = self.mid2_bn1(mid2)
        mid3 = self.leaky_relu(mid2)

        mid3 = self.mid3_conv1(mid3)
        mid3 = self.mid3_bn1(mid3)
        mid4 = self.leaky_relu(mid3)

        output_condition = self.output_condition_conv1(condition_data)
        output_condition = self.leaky_relu(output_condition)

        c1 = self.c1_conv1(output_condition)
        c1 = self.c1_bn1(c1)
        c2 = self.leaky_relu(c1)

        d4 = torch.cat((mid4, c2), 1)
        output = self.output_fake(d4)
        return output
