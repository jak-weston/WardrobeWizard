import torch
import torch.optim as optim
import torch.nn as nn
from config import config
from dataloader import get_text_data, load_segmented_images
from net_graph_sr1 import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
input_data = get_text_data().to(device)
input_images = load_segmented_images().to(device)

# Initialize models
netG = Generator().to(device)
netD = Discriminator().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 100  # Change as needed
for epoch in range(num_epochs):
    for i in range(0, len(input_data), config['batchSize']):
        # Prepare batch data
        batch_data = input_data[i:i + config['batchSize']]
        batch_images = input_images[i:i + config['batchSize']]

        # Create labels
        real_label = torch.ones(config['batchSize'], 1, 1, 1).to(device)
        fake_label = torch.zeros(config['batchSize'], 1, 1, 1).to(device)

        ############################
        # (1) Update D network
        ###########################
        netD.zero_grad()

        # Train with real
        output = netD(batch_images, batch_data)
        errD_real = criterion(output, real_label)
        errD_real.backward()

        # Train with fake
        noise = torch.randn(config['batchSize'], config['nz'], 1, 1).to(device)
        fake = netG(noise, batch_data, batch_images)
        output = netD(fake.detach(), batch_data)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()
        optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        output = netD(fake, batch_data)
        errG = criterion(output, real_label)
        errG.backward()
        optimizerG.step()

    print(f'Epoch [{epoch}/{num_epochs}] - Loss D: {errD_real.item() + errD_fake.item()}, Loss G: {errG.item()}')
