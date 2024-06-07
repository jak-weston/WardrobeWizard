#%%
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import dataloader
import G_image_model
# Read results
#%%
results = torch.load('G_image_results_0.8/training_data.pth')

G_losses = results['G_losses']
D_losses = results['D_losses']
img_list = results['img_list']

#%%


# Plot the training losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("G_image_results_0.8_loss.pdf")
plt.show()
# %%
fig, axs = plt.subplots(2, 4, figsize=(10,5))
a = 0
img_length = len(img_list)
plot_index = range(len(img_list))
mean_image = dataloader.get_image_mean()
# Plot the generated images
for i in plot_index:
    img = img_list[i]

    if i * 500 in [0, 500, 5500, 11000, 19000, 38000, 70000,85000]:
        axs[a//4, a%4].imshow(np.transpose(img[0].detach().cpu().numpy()) + mean_image.numpy().transpose(2,1,0))
        axs[a//4, a%4].axis("off")
        axs[a//4, a%4].set_title("iteration {}".format(i * 500))
        a += 1

fig.tight_layout()
plt.savefig("G_image_results_0.8_images.pdf")
plt.show()
# %%


segments = dataloader.load_segmented_images()
sentence = dataloader.get_text_data()


G = G_image_model.Generator()

# Attempt with specific sentence

#%%

G.load_state_dict(torch.load('G_image_results/netG_1000.pth'))
sen = sentence[10].unsqueeze(0).unsqueeze(2).unsqueeze(3).float()
seg = segments[0].unsqueeze(0).float()
nz = 80
noise = torch.randn(1, nz, 1, 1)
fake = G(noise, sen, seg)[0]

plt.figure(figsize=(10,10))
plt.axis("off")
plt.title("Generated Image")
plt.imshow(np.transpose(fake.detach().numpy()))
plt.show()


# %%
