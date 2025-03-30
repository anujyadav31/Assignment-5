import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_msssim
from torch.distributions import Normal, kl_divergence
import matplotlib.pyplot as plt

# ---------------------------
# Reparameterization Trick
# ---------------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# ---------------------------
# Hierarchical VAE Encoder
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim1=128, latent_dim2=64):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU()
        )
        self.fc_mu2 = nn.Linear(256 * 4 * 4, latent_dim2)
        self.fc_logvar2 = nn.Linear(256 * 4 * 4, latent_dim2)
        self.fc_mu1 = nn.Linear(latent_dim2, latent_dim1)
        self.fc_logvar1 = nn.Linear(latent_dim2, latent_dim1)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        mu2, logvar2 = self.fc_mu2(x), self.fc_logvar2(x)
        z2 = reparameterize(mu2, logvar2)
        mu1, logvar1 = self.fc_mu1(z2), self.fc_logvar1(z2)
        z1 = reparameterize(mu1, logvar1)
        return z1, mu1, logvar1, z2, mu2, logvar2

# ---------------------------
# Prior Network (p(z1 | z2))
# ---------------------------
class PriorNetwork(nn.Module):
    def __init__(self, latent_dim1=128, latent_dim2=64):
        super(PriorNetwork, self).__init__()
        self.fc_mu1_prior = nn.Linear(latent_dim2, latent_dim1)
        self.fc_logvar1_prior = nn.Linear(latent_dim2, latent_dim1)

    def forward(self, z2):
        mu1_prior = self.fc_mu1_prior(z2)
        logvar1_prior = self.fc_logvar1_prior(z2)
        return mu1_prior, logvar1_prior

# ---------------------------
# Decoder (Modified: Uses Only z1)
# ---------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim1=128):
        super(Decoder, self).__init__()
        self.fc_z1 = nn.Linear(latent_dim1, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, z1):
        x = self.fc_z1(z1).view(-1, 256, 4, 4)
        return self.deconv(x)

# ---------------------------
# Hierarchical VAE Model
# ---------------------------
class HierarchicalVAE(nn.Module):
    def __init__(self):
        super(HierarchicalVAE, self).__init__()
        self.encoder = Encoder()
        self.prior = PriorNetwork()
        self.decoder = Decoder()

    def forward(self, x):
        z1, mu1, logvar1, z2, mu2, logvar2 = self.encoder(x)
        mu1_prior, logvar1_prior = self.prior(z2)
        return self.decoder(z1), mu1, logvar1, mu1_prior, logvar1_prior, mu2, logvar2

# ---------------------------
# Training Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = datasets.ImageFolder(root="data", transform=transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
]))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = HierarchicalVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# Training Loop with KL Logging
# ---------------------------
epochs = 300
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_kld1, total_kld2 = 0, 0
    for images, _ in dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        recon_images, mu1, logvar1, mu1_prior, logvar1_prior, mu2, logvar2 = model(images)
        kld1 = kl_divergence(Normal(mu1, torch.exp(0.5 * logvar1)), Normal(mu1_prior, torch.exp(0.5 * logvar1_prior))).mean()
        kld2 = kl_divergence(Normal(mu2, torch.exp(0.5 * logvar2)), Normal(torch.zeros_like(mu2), torch.ones_like(mu2))).mean()
        mse_loss = nn.functional.mse_loss(recon_images, images, reduction='mean')
        loss = mse_loss + 0.01 * kld1 + 0.005 * kld2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_kld1 += kld1.item()
        total_kld2 += kld2.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}, KLD1: {total_kld1/len(dataloader):.6f}, KLD2: {total_kld2/len(dataloader):.6f}")

# ---------------------------
# Enhanced Visualization
# ---------------------------
def visualize_reconstruction(model, dataloader, device, num_images=6):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(dataloader))
        images = images.to(device)
        recon_images, _, _, _, _, _, _ = model(images)
        images, recon_images = images.cpu(), recon_images.cpu()
        fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
        for i in range(num_images):
            axes[0, i].imshow(images[i].permute(1, 2, 0))
            axes[1, i].imshow(recon_images[i].permute(1, 2, 0))
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        plt.show()

visualize_reconstruction(model, dataloader, device)
