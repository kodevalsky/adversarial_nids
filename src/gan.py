import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch_directml
from preprocess_data import DatasetPreprocessor

device = torch_directml.device()

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x): return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.model(x)

def train_gan_pair(gen, disc, loader, ds_name, epochs=50, latent_dim=64):
    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(gen.parameters(), lr=0.0002)
    d_opt = torch.optim.Adam(disc.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for batch in loader:
            real_data = batch[0].to(device)
            b_size = real_data.size(0)

            # Train Discriminator
            d_opt.zero_grad()
            real_label = torch.ones(b_size, 1, device=device)
            fake_label = torch.zeros(b_size, 1, device=device)
            
            d_loss_real = criterion(disc(real_data), real_label)
            z = torch.randn(b_size, latent_dim, device=device)
            fake_data = gen(z)
            d_loss_fake = criterion(disc(fake_data.detach()), fake_label)
            
            (d_loss_real + d_loss_fake).backward()
            d_opt.step()

            # Train Generator
            g_opt.zero_grad()
            g_loss = criterion(disc(fake_data), real_label)
            g_loss.backward()
            g_opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"[{ds_name}] Epoch {epoch+1}/{epochs} | D Loss: {d_loss_real.item():.4f} | G Loss: {g_loss.item():.4f}")

    torch.save(gen.state_dict(), f"generator_final_{ds_name.lower()}.pth")
    torch.save(disc.state_dict(), f"discriminator_final_{ds_name.lower()}.pth")

if __name__ == '__main__':
    configs = [
        {"name": "UNSW", "path": "./src/datasets/unsw"},
        {"name": "CIC", "path": "./src/datasets/cic"}
    ]

    for conf in configs:
        print(f"\n--- Training GAN on {conf['name']} ---")
        prep = DatasetPreprocessor(conf['path'], conf['name'])
        train_t, _, _, _ = prep.get_tensors()
        
        loader = DataLoader(TensorDataset(train_t), batch_size=4096, shuffle=True)
        gen = Generator(64, train_t.shape[1]).to(device)
        disc = Discriminator(train_t.shape[1]).to(device)
        
        train_gan_pair(gen, disc, loader, conf['name'])