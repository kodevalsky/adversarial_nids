import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch_directml
from preprocess_data import DatasetPreprocessor
from gan import Generator

class Encoder(nn.Module):
    """Maps the input data space directly to the latent space z."""
    def __init__(self, input_dim, latent_dim=64):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

class F_AnoGAN(nn.Module):
    def __init__(self, generator, encoder, discriminator, lambda_weight=0.1):
        super(F_AnoGAN, self).__init__()
        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator # Add this!
        self.lambda_weight = lambda_weight
        self.criterion_mse = nn.MSELoss(reduction='none')

    def forward(self, x):
        return self.generator(self.encoder(x))

    def get_anomaly_score(self, x):
        """Calculates the FULL AnoGAN score in a single forward pass."""
        z = self.encoder(x)
        gen_x = self.generator(z)
        
        # 1. Residual Loss
        residual_loss = self.criterion_mse(gen_x, x).mean(dim=1)
        
        # 2. Discrimination Loss
        d_out = self.discriminator(gen_x).squeeze()
        discrimination_loss = torch.abs(1.0 - d_out)
        
        # 3. Total Anomaly Score
        return (1 - self.lambda_weight) * residual_loss + self.lambda_weight * discrimination_loss

def train_f_anogan_encoder(encoder, generator, loader, device, ds_name, epochs=15):
    """Trains the Encoder to invert the pre-trained Generator (IZI mapping)."""
    criterion_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    # CRITICAL: Freeze the Generator. We only train the Encoder!
    generator.eval() 
    for param in generator.parameters():
        param.requires_grad = False
        
    encoder.train()

    print(f"\n--- Training f-AnoGAN Encoder on {ds_name} ---")
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            real_x = batch[0].to(device)
            optimizer.zero_grad()
            
            # Forward pass: x -> Encoder -> z -> Generator -> reconstructed_x
            z = encoder(real_x)
            reconstructed_x = generator(z)
            
            # We want the Encoder to find a z that perfectly reconstructs x
            loss = criterion_mse(reconstructed_x, real_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Reconstruction Loss: {total_loss/len(loader):.6f}")

    # Save the trained Encoder
    save_path = f"encoder_final_{ds_name.lower()}.pth"
    torch.save(encoder.state_dict(), save_path)
    print(f"-> Saved {save_path}")

if __name__ == '__main__':
    # Initialize GPU
    device = torch_directml.device()
    print(f"Executing f-AnoGAN training on device: {device}")

    configs = [
        {"name": "UNSW", "path": "./src/datasets/unsw", "gen_path": "./models/generator_final_unsw.pth"},
        {"name": "CIC", "path": "./src/datasets/cic", "gen_path": "./models/generator_final_cic.pth"}
    ]

    for conf in configs:
        try:
            # 1. Load Data
            prep = DatasetPreprocessor(conf['path'], conf['name'])
            train_t, _, _, _ = prep.get_tensors()
            loader = DataLoader(TensorDataset(train_t), batch_size=2048, shuffle=True)
            input_dim = train_t.shape[1]

            # 2. Load Existing Frozen Generator
            gen = Generator(64, input_dim).to(device)
            gen.load_state_dict(torch.load(conf['gen_path'], map_location=device))
            
            # 3. Initialize New Encoder
            enc = Encoder(input_dim, latent_dim=64).to(device)
            
            # 4. Train and Save Encoder
            train_f_anogan_encoder(enc, gen, loader, device, conf['name'])
            
        except FileNotFoundError as e:
            print(f"Skipping {conf['name']}: {e}")