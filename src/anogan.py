import torch
from torch import nn
import torch.optim as optim
from utils import set_deterministic_seed

class StaticAnoGAN(nn.Module):
    """Wrapper to lock the optimized latent vector for the FGSM attack."""
    def __init__(self, generator, optimized_z):
        super().__init__()
        self.generator = generator
        self.optimized_z = optimized_z

    def forward(self, x):
        return self.generator(self.optimized_z)

def anogan_evaluate(generator, discriminator, real_x, device, iterations=500, lambda_weight=0.1):
    """Optimizes the latent vector z to find the closest normal data representation."""
    batch_size = real_x.size(0)
    latent_dim = 64
    z = torch.randn(batch_size, latent_dim, device=device, requires_grad=True)
    z_optimizer = optim.Adam([z], lr=0.1)
    criterion_mse = nn.MSELoss(reduction='none')
    
    generator.eval()
    discriminator.eval()
    
    for _ in range(iterations):
        z_optimizer.zero_grad()
        generated_x = generator(z)
        residual_loss = criterion_mse(generated_x, real_x).mean(dim=1)
        
        d_out = discriminator(generated_x).squeeze()
        discrimination_loss = torch.abs(1.0 - d_out)
        
        loss = (1 - lambda_weight) * residual_loss + lambda_weight * discrimination_loss
        loss.sum().backward()
        z_optimizer.step()
        
    return loss.detach(), z.detach()