import torch
import torch.nn as nn
import torch.nn.functional as F
from vaes import FMCVAE

class SimpleDiffusionRefiner(nn.Module):
    def __init__(self, num_steps=2, hidden_dim=64):
        super(SimpleDiffusionRefiner, self).__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        for _ in range(self.num_steps):
            residual = x
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            x = x + 0.1 * out 
            x = torch.clamp(x, 0.0, 1.0) 
        return x

class FMCVAE_With_Diffusion(FMCVAE):
    def __init__(self, *args, diffusion_steps=2, diffusion_hidden_dim=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_refiner = SimpleDiffusionRefiner(
            num_steps=diffusion_steps,
            hidden_dim=diffusion_hidden_dim
        )
    
    def forward(self, x):
        z_params = self.encode(x)
        z = self.reparameterize(z_params)
        x_recon_logits = self.decode(z)
        x_recon = torch.sigmoid(x_recon_logits)
        
        x_recon_refined = self.diffusion_refiner(x_recon)

        return x_recon_logits, x_recon_refined, z_params

    def generate_samples(self, n_samples=16, device='cuda'):
        z = torch.randn(n_samples, self.hidden_dim).to(device)
        x_gen_logits = self.decode(z)
        x_gen = torch.sigmoid(x_gen_logits)
        x_gen_refined = self.diffusion_refiner(x_gen)
        return x_gen_refined

