import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.utils.timers import CudaTimer as Timer
# from src.utils.timers import TimerDummy as Timer

class BaseVAE(nn.Module):
    def __init__(self, latent_dim: int, input_shape: Tuple[int, int, int] = (3, 64, 64)):
        super().__init__()
        self.latent_size = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def latent(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        with Timer(device=mu.device, timer_name="latent"):
            sigma = torch.exp(0.5 * logvar)
            eps = torch.randn_like(logvar).to(self.device)
            z = mu + eps * sigma
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with Timer(device=x.device, timer_name="encode + decode"):
            mu, logvar = self.encode(x)
            z = self.latent(mu, logvar)
            out = self.decode(z)
        return out, mu, logvar
    
    def vae_loss(self, out, y, mu, logvar):
        BCE = F.binary_cross_entropy(out, y, reduction="sum")
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KL, BCE, KL
    
    def get_latent_size(self):
        
        return self.latent_size