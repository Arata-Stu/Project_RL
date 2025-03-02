import sys
sys.path.append('../')
import torch
from omegaconf import OmegaConf
from src.models.VAE.VAE import get_vae

if __name__ == "__main__":
    # テスト用：ダミー入力
    yaml_path = '../configs/vae/maxvit.yaml'
    cfg = OmegaConf.load(yaml_path)
    model = get_vae(cfg)
    dummy_input = torch.rand(1, 3, 64, 64)
    x_recon, mu, log_var = model(dummy_input)
    loss, recon_loss, kl_loss = model.vae_loss(dummy_input, x_recon, mu, log_var)

    print("Reconstructed output shape:", x_recon.shape)  # (B, 3, 256, 256)
    print("Latent mean shape:", mu.shape)  # (B, 512)
    print("Latent log variance shape:", log_var.shape)  # (B, 512)
    print("Total loss:", loss.item(), "Recon loss:", recon_loss.item(), "KL loss:", kl_loss.item())
