import sys
sys.path.append('../')
import torch
from omegaconf import OmegaConf
from src.models.VAE.VAE import get_vae

def test_vae(yaml_path: str, name:str = "vae" ):
    cfg = OmegaConf.load(yaml_path)
    model = get_vae(cfg)
    dummy_input = torch.rand(1, 3, 64, 64)
    x_recon, mu, log_var = model(dummy_input)
    loss, recon_loss, kl_loss = model.vae_loss(dummy_input, x_recon, mu, log_var)
    ## log にnameを含める. shapeの確認も追加
    print(f"{name} test passed")
    print(f"output shape: {x_recon.shape}")
    print(f"loss: {loss}")
    print(f"recon_loss: {recon_loss}")
    print(f"kl_loss: {kl_loss}")


test_vae("../configs/vae/cnn.yaml")
test_vae("../configs/vae/maxvit.yaml")