from omegaconf import DictConfig
from .off_policy_buffer import OffPolicyBuffer
from .on_policy_buffer import OnPolicyBuffer

def get_buffers(buffer_cfg: DictConfig, state_dim: int, action_dim: int):
    if buffer_cfg.type == "off_policy":
        return OffPolicyBuffer(size=int(buffer_cfg.size),
                                      state_dim=state_dim,
                                      action_dim=action_dim,
                                      n_step=buffer_cfg.n_step,
                                      gamma=buffer_cfg.gamma)
    elif buffer_cfg.type == "on_policy":
        return OnPolicyBuffer(size=int(buffer_cfg.size),
                              state_dim=state_dim,
                              action_dim=action_dim)
    else:
        raise ValueError(f"Unexpected buffer type: {buffer_cfg.type}")