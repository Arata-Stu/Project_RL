from omegaconf import DictConfig
from .sac import SACAgent
from .td3 import TD3Agent

def get_agents(agent_cfg: DictConfig, state_dim: int, action_dim: int):
    if agent_cfg.name == "SAC":
        return SACAgent(state_dim=state_dim,
                        action_dim=action_dim,
                        actor_lr=agent_cfg.actor_lr,
                        critic_lr=agent_cfg.critic_lr,
                        alpha_lr=agent_cfg.alpha_lr,
                        gamma=agent_cfg.gamma,
                        tau=agent_cfg.tau,
                        hidden_dim=agent_cfg.hidden_dim)
    elif agent_cfg.name == "TD3":
        return TD3Agent(state_dim=state_dim,
                        action_dim=action_dim,
                        actor_lr=agent_cfg.actor_lr,
                        critic_lr=agent_cfg.critic_lr,
                        gamma=agent_cfg.gamma,
                        tau=agent_cfg.tau,
                        hidden_dim=agent_cfg.hidden_dim,
                        policy_noise=agent_cfg.policy_noise,
                        noise_clip=agent_cfg.noise_clip,
                        policy_delay=agent_cfg.policy_delay)
    
    else:
        raise NotImplementedError(f"Unknown agent: {agent_cfg.name}")