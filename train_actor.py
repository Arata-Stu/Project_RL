from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
import traceback
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import os

from envs.car_racing import CarRacingWithInfoWrapper
from agents.agents import get_agents
from buffers.buffers import get_buffers
from models.VAE.VAE import get_vae
from utils.helppers import numpy2img_tensor
from utils.timers import Timer as Timer
# from utils.timers import TimerDummy as Timer

@hydra.main(config_path='configs', config_name='train', version_base='1.2')
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## 環境の作成
    width, height = config.envs.img_size, config.envs.img_size
    env = gym.make('CarRacing-v3', render_mode=config.envs.render_mode)
    env = TimeLimit(env, max_episode_steps=config.envs.num_steps)
    env = CarRacingWithInfoWrapper(env, width=width, height=height)

    ## エージェントの作成
    state_dim = 64
    action_dim = 3
    agent = get_agents(agent_cfg=config.agent, state_dim=state_dim, action_dim=action_dim)

    # バッファの作成
    buffer = get_buffers(buffer_cfg=config.buffer, state_dim=state_dim, action_dim=action_dim)

    ## エンコーダの作成
    vae = get_vae(vae_cfg=config.vae).to(device).eval()
    
    # TensorBoardの初期化
    writer = SummaryWriter(log_dir=config.get("log_dir", "./runs"))

    obs, info = env.reset()

    max_episodes = config.max_episodes
    max_steps = config.max_steps

    episode_rewards = []
    top_models = []  # トップモデルの初期化

    # 再構成画像のログ出力間隔（例: 50ステップごと）
    log_interval = config.get("reconstructed_log_interval", 50)

    try:
        # エピソードループ
        for episode in range(max_episodes):
            obs, vehicle_info = env.reset()
            episode_reward = 0

            print(f"Episode {episode} started.")
            # ステップループ
            for step in range(max_steps):
                obs_img = obs["image"].copy()

                with Timer("Env: Encoding"):
                    obs_img = numpy2img_tensor(obs_img).unsqueeze(0).to(device)
                    state = vae.obs_to_z(obs_img) ## [1, 64]

                with Timer("Env: Decoding"):
                    reconstucted_img = vae.decode(state)

                # 再構成画像を間隔ごとにTensorBoardに記録（HWC→CHW変換）
                if step % log_interval == 0:
                    reconstucted_img= reconstucted_img.squeeze(0).float().cpu().detach()
                    global_step = episode * max_steps + step
                    writer.add_image("Reconstructed/Image", reconstucted_img, global_step)

                with Timer("Agent Action"):
                    action = agent.select_action(state=state, evaluate=False)

                # TensorBoard にアクションの分布を記録
                writer.add_histogram("Action/Distribution", action, episode)

                with Timer("Environment Step"):
                    next_obs, reward, terminated, truncated, info = env.step(action)

                next_obs_img = next_obs["image"].copy()

                with Timer("Next Encoding"):
                    next_obs_img = numpy2img_tensor(next_obs_img).unsqueeze(0).to(device)
                    next_state = vae.obs_to_z(next_obs_img)

                with Timer("Buffer Add"):
                    done = terminated or truncated
                    buffer.add(state, action, reward, next_state, done)

                episode_reward += reward

                if len(buffer) >= config.batch_size:
                    update_info = agent.update(buffer, batch_size=config.batch_size)
                    global_step = episode * max_steps + step
                    writer.add_scalar("Loss/critic", update_info["critic_loss"], global_step)
                    writer.add_scalar("Loss/actor", update_info["actor_loss"], global_step)
                    writer.add_scalar("Loss/alpha", update_info["alpha_loss"], global_step)
                    writer.add_scalar("Alpha", update_info["alpha"], global_step)

                obs = next_obs

                if terminated or truncated:
                    with Timer("Environment Reset"):
                        obs, vehicle_info = env.reset()
                        print(f"Episode {episode}: Step {step} terminated because of terminated: {terminated} or truncated: {truncated}")
                    break

            episode_rewards.append(episode_reward)
            writer.add_scalar("Reward/Episode", episode_reward, episode)

            os.makedirs(config.save_ckpt_dir, exist_ok=True)
            # トップモデルの保存処理
            if len(top_models) < 3:
                top_models.append((episode, episode_reward))
                agent.save(f"{config.save_ckpt_dir}/best_{episode_reward:.2f}_ep_{episode}.pt", episode)
            else:
                min_reward = min(top_models, key=lambda x: x[1])[1]
                if episode_reward > min_reward:
                    top_models = [model for model in top_models if model[1] != min_reward]
                    top_models.append((episode, episode_reward))
                    agent.save(f"{config.save_ckpt_dir}/best_{episode_reward:.2f}_ep_{episode}.pt", episode)

            print(f"Episode {episode}: Reward = {episode_reward:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        writer.close()
        print("Cleaned up resources.")


if __name__ == '__main__':
    main()
