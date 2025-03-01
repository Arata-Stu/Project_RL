import os
import cv2
import numpy as np
import pygame
import argparse
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from src.envs.car_racing import CarRacingWithInfoWrapper
from src.utils.timers import Timer as Timer

def main(config: argparse.Namespace):
    # 環境のセットアップ
    width, height = config.img_size, config.img_size
    env = gym.make('CarRacing-v3', render_mode=config.render)
    env = TimeLimit(env, max_episode_steps=config.num_steps)
    env = CarRacingWithInfoWrapper(env, width=width, height=height)

    mode = config.mode  # 'manual' または 'random'

    # pygame の初期化（manual モードのみ）
    if mode == "manual":
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("CarRacing Control")
        clock = pygame.time.Clock()

    # 画像保存用のベースディレクトリ
    save_dir = config.output_dir
    os.makedirs(save_dir, exist_ok=True)

    max_episodes = config.num_episodes
    max_steps = config.num_steps

    episode_count = 0
    while episode_count < max_episodes:
        obs, info = env.reset()
        step = 0
        episode_reward = 0
        done = False

        # エピソードごとのディレクトリ作成
        episode_dir = os.path.join(save_dir, f"ep{episode_count:03d}")
        os.makedirs(episode_dir, exist_ok=True)

        while not done and step < max_steps:
            if mode == "manual":
                screen.fill((0, 0, 0))
                keys = pygame.key.get_pressed()
                action = np.array([0.0, 0.0, 0.0])

                if keys[pygame.K_LEFT]:
                    action[0] = -0.3
                if keys[pygame.K_RIGHT]:
                    action[0] = 0.3
                if keys[pygame.K_UP]:
                    action[1] = 0.3
                if keys[pygame.K_DOWN]:
                    action[2] = 0.3

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        env.close()
                        return
                clock.tick(30)
            
            else:  # 'random' モード
                action = env.action_space.sample()

            # 環境をステップ実行
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if isinstance(obs, dict):
                obs = obs["image"]

            if isinstance(obs, np.ndarray):
                img_path = os.path.join(episode_dir, f"step{step:04d}.png")
                cv2.imwrite(img_path, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            step += 1

        print(f"Episode {episode_count + 1}/{max_episodes} finished. Total reward: {episode_reward:.2f}")
        episode_count += 1

    env.close()
    if mode == "manual":
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", type=str, default="manual", help="manual or random")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--num_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save images")
    parser.add_argument("--img_size", type=int, default=64, help="Image size")
    parser.add_argument("--render", type=str, default="human", help="Render mode: human, rgb_array, none")
    config = parser.parse_args()
    main(config)
