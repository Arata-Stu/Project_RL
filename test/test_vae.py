import sys
sys.path.append('../')

import hydra
from omegaconf import DictConfig, OmegaConf
import cv2
import numpy as np
import torch
import pygame
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from src.envs.car_racing import CarRacingWithInfoWrapper
from src.utils.timers import Timer as Timer
from src.models.VAE.VAE import get_vae

@hydra.main(config_path="../configs", config_name="default", version_base="1.2")
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 環境のセットアップ
    ch, width, height = config.vae.input_shape
    env = gym.make('CarRacing-v3', render_mode="human")
    env = TimeLimit(env, max_episode_steps=1000)
    env = CarRacingWithInfoWrapper(env, width=width, height=height)

    vae = get_vae(vae_cfg=config.vae).eval().to(device)

    mode = config.mode  # 'manual' または 'random'
    save_video = config.get("save_video", False)  # 動画保存の選択

    # pygame の初期化（manual モードのみ）
    if mode == "manual":
        pygame.init()
        pygame.display.set_mode((200, 150))
        pygame.display.set_caption("CarRacing Control")
        clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    step_count = 0

    # 動画出力の設定（必要な場合のみ）
    if save_video:
        video_filename = "car_racing_output.mp4"
        frame_size = (width * 2, height)  # 入力画像と再構成画像を横に並べるため width * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, frame_size)

    while not done:
        if mode == "manual":
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
                    cv2.destroyAllWindows()
                    if save_video:
                        video_writer.release()
                    return
            clock.tick(30)
        
        else:  # 'random' モード
            action = env.action_space.sample()

        ## 画像をVAEに入力して、再構成画像を得る
        obs_img = obs["image"]
        obs_img_tensor = torch.from_numpy(obs_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            recon_img_tensor, _, _ = vae(obs_img_tensor)

        recon_img = (recon_img_tensor.squeeze(0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        recon_img = np.moveaxis(recon_img, 0, -1)  # CHW → HWC

        ## 入力と出力画像を結合して表示
        img = np.hstack((obs_img, recon_img))
        ## RGB 画像を BGR 画像に変換
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("CarRacing", img)
        cv2.waitKey(1)

        # 動画にフレームを保存（必要な場合のみ）
        if save_video:
            video_writer.write(img)

        # 環境をステップ実行
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    # 動画のリソース解放（必要な場合のみ）
    if save_video:
        video_writer.release()

    if mode == "manual":
        pygame.quit()
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()