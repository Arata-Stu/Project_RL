# Project RL
このレポジトリはVAEを用いた強化学習を行う

## Setup
ubuntu 22.04でテストを動作を確認済み

``` shell
# 例 python3.11 をインストール
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt -y update
sudo apt install -y python3.11
sudo apt install -y python3.11-venv
```

```shell
# python3.11 -m venv <env_name>
# source <env_name>/bin/activate

#例 gym_envという名前の仮想環境を作成
python3.11 -m venv gym_env
source gym_env/bin/activate
```

## 1. VAEを学習させるためのデータセットを集める

```shell
python3 collect_data.py \
--mode random \
--num_episodes 10 \
--num_steps 1000 \
--output_dir ./data/vae_train \
--img_size 64 \
--render human
```

## 2. VAEを学習させる

### 2-1. スクラッチからCOCOで学習
```shell
python3 train_vae.py \
save_ckpt_dir=./ckpts/scratch_cnn_vae/ \
data.data_dir=./data/car_racing/ \
```