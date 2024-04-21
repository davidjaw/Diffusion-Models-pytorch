import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
from lion_pytorch import Lion


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        """
        初始化 Diffusion 模型的參數。
        参数：
        noise_steps (int): 噪聲步數，控制噪聲增加的細膩度。
        beta_start (float): β序列的起始值，控制噪聲的初步強度。
        beta_end (float): β序列的終止值，控制最終噪聲的強度。
        img_size (int): 生成圖像的大小（寬和高相同）。
        device (str): 計算將執行的設備（例如“cuda”或“cpu”）。
        """
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """
        準備噪聲程度的時間表。
        return:
            一個從 beta_start 到 beta_end 的線性間隔 Tensor，長度為 noise_steps。
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        對輸入圖像應用擴散模型的前向公式加入隨機噪聲
        参数：
            x (Tensor): 原始圖像的 Tensor
            t (Tensor): 指定的時間步長
        return:
            帶噪聲的圖像和用於生成噪聲的隨機向量ɛ
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """
        隨機生成時間步長。
            n (int): 需要生成的時間步長數量。
        return:
            一個隨機選擇的時間步長的Tensor。
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        """
        使用 Diffusion 模型的反向公式生成新圖像。
        参数：
            model (torch.nn.Module): 預訓練的生成模型。
            n (int): 要生成的圖像數量。
        return:
            生成的圖像 Tensor。
        """
        logging.info(f"Sampling {n} new images with DDPM....")
        model.eval()
        with torch.no_grad():
            # 生成隨機噪聲
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # 定義當前時態 t
                t = (torch.ones(n) * i).long().to(self.device)
                # 生成預測的噪聲
                predicted_noise = model(x, t)
                # 進行反向計算
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        # 將生成的圖像轉換為 0-255 的範圍
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        model.train()
        return x


def train(args):
    # 設定訓練過程的 logging 參數
    setup_logging(args.run_name)
    # 設定訓練相關參數
    device = args.device
    dataloader = get_data(args)
    # 初始化模型、優化器、損失函數、Diffusion 模型、Tensorboard Logger、學習率調度器
    model = UNet().to(device)
    optimizer = Lion(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device, noise_steps=args.noise_steps)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 預設最佳 epoch 的 loss 為 100
    best_loss = 100
    # Early stop 相關變數
    early_stop_counter = 0
    last_epoch_loss = 100
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        epoch_running_loss = None
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            # 前向傳遞獲得帶噪聲的圖像和噪聲
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            # 網路進行噪聲的預測
            predicted_noise = model(x_t, t)
            # 計算損失函數
            loss = mse(noise, predicted_noise)
            # 計算訓練 epoch 的 running loss
            epoch_running_loss = loss.item() if epoch_running_loss is None else loss.item() * .9 + epoch_running_loss * .1
            # 更新梯度相關
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新進度條
            pbar.set_postfix(MSE=epoch_running_loss)
        # 更新學習率
        lr_scheduler.step()
        logger.add_scalar("Loss", epoch_running_loss, epoch)
        # 每 10 個 epochs 儲存模型
        if epoch % 10 == 0:
            sampled_images = diffusion.sample(model, n=16)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"{epoch}.pt"))
        # 本次 epoch 的 loss 比最佳 loss 還要低, 則儲存模型
        if epoch_running_loss < best_loss:
            best_loss = epoch_running_loss
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"best.pt"))
        # 透過確認當前 epoch 的 loss 是否比上一個 epoch 的 loss 還要大, 來判斷是否提早停止訓練
        early_stop_counter = early_stop_counter + 1 if epoch_running_loss > last_epoch_loss else 0
        if early_stop_counter >= 10:
            logging.info("損失函數已經連續 10 個 epochs 都沒有下降, 提早停止訓練")
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"last.pt"))
            break


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 1000
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"G:\dataset\CelebA"
    args.device = "cuda"
    args.lr = 3e-4
    args.num_sample = 2000
    args.noise_steps = 700
    train(args)


if __name__ == '__main__':
    launch()
