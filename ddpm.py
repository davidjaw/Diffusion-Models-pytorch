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
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images with DDPM....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = Lion(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device, noise_steps=args.noise_steps)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = 100
    # Early stop 相關變數
    early_stop_counter = 0
    last_epoch_loss = 100
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        epoch_running_loss = None
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            epoch_running_loss = loss.item() if epoch_running_loss is None else loss.item() * .9 + epoch_running_loss * .1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=epoch_running_loss)
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
    # device = "cuda"
    # model = UNet().to(device)
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 8)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()
