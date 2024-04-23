import torch
from ddpm import Diffusion
from utils import get_data, save_images
from network import UNet
import os


if __name__ == '__main__':
    # 設定推測相關參數
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/sample'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sample_img_num = 20
    img_num_per_sample = 32
    # 載入模型權重
    model = UNet().to(device)
    model_weights = 'models/DDPM_Uncondtional/980.pt'
    model.load_state_dict(torch.load(model_weights))
    # 初始化 Diffusion sampler, 依照訓練時的參數設定
    kwargs = {
        'img_size': 64,
        'device': device,
        'noise_steps': 700
    }
    diffusion = Diffusion(**kwargs)
    for i in range(sample_img_num):
        # 進行推測並儲存結果
        with torch.no_grad():
            sampled_images = diffusion.sample(model, n=img_num_per_sample)
        save_images(sampled_images, os.path.join(output_dir, f"{i:02d}.jpg"))

