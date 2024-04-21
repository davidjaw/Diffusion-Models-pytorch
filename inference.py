import torch
from ddpm import Diffusion
from utils import get_data, save_images
from modules import UNet
import os


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'results/sample'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_weights = 'models/DDPM_Uncondtional/980.pt'
    sample_img_num = 20
    img_num_per_sample = 32
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_weights))
    kwargs = {
        'img_size': 64,
        'device': device,
        'noise_steps': 700
    }
    diffusion = Diffusion(**kwargs)
    for i in range(sample_img_num):
        with torch.no_grad():
            sampled_images = diffusion.sample(model, n=img_num_per_sample)
        save_images(sampled_images, os.path.join(output_dir, f"{i:02d}.jpg"))

