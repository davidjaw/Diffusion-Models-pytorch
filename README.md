# Diffusion Models

This is an easy-to-understand implementation of diffusion models within 100 lines of code. Different from other implementations, this code doesn't use the lower-bound formulation for sampling and strictly follows Algorithm 1 from the [DDPM](https://arxiv.org/pdf/2006.11239.pdf) paper, which makes it extremely short and easy to follow. 

<a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">
   <img alt="Qries" src="https://user-images.githubusercontent.com/61938694/191407922-f613759e-4bea-4ac9-9135-d053a6312421.jpg"
   width="300">
</a>

<a href="https://www.youtube.com/watch?v=TBCRlnwJtZU">
   <img alt="Qries" src="https://user-images.githubusercontent.com/61938694/191407849-6d0376c7-05b2-43cd-a75c-1280b0e33af1.png"
   width="300">
</a>

<hr>

## Train a Diffusion Model on your own data:
### Unconditional Training
1. (optional) Configure Hyperparameters in ```ddpm.py```
2. Set path to dataset in ```ddpm.py```
3. ```python ddpm.py```

## Sampling
The following examples show how to sample images using the models trained in the video on the [Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures). You can download the checkpoints for the models [here](https://drive.google.com/drive/folders/1beUSI-edO98i6J9pDR67BKGCfkzUL5DX?usp=sharing).
Examples also showed in `inference.py` file.
### Unconditional Model
```python
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("unconditional_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    plot_images(x)
```
<hr>

* This is a simplified version of the forked repo [here](https://github.com/dome272/Diffusion-Models-pytorch)
* A more advanced version of this code can be found [here](https://github.com/tcapelle/Diffusion-Models-pytorch) by [@tcapelle](https://github.com/tcapelle). It introduces better logging, faster & more efficient training and other nice features and is also being followed by a nice [write-up](https://wandb.ai/capecape/train_sd/reports/Training-a-Conditional-Diffusion-model-from-scratch--VmlldzoyODMxNjE3).
