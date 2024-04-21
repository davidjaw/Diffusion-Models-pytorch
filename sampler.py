import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DDIM_Sampler(nn.Module):
    def __init__(self, channel, img_size, device, time_step, alpha_bar, ddim_sampling_steps=100, eta=0, sample_every=5000, fixed_noise=False,
                 calculate_fid=False, num_fid_sample=None, generate_image=True, clip=True, save=False):
        """
        Denoising Diffusion Implicit Models (DDIM), Jiaming Song et al.
        :param ddim_sampling_steps: Total sampling steps for DDIM sampling process. It corresponds to S in DDIM paper.
        Consult section 4.2 Accelerated Generation Processes in DDIM paper.
        :param eta: Hyperparameter to control the stochasticity, see (16) in DDIM paper.
        0: deterministic(DDIM) , 1: fully stochastic(DDPM)
        :param sample_every: The interval for calling this DDIM sampler. It is only valid during training.
        For example if sample_every=5000 then during training, every 5000steps, trainer will call this DDIM sampler.
        :param fixed_noise: If set to True, then this Sampler will always use same starting noise for image generation.
        :param calculate_fid: Whether to calculate FID score for this sampler.
        :param num_fid_sample: # of generating samples for FID calculation.
        If calculate_fid==True and num_fid_sample==None, then it will automatically set # of generating image to the
        total number of image in original dataset.
        :param generate_image: Whether to save the generated image to folder.
        :param clip: [True, False, 'both'] 'both' will sample twice for clip==True and clip==False.
        Details in ddim_p_sample function.
        :param save: Whether to save the diffusion model based on the FID score calculated by this sampler.
        So calculate_fid must be set to True, if you want to set this parameter to be True.
        """
        super().__init__()
        self.ddim_steps = ddim_sampling_steps
        self.eta = eta
        self.sample_every = sample_every
        self.fixed_noise = fixed_noise
        self.calculate_fid = calculate_fid
        self.num_fid_sample = num_fid_sample
        self.generate_image = generate_image
        self.channel = channel
        self.image_size = img_size
        self.device = device
        self.clip = clip
        self.save = save
        self.sampler_name = None
        self.save_path = None
        ddpm_steps = time_step
        assert self.ddim_steps <= ddpm_steps, 'DDIM sampling step must be smaller or equal to DDPM sampling step'
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if self.save:
            assert self.calculate_fid is True, 'To save model based on FID score, you must set [calculate_fid] to True'
        self.register_buffer('best_fid', torch.tensor([1e10], dtype=torch.float32))

        alpha_bar = alpha_bar
        # One thing you mush notice is that although sampling time is indexed as [1,...T] in paper,
        # since in computer program we index from [0,...T-1] rather than [1,...T],
        # value of tau ranges from [-1, ...T-1] where t=-1 indicate initial state (Data distribution)

        # [tau_1, tau_2, ... tau_S] sec 4.2
        self.register_buffer('tau', torch.linspace(-1, ddpm_steps - 1, steps=self.ddim_steps + 1, dtype=torch.long)[1:])

        alpha_tau_i = alpha_bar[self.tau]
        alpha_tau_i_min_1 = F.pad(alpha_bar[self.tau[:-1]], pad=(1, 0), value=1.)  # alpha_0 = 1

        # (16) in DDIM
        self.register_buffer('sigma', eta * (((1 - alpha_tau_i_min_1) / (1 - alpha_tau_i) *
                                              (1 - alpha_tau_i / alpha_tau_i_min_1)).sqrt()))
        # (12) in DDIM
        self.register_buffer('coeff', (1 - alpha_tau_i_min_1 - self.sigma ** 2).sqrt())
        self.register_buffer('sqrt_alpha_i_min_1', alpha_tau_i_min_1.sqrt())

        assert self.coeff[0] == 0.0 and self.sqrt_alpha_i_min_1[0] == 1.0, 'DDIM parameter error'

    @torch.inference_mode()
    def ddim_p_sample(self, model, xt, i, clip=True):
        """
        Sample x_{tau_(i-1)} from p(x_{tau_(i-1)} | x_{tau_i}), consult (56) in DDIM paper.
        Calculation is done using (12) in DDIM paper where t-1 has to be changed to tau_(i-1) and t has to be
        changed to tau_i in (12), for accelerated generation process where total # of de-noising step is S.

        :param model: Diffusion model
        :param xt: noisy image at time step tau_i
        :param i: i is the index of array tau which is an sub-sequence of [1, ..., T] of length S. See sec. 4.2
        :param clip: Like in GaussianDiffusion p_sample, we can clip(or clamp) the predicted x_0 to -1 ~ 1
        for better sampling result. If you see (12) in DDIM paper, sampling x_(t-1) depends on epsilon_theta which is
        U-net network predicted noise at time step t. If we want to clip the "predicted x0", we have to
        re-calculate the epsilon_theta to make "predicted x0" lie in -1 ~ 1. This is exactly what is going on
        if you set clip==True.
        :return: de-noised image at time step tau_(i-1)
        """
        t = self.tau[i]
        batched_time = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
        pred_noise = model.unet(xt, batched_time)  # corresponds to epsilon_{theta}
        x0 = model.sqrt_recip_alpha_bar[t] * xt - model.sqrt_recip_alpha_bar_min_1[t] * pred_noise
        if clip:
            x0.clamp_(-1., 1.)
            pred_noise = (model.sqrt_recip_alpha_bar[t] * xt - x0) / model.sqrt_recip_alpha_bar_min_1[t]

        # x0 corresponds to "predicted x0" and pred_noise corresponds to epsilon_theta(xt) in (12) DDIM
        # Thus self.coeff[i] * pred_noise corresponds to "direction pointing to xt" in (12)
        mean = self.sqrt_alpha_i_min_1[i] * x0 + self.coeff[i] * pred_noise
        noise = torch.randn_like(xt) if i > 0 else 0.
        # self.sigma[i] * noise corresponds to "random noise" in (12)
        x_t_minus_1 = mean + self.sigma[i] * noise
        return x_t_minus_1

    @torch.inference_mode()
    def sample(self, diffusion_model, batch_size, noise=None, return_all_timestep=False, clip=True, min1to1=False):
        """
        :param diffusion_model: Diffusion model
        :param batch_size: # of image to generate.
        :param noise: If set to True, then this Sampler will always use same starting noise for image generation.
        :param return_all_timestep: Whether to return all images during de-noising process. So it will return the
        images from time step tau_S ~ time step tau_0
        :param clip: See ddim_p_sample function
        :return: Generated image of shape (b, 3, h, w) if return_all_timestep==False else (b, S, 3, h, w)
        """
        clip = clip if clip is not None else self.clip
        xT = torch.randn([batch_size, self.channel, self.image_size, self.image_size], device=self.device) \
            if noise is None else noise.to(self.device)
        denoised_intermediates = [xT]
        xt = xT
        for i in tqdm(reversed(range(0, self.ddim_steps)), desc='DDIM Sampling', total=self.ddim_steps, leave=False):
            x_t_minus_1 = self.ddim_p_sample(diffusion_model, xt, i, clip)
            denoised_intermediates.append(x_t_minus_1)
            xt = x_t_minus_1

        images = xt if not return_all_timestep else torch.stack(denoised_intermediates, dim=1)
        # images = (images + 1.0) * 0.5  # scale to 0~1
        images.clamp_(min=-1.0, max=1.0)
        if not min1to1:
            images.sub_(-1.0).div_(2.0)
        return images