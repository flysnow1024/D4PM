import numpy as np
import torch
import torch.nn as nn
from functools import partial
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DDPM(nn.Module):
    def __init__(self, base_model, config, device, conditional=True):
        super().__init__()
        self.device = device
        self.model = base_model
        self.config = config
        self.device = device
        self.conditional = conditional

        self.loss_func = nn.L1Loss(reduction='sum').to(device)

        self.config_diff = config["diffusion"]

        self.set_new_noise_schedule(device)

    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quadratic":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        elif schedule == "cosine":
            t = torch.linspace(0, n_timesteps, n_timesteps + 1)
            alphas_cumprod = torch.cos((t / n_timesteps + 0.008) * torch.pi / 2) ** 2
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = betas / betas.max() * (end - start) + start
        elif schedule == "polynomial":
            betas = (torch.linspace(start ** (1 / 3), end ** (1 / 3), n_timesteps) ** 3)
        elif schedule == "exponential":
            betas = torch.logspace(torch.log10(torch.tensor(start)), torch.log10(torch.tensor(end)), n_timesteps)
        return betas

    def set_new_noise_schedule(self, device):

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = self.make_beta_schedule(schedule=self.config_diff["schedule"],
                                        n_timesteps=self.config_diff["num_steps"],
                                        start=self.config_diff["beta_start"], end=self.config_diff["beta_end"])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
               self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, label, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        label = label.view(batch_size, 1)
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(x, condition_x, noise_level, label))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, label, clip_denoised=False, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, label=label, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, label, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.config_diff['num_steps'] // 10))
        if not self.conditional:
            shape = x_in
            cur_x = torch.randn(shape, device=device)
            ret_x = cur_x
            for i in reversed(range(0, self.config_diff['num_steps'])):
                cur_x = self.p_sample(cur_x, i)
                if i % sample_inter == 0:
                    ret_x = torch.cat([ret_x, cur_x], dim=0)
        else:
            x = x_in
            shape = x.shape
            cur_x = torch.randn(shape, device=device)
            ret_x = [cur_x]
            for i in reversed(range(0, self.config_diff['num_steps'])):
                cur_x = self.p_sample(cur_x, i, label, condition_x=x)
                if i % sample_inter == 0:
                    ret_x.append(cur_x)

        if continous:
            return ret_x
        else:
            return ret_x[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, shape=[1, 512], continous=False):
        return self.p_sample_loop((batch_size, shape[0], shape[1]), continous)

    @torch.no_grad()
    def denoising(self, x_in, label, continous=False):
        return self.p_sample_loop(x_in, label, continous)

    @torch.no_grad()
    def joint_denoising(self, y_observed, label, lambda_dc=0.5, gamma=1.0, eta=0.8, continous=False):
        """
        Joint posterior sampling function, assuming the observation model: y = x + gamma * h

        Args:
            y_observed (Tensor): The observed signal containing artifacts, with shape [B, C, L].
            lambda_dc (float): Weight coefficient for the data consistency term.
            gamma (float): Scaling factor for the artifact component (used for non-uniform artifact strength).
            continous (bool): Whether to return the full sampling trajectory.

        Returns:
            x0_pred (Tensor): The final predicted clean signal.
            h0_pred (Tensor): The final predicted artifact signal.
        """
        device = y_observed.device
        T = self.config_diff["num_steps"]
        B, C, L = y_observed.shape
        label = label.view(B, 1)

        x_t = torch.randn_like(y_observed).to(device)
        h_t = torch.randn_like(y_observed).to(device)

        ret_x, ret_h = [x_t], [h_t]

        for i in reversed(range(T)):
            t = i
            noise_level = torch.tensor(self.sqrt_alphas_cumprod_prev[i + 1]).repeat(B, 1).to(device).float()

            eps_x = self.model(x_t, y_observed, noise_level, label)  # xt
            eps_h = self.model_h(h_t, y_observed, noise_level, label)  # xt'

            # Step 2: Reverse x0, h0
            x0_pred = self.predict_start_from_noise(x_t, t, eps_x)
            h0_pred = self.predict_start_from_noise(h_t, t, eps_h)

            residual = y_observed - (x0_pred + gamma * h0_pred)
            
            # Data consistency constraint
            x0_pred = x0_pred + lambda_dc * residual * eta
            h0_pred = h0_pred + lambda_dc * residual * (1 - eta) / gamma

            mean_x, logvar_x = self.q_posterior(x0_pred, x_t, t)
            mean_h, logvar_h = self.q_posterior(h0_pred, h_t, t)

            noise = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)
            x_t = mean_x + torch.exp(0.5 * logvar_x) * noise
            h_t = mean_h + torch.exp(0.5 * logvar_h) * noise

            if continous:
                ret_x.append(x_t)
                ret_h.append(h_t)

        if continous:
            return ret_x
        else:
            return x0_pred



    def q_sample_loop(self, x_start, continous=False):
        sample_inter = (1 | (self.config_diff['num_steps'] // 10))
        ret_x = [x_start]
        cur_x = x_start
        for t in range(1, self.config_diff['num_steps'] + 1):
            B, C, L = cur_x.shape
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t - 1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=B
                )
            ).to(cur_x.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                B, -1)

            noise = torch.randn_like(cur_x)
            cur_x = self.q_sample(
                x_start=cur_x, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1), noise=noise)
            if t % sample_inter == 0:
                ret_x.append(cur_x)
        if continous:
            return ret_x
        else:
            return ret_x[-1]

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def p_losses(self, x_in, y_in, label, noise=None):
        # x_in: clean signal
        # y_in: noisy signal as condition
        x_start = x_in
        B, C, L = x_start.shape
        t = np.random.randint(1, self.config_diff['num_steps'] + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=B
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            B, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1), noise=noise)

        label = label.view(B, 1)

        if not self.conditional:
            x_recon = self.model(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.model(x_noisy, y_in, continuous_sqrt_alpha_cumprod, label)

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, y, label, *args, **kwargs):
        return self.p_losses(x, y, label, *args, **kwargs)


class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict