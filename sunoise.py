import comfy.samplers
import comfy.model_patcher
from comfy.k_diffusion.sampling import get_ancestral_step, to_d, BrownianTreeNoiseSampler

import torch
import numpy as np
from tqdm.auto import trange


def su_noise_sampler(x, _seed, noise_type):
    def scaled_uniform_noise_multires(sigma_down):
        range_limit = sigma_down.item()
        noise_batch = []
        noise_channels = []
        noise_stack = []
        noise_generator = torch.Generator(device='cpu')
        seed = _seed + int(1000*range_limit)
        noise_generator.manual_seed(seed)
        latent_size = torch.tensor(x.size()[2:])
        for i in range(x.size()[0]): # batch
            noise_channels = []
            for j in range(x.size()[1]): # channels
                noise_stack = []
                for f in (1, 2):
                    noise = torch.rand(*((latent_size/f).to(dtype=torch.int32).tolist()), generator=noise_generator, dtype=torch.float32, device="cpu")
                    noise = torch.unsqueeze(noise, 0)
                    noise = torch.unsqueeze(noise, 0)
                    noise = torch.nn.functional.interpolate(noise, size=x.size()[2:], mode='nearest-exact')
                    noise = torch.squeeze(noise, (0, 1))
                    noise_stack.append(noise)
                noise_stack = torch.stack(noise_stack, 0)
                noise_channels_multires = torch.sum(noise_stack, dim=0, keepdim=False)
                scaled_noise = ((noise_channels_multires-noise_channels_multires.min())*(2*range_limit/(noise_channels_multires.max()-noise_channels_multires.min()))) - range_limit
                noise_channels.append(scaled_noise)
            scaled_noise_channels = torch.stack(noise_channels, 0)
            noise_batch.append(scaled_noise_channels)
        scaled_noise_batch = torch.stack(noise_batch, 0)
        scaled_noise_batch = scaled_noise_batch.to(device=x.device, dtype=x.dtype)
        return scaled_noise_batch

    def scaled_uniform_noise(sigma_down):
        range_limit = sigma_down.item()
        noise_batch = []
        noise_channels = []
        noise_generator = torch.Generator(device='cpu')
        seed = _seed + int(1000*range_limit)
        noise_generator.manual_seed(seed)
        for i in range(x.size()[0]): # batch
            noise_channels = []
            for j in range(x.size()[1]): # channels
                noise = torch.rand(x.size()[2:], generator=noise_generator, dtype=torch.float32, device="cpu")
                scaled_noise = (-1*range_limit) + (2*range_limit*noise)
                noise_channels.append(scaled_noise)
            scaled_noise_channels = torch.stack(noise_channels, 0)
            noise_batch.append(scaled_noise_channels)
        scaled_noise_batch = torch.stack(noise_batch, 0)
        scaled_noise_batch = scaled_noise_batch.to(device=x.device, dtype=x.dtype)
        return scaled_noise_batch
    
    if noise_type=='standard':
        return scaled_uniform_noise
    elif noise_type=='multires':
        return scaled_uniform_noise_multires

def prepare_su_noise(latent_image, _seed, noise_inds=None, scale=1.0):
    noise_batch = []
    noise_channels = []
    noise_generator = torch.Generator(device='cpu')
    seed = _seed + int(1000*scale)
    noise_generator.manual_seed(seed)

    if noise_inds is None:
        for i in range(latent_image.size()[0]): # channels
            noise = torch.rand(latent_image.size()[1:], dtype=torch.float32, layout=latent_image.layout, generator=noise_generator, device="cpu")
            scaled_noise = (-1*scale) + (2*scale*noise)
            noise_channels.append(scaled_noise)
        return torch.stack(noise_channels, 0)
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    for i in range(unique_inds[-1]+1):
        for j in range(latent_image.size()[1]): # channels
            noise = torch.rand(latent_image.size()[2:], dtype=torch.float32, layout=latent_image.layout, generator=noise_generator, device="cpu")
            scaled_noise = (-1*scale) + (2*scale*noise)
            noise_channels.append(scaled_noise)
        scaled_noise_channels = torch.stack(noise_channels, 0)
        if i in unique_inds:
            noise_batch.append(scaled_noise_channels)
    noises = [noise_batch[i] for i in inverse]
    noises = torch.stack(noises, 0)
    return noises

def prepare_su_noise_multires(latent_image, _seed, noise_inds=None, scale=1.0):
    noise_batch = []
    noise_channels = []
    noise_generator = torch.Generator(device='cpu')
    seed = _seed + int(1000*scale)
    noise_generator.manual_seed(seed)
    latent_size = torch.tensor(latent_image.size()[1:])
    if noise_inds is None:
        for i in range(latent_image.size()[0]): # channels
            noise_stack = []
            for f in (1, 2):
                noise = torch.rand(*((latent_size/f).to(dtype=torch.int32).tolist()), generator=noise_generator, dtype=torch.float32, device="cpu")
                noise = torch.unsqueeze(noise, 0)
                noise = torch.unsqueeze(noise, 0)
                noise = torch.nn.functional.interpolate(noise, size=latent_image.size()[1:], mode='nearest-exact')
                noise = torch.squeeze(noise, (0, 1))
                noise_stack.append(noise)
            noise_stack = torch.stack(noise_stack, 0)
            noise_multires = torch.sum(noise_stack, dim=0, keepdim=False)
            scaled_noise = ((noise_multires-noise_multires.min())*(2*scale/(noise_multires.max()-noise_multires.min()))) - scale
            noise_channels.append(scaled_noise)
        return torch.stack(noise_channels, 0)
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    for i in range(unique_inds[-1]+1):
        for j in range(latent_image.size()[1]): # channels
            noise_stack = []
            for f in (1, 2):
                noise = torch.rand(*((latent_size/f).to(dtype=torch.int32).tolist()), generator=noise_generator, dtype=torch.float32, device="cpu")
                noise = torch.unsqueeze(noise, 0)
                noise = torch.unsqueeze(noise, 0)
                noise = torch.nn.functional.interpolate(noise, size=latent_image.size()[1:], mode='nearest-exact')
                noise = torch.squeeze(noise, (0, 1))
                noise_stack.append(noise)
            noise_stack = torch.stack(noise_stack, 0)
            noise_multires = torch.sum(noise_stack, dim=0, keepdim=False)
            scaled_noise = ((noise_multires-noise_multires.min())*(2*scale/(noise_multires.max()-noise_multires.min()))) - scale
            noise_channels.append(scaled_noise)
        scaled_noise_channels = torch.stack(noise_channels, 0)
        if i in unique_inds:
            noise_batch.append(scaled_noise_channels)
    noises = [noise_batch[i] for i in inverse]
    noises = torch.stack(noises, 0)
    return noises


@torch.no_grad()
def sample_euler_ancestral_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_type='standard'):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    # sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, seed, noise_type) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + (noise_sampler(sigma_down) if i%2!=0 else torch.randn_like(x) * s_noise * sigma_up)
    return x

@torch.no_grad()
def sample_euler_ancestral_cfg_pp_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_type='standard'):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    # sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, seed, noise_type) if noise_sampler is None else noise_sampler
    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], temp[0])
        # Euler method
        dt = sigma_down - sigmas[i]
        x = denoised + (d * sigma_down)
        if sigmas[i + 1] > 0:
            x = x + (noise_sampler(sigma_down) if i%2!=0 else torch.randn_like(x) * s_noise * sigma_up)
    return x

@torch.no_grad()
def sample_dpm_2_ancestral_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_type='standard'):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    # sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, seed, noise_type) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + (noise_sampler(sigma_down) if i%2!=0 else torch.randn_like(x) * s_noise * sigma_up)
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_type='standard'):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    # sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, seed, noise_type) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + (noise_sampler(sigma_down) if i%2!=0 else torch.randn_like(x) * s_noise * sigma_up)
    return x

@torch.no_grad()
def sample_dpmpp_sde_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1/2, noise_type='standard'):
    """DPM-Solver++ (stochastic)."""
    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler_bt = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    noise_sampler = su_noise_sampler(x, seed, noise_type) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_2 = t_fn(sd)
            x_2 = (sigma_fn(s_2) / sigma_fn(t)) * x - (t - s_2).expm1() * denoised
            x_2 = x_2 + (noise_sampler(sd) if i%2!=0 else noise_sampler_bt(sigma_fn(t), sigma_fn(s)) * s_noise * su)
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (t - t_next).expm1() * denoised_d
            x = x + (noise_sampler(sd) if i%2!=0 else noise_sampler_bt(sigma_fn(t), sigma_fn(t_next)) * s_noise * su)
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint', noise_type='standard'):
    """DPM-Solver++(2M) SDE."""

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler_bt = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    noise_sampler = su_noise_sampler(x, seed, noise_type) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + (noise_sampler(sigma_down) if i%2!=0 else noise_sampler_bt(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise)

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, noise_type='standard'):
    """DPM-Solver++(3M) SDE."""
    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler_bt = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    noise_sampler = su_noise_sampler(x, seed, noise_type) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + (noise_sampler(sigma_down) if i%2!=0 else noise_sampler_bt(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise)

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


class SamplersSUNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sampler_name": (["euler_ancestral", "euler_ancestral_cfg_pp", "dpm_2_ancestral", "dpmpp_2s_ancestral",
                                       "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde"], ),
                    "noise_type": (['standard', 'multires'], ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name, noise_type):
        s_noise = 1.0
        solver_type = 'heun'
        eta = 1.0
        r = 0.5
        if sampler_name == "euler_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        elif sampler_name == "euler_ancestral_cfg_pp":
            sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_cfg_pp_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        elif sampler_name == "dpm_2_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_dpm_2_ancestral_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        elif sampler_name == "dpmpp_2s_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2s_ancestral_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        elif sampler_name == "dpmpp_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_sde_sun, {"eta": eta, "s_noise": s_noise, "r": r, "noise_type": noise_type}, {})
        elif sampler_name == "dpmpp_2m_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2m_sde_sun, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type, "noise_type": noise_type}, {})
        elif sampler_name == "dpmpp_3m_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_3m_sde_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        return (sampler, )

class SamplersSUNoiseAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sampler_name": (["euler_ancestral", "euler_ancestral_cfg_pp", "dpm_2_ancestral", "dpmpp_2s_ancestral",
                                       "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde"], ),
                    "noise_type": (['standard', 'multires'], ),
                    "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step":0.01, "round": False}),
                    "solver_type": (['midpoint', 'heun'], ),
                    "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name, noise_type, s_noise, solver_type, eta, r):
        if sampler_name == "euler_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        elif sampler_name == "euler_ancestral_cfg_pp":
            sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_cfg_pp_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        elif sampler_name == "dpm_2_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_dpm_2_ancestral_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        elif sampler_name == "dpmpp_2s_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2s_ancestral_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        elif sampler_name == "dpmpp_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_sde_sun, {"eta": eta, "s_noise": s_noise, "r": r, "noise_type": noise_type}, {})
        elif sampler_name == "dpmpp_2m_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2m_sde_sun, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type, "noise_type": noise_type}, {})
        elif sampler_name == "dpmpp_3m_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_3m_sde_sun, {"eta": eta, "s_noise": s_noise, "noise_type": noise_type}, {})
        return (sampler, )

class Noise_SUNoise:
    def __init__(self, seed, scale, noise_type):
        self.seed = seed
        self.scale = scale
        self.noise_type = noise_type

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        if self.noise_type=='standard':
            return prepare_su_noise(latent_image, self.seed, batch_inds, self.scale)
        elif self.noise_type=='multires':
            return prepare_su_noise_multires(latent_image, self.seed, batch_inds, self.scale)

class SUNoiseLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100, "step": 0.01}),
                    "noise_type": (['standard', 'multires'], ),
                     }
                }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self, noise_seed, scale, noise_type):
        return (Noise_SUNoise(noise_seed, scale, noise_type),)


NODE_CLASS_MAPPINGS = {
    "SamplersSUNoise": SamplersSUNoise,
    "SamplersSUNoiseAdvanced": SamplersSUNoiseAdvanced,
    "SUNoiseLatent": SUNoiseLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplersSUNoise": "SamplersSUNoise",
    "SamplersSUNoiseAdvanced": "SamplersSUNoiseAdvanced",
    "SUNoiseLatent": 'SUNoiseLatent',
}
