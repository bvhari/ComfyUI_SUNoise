import comfy.samplers
import comfy.model_patcher
from comfy.k_diffusion.sampling import get_ancestral_step, to_d

import torch
import numpy as np
from tqdm.auto import trange


def su_noise_sampler(x, sigma_min, sigma_max, s_noise, _seed):
    def scaled_uniform_noise(sigma, sigma_next):
        range_limit_sigma = (sigma - sigma_min) / (sigma_max - sigma_min)
        range_limit = s_noise * range_limit_sigma
        range_limit = range_limit.item()
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
    return scaled_uniform_noise

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


@torch.no_grad()
def sample_euler_ancestral_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, sigma_min, sigma_max, s_noise, seed) if noise_sampler is None else noise_sampler
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
            x = x + noise_sampler(sigmas[i], sigmas[i + 1])
    return x

@torch.no_grad()
def sample_euler_ancestral_cfg_pp_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, sigma_min, sigma_max, s_noise, seed) if noise_sampler is None else noise_sampler
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
            x = x + noise_sampler(sigmas[i], sigmas[i + 1])
    return x

@torch.no_grad()
def sample_dpm_2_ancestral_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, sigma_min, sigma_max, s_noise, seed) if noise_sampler is None else noise_sampler
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
            x = x + noise_sampler(sigmas[i], sigmas[i + 1])
    return x

@torch.no_grad()
def sample_dpmpp_2s_ancestral_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, sigma_min, sigma_max, s_noise, seed) if noise_sampler is None else noise_sampler
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
            x = x + noise_sampler(sigmas[i], sigmas[i + 1])
    return x

@torch.no_grad()
def sample_dpmpp_sde_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic)."""
    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, sigma_min, sigma_max, s_noise, seed) if noise_sampler is None else noise_sampler
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
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s))
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next))
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    """DPM-Solver++(2M) SDE."""

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, sigma_min, sigma_max, s_noise, seed) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None
    h = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
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
                x = x + noise_sampler(sigmas[i], sigmas[i + 1])

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde_sun(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE."""
    seed = extra_args.get("seed", None)

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = su_noise_sampler(x, sigma_min, sigma_max, s_noise, seed) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
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
                x = x + noise_sampler(sigmas[i], sigmas[i + 1])

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


class SamplerEulerAncestral_SUN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        return (sampler, )

class SamplerEulerAncestralCFGpp_SUN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_cfg_pp_sun, {"eta": eta, "s_noise": s_noise}, {})
        return (sampler, )

class SamplerDPM2Ancestral_SUN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_dpm_2_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        return (sampler, )

class SamplerDPMPP2SAncestral_SUN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2s_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        return (sampler, )
    
class SamplerDPMPP_SDE_SUN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, r):
        sampler = comfy.samplers.KSAMPLER(sample_dpmpp_sde_sun, {"eta": eta, "s_noise": s_noise, "r": r}, {})
        return (sampler, )

class SamplerDPMPP_2M_SDE_SUN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"solver_type": (['midpoint', 'heun'], ),
                     "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, solver_type, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2m_sde_sun, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type}, {})
        return (sampler, )

class SamplerDPMPP_3M_SDE_SUN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.KSAMPLER(sample_dpmpp_3m_sde_sun, {"eta": eta, "s_noise": s_noise}, {})
        return (sampler, )

class SamplersSUNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sampler_name": (["euler_ancestral", "euler_ancestral_cfg_pp", "dpm_2_ancestral", "dpmpp_2s_ancestral",
                                       "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde"], ),
                    "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name, s_noise):
        solver_type = 'midpoint'
        eta = 1.0
        r = 0.5
        if sampler_name == "euler_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        elif sampler_name == "euler_ancestral_cfg_pp":
            sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_cfg_pp_sun, {"eta": eta, "s_noise": s_noise}, {})
        elif sampler_name == "dpm_2_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_dpm_2_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        elif sampler_name == "dpmpp_2s_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2s_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        elif sampler_name == "dpmpp_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_sde_sun, {"eta": eta, "s_noise": s_noise, "r": r}, {})
        elif sampler_name == "dpmpp_2m_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2m_sde_sun, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type}, {})
        elif sampler_name == "dpmpp_3m_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_3m_sde_sun, {"eta": eta, "s_noise": s_noise}, {})
        return (sampler, )

class SamplersSUNoiseAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sampler_name": (["euler_ancestral", "euler_ancestral_cfg_pp", "dpm_2_ancestral", "dpmpp_2s_ancestral",
                                       "dpmpp_sde", "dpmpp_2m_sde", "dpmpp_3m_sde"], ),
                    "s_noise": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step":0.01, "round": False}),
                    "solver_type": (['midpoint', 'heun'], ),
                    "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name, s_noise, solver_type, eta, r):
        if sampler_name == "euler_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        elif sampler_name == "euler_ancestral_cfg_pp":
            sampler = comfy.samplers.KSAMPLER(sample_euler_ancestral_cfg_pp_sun, {"eta": eta, "s_noise": s_noise}, {})
        elif sampler_name == "dpm_2_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_dpm_2_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        elif sampler_name == "dpmpp_2s_ancestral":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2s_ancestral_sun, {"eta": eta, "s_noise": s_noise}, {})
        elif sampler_name == "dpmpp_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_sde_sun, {"eta": eta, "s_noise": s_noise, "r": r}, {})
        elif sampler_name == "dpmpp_2m_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_2m_sde_sun, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type}, {})
        elif sampler_name == "dpmpp_3m_sde":
            sampler = comfy.samplers.KSAMPLER(sample_dpmpp_3m_sde_sun, {"eta": eta, "s_noise": s_noise}, {})
        return (sampler, )

class Noise_SUNoise:
    def __init__(self, seed, scale):
        self.seed = seed
        self.scale = scale

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return prepare_su_noise(latent_image, self.seed, batch_inds, self.scale)

class SUNoiseLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "scale": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self, noise_seed, scale):
        return (Noise_SUNoise(noise_seed, scale),)


NODE_CLASS_MAPPINGS = {
    "SamplerEulerAncestral_SUN": SamplerEulerAncestral_SUN,
    "SamplerEulerAncestralCFGpp_SUN": SamplerEulerAncestralCFGpp_SUN,
    "SamplerDPM2Ancestral_SUN": SamplerDPM2Ancestral_SUN,
    "SamplerDPMPP2SAncestral_SUN": SamplerDPMPP2SAncestral_SUN,
    "SamplerDPMPP_SDE_SUN": SamplerDPMPP_SDE_SUN,
    "SamplerDPMPP_2M_SDE_SUN": SamplerDPMPP_2M_SDE_SUN,
    "SamplerDPMPP_3M_SDE_SUN": SamplerDPMPP_3M_SDE_SUN,
    "SamplersSUNoise": SamplersSUNoise,
    "SamplersSUNoiseAdvanced": SamplersSUNoiseAdvanced,
    "SUNoiseLatent": SUNoiseLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerEulerAncestral_SUN": "SamplerEulerAncestral_SUN",
    "SamplerEulerAncestralCFGpp_SUN": "SamplerEulerAncestralCFGpp_SUN",
    "SamplerDPM2Ancestral_SUN": "SamplerDPM2Ancestral_SUN",
    "SamplerDPMPP2SAncestral_SUN": "SamplerDPMPP2SAncestral_SUN",
    "SamplerDPMPP_SDE_SUN": "SamplerDPMPP_SDE_SUN",
    "SamplerDPMPP_2M_SDE_SUN": "SamplerDPMPP_2M_SDE_SUN",
    "SamplerDPMPP_3M_SDE_SUN": "SamplerDPMPP_3M_SDE_SUN",
    "SamplersSUNoise": "SamplersSUNoise",
    "SamplersSUNoiseAdvanced": "SamplersSUNoiseAdvanced",
    "SUNoiseLatent": 'SUNoiseLatent',
}
