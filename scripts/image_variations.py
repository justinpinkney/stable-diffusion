from io import BytesIO
import os
from contextlib import nullcontext

import fire
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms
import requests

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def load_im(im_path):
    if im_path.startswith("http"):
        response = requests.get(im_path)
        response.raise_for_status()
        im = Image.open(BytesIO(response.content))
    else:
        im = Image.open(im_path).convert("RGB")
    tforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    inp = tforms(im).unsqueeze(0)
    return inp*2-1

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples,1,1)

            if scale != 1.0:
                uc = torch.zeros_like(c)
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

def main(
    im_path="data/example_conditioning/superresolution/sample_0.jpg",
    ckpt="models/ldm/stable-diffusion-v1/sd-clip-vit-l14-img-embed_ema_only.ckpt",
    config="configs/stable-diffusion/sd-image-condition-finetune.yaml",
    outpath="im_variations",
    scale=3.0,
    h=512,
    w=512,
    n_samples=4,
    precision="fp32",
    plms=True,
    ddim_steps=50,
    ddim_eta=1.0,
    device_idx=0,
    ):

    device = f"cuda:{device_idx}"
    input_im = load_im(im_path).to(device)
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    if plms:
        sampler = PLMSSampler(model)
        ddim_eta = 0.0
    else:
        sampler = DDIMSampler(model)

    os.makedirs(outpath, exist_ok=True)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta)
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        Image.fromarray(x_sample.astype(np.uint8)).save(
            os.path.join(sample_path, f"{base_count:05}.png"))
        base_count += 1

if __name__ == "__main__":
    fire.Fire(main)
