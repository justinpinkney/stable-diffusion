from contextlib import nullcontext
from functools import partial

import fire
import gradio as gr
import numpy as np
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms

from scripts.image_variations import load_model_from_config


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
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main(
    model,
    device,
    input_im,
    scale=3.0,
    n_samples=4,
    plms=True,
    ddim_steps=50,
    ddim_eta=1.0,
    precision="fp32",
    h=512,
    w=512,
    ):

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im*2-1

    if plms:
        sampler = PLMSSampler(model)
        ddim_eta = 0.0
    else:
        sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta)
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return output_ims


def run_demo(
    device_idx=0,
    ckpt="models/ldm/stable-diffusion-v1/sd-clip-vit-l14-img-embed_ema_only.ckpt",
    config="configs/stable-diffusion/sd-image-condition-finetune.yaml",
    ):

    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    inputs = [
        gr.Image(),
        gr.Slider(0, 25, value=3, step=1, label="cfg scale"),
        gr.Slider(1, 4, value=1, step=1, label="Number images"),
        gr.Checkbox(True, label="plms"),
        gr.Slider(5, 250, value=25, step=5, label="steps"),
    ]
    output = gr.Gallery(label="Generated variations")
    output.style(height="auto", grid=2)

    fn_with_model = partial(main, model, device)
    fn_with_model.__name__ = "fn_with_model"

    demo = gr.Interface(
        fn=fn_with_model,
        title="Stable Diffusion Image Variations",
        description="Generate variations on an input image using a fine-tuned version of Stable Diffision",
        article="TODO",
        inputs=inputs,
        outputs=output,
        )
    # demo.queue()
    demo.launch(share=False, server_name="0.0.0.0")

if __name__ == "__main__":
    fire.Fire(run_demo)
