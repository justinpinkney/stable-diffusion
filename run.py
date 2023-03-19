import argparse
import cv2
from io import BytesIO
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
import requests
import functools

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.extras import load_model_from_config, load_training_dir
import clip

from PIL import Image

from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(
    repo_id="lambdalabs/image-mixer", filename="image-mixer-pruned.ckpt"
)
config = hf_hub_download(
    repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml"
)

device = "cuda:0"
model = load_model_from_config(config, ckpt, device=device, verbose=False)
model = model.to(device).half()

clip_model, preprocess = clip.load("ViT-L/14", device=device)

n_inputs = 5

torch.cuda.empty_cache()


@functools.lru_cache()
def get_url_im(t):
    user_agent = {"User-agent": "gradio-app"}
    response = requests.get(t, headers=user_agent)
    return Image.open(BytesIO(response.content))


@torch.no_grad()
def get_im_c(im_path, clip_model):
    # im = Image.open(im_path).convert("RGB")
    prompts = preprocess(im_path).to(device).unsqueeze(0)
    return clip_model.encode_image(prompts).float()


@torch.no_grad()
def get_txt_c(txt, clip_model):
    text = clip.tokenize(
        [
            txt,
        ]
    ).to(device)
    return clip_model.encode_text(text)


def get_txt_diff(txt1, txt2, clip_model):
    return get_txt_c(txt1, clip_model) - get_txt_c(txt2, clip_model)


def to_im_list(x_samples_ddim):
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return ims


@torch.no_grad()
def sample(
    sampler,
    model,
    c,
    uc,
    scale,
    start_code,
    h=512,
    w=512,
    precision="autocast",
    ddim_steps=50,
):
    ddim_eta = 0.0
    precision_scope = autocast if precision == "autocast" else nullcontext
    with precision_scope("cuda"):
        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=c.shape[0],
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            x_T=start_code,
        )

        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return to_im_list(x_samples_ddim)


def run(*args):
    inps = []
    for i in range(0, len(args) - 4, n_inputs):
        inps.append(args[i : i + n_inputs])

    scale, n_samples, seed, steps = args[-4:]
    h = w = 640

    sampler = DDIMSampler(model)
    # sampler = PLMSSampler(model)

    torch.manual_seed(seed)
    start_code = torch.randn(n_samples, 4, h // 8, w // 8, device=device)
    conds = []

    for b, t, im, s in zip(*inps):
        if b == "Image":
            this_cond = s * get_im_c(im, clip_model)
        elif b == "Text/URL":
            if t.startswith("http"):
                im = get_url_im(t)
                this_cond = s * get_im_c(im, clip_model)
            else:
                this_cond = s * get_txt_c(t, clip_model)
        else:
            this_cond = torch.zeros((1, 768), device=device)
        conds.append(this_cond)
    conds = torch.cat(conds, dim=0).unsqueeze(0)
    conds = conds.tile(n_samples, 1, 1)

    ims = sample(sampler, model, conds, 0 * conds, scale, start_code, ddim_steps=steps)
    # return make_row(ims)
    return ims

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_1",
        type=str,
        help="path to img 1",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--path_2",
        type=str,
        help="path to img 2",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to output image",
        required=False,
        default="output.jpg",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    img1 = Image.open(args.path_1).convert("RGB")
    img2 = Image.open(args.path_2).convert("RGB")
    inputs = (
        "Image",
        "Image",
        "Nothing",
        "Nothing",
        "Nothing",
        "",
        "",
        "",
        "",
        "",
        img1,
        img2,
        None,
        None,
        None,
        1,
        1,
        1,
        1,
        1,
        3,
        1,
        0,
        30,
    )
    ims = run(*inputs)
    ims[0].save(args.save_path)
