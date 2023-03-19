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
ckpt = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-pruned.ckpt")
config = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")

device = "cuda:0"
model = load_model_from_config(config, ckpt, device=device, verbose=False)
model = model.to(device).half()

clip_model, preprocess = clip.load("ViT-L/14", device=device)

n_inputs = 5

torch.cuda.empty_cache()

@functools.lru_cache()
def get_url_im(t):
    user_agent = {'User-agent': 'gradio-app'}
    response = requests.get(t, headers=user_agent)
    return Image.open(BytesIO(response.content))

@torch.no_grad()
def get_im_c(im_path, clip_model):
    # im = Image.open(im_path).convert("RGB")
    prompts = preprocess(im_path).to(device).unsqueeze(0)
    return clip_model.encode_image(prompts).float()

@torch.no_grad()
def get_txt_c(txt, clip_model):
    text = clip.tokenize([txt,]).to(device)
    return clip_model.encode_text(text)

def get_txt_diff(txt1, txt2, clip_model):
    return get_txt_c(txt1, clip_model) - get_txt_c(txt2, clip_model)

def to_im_list(x_samples_ddim):
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return ims

@torch.no_grad()
def sample(sampler, model, c, uc, scale, start_code, h=512, w=512, precision="autocast",ddim_steps=50):
    ddim_eta=0.0
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        shape = [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=c.shape[0],
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=start_code)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
    return to_im_list(x_samples_ddim)

def run(*args):

    inps = []
    for i in range(0, len(args)-4, n_inputs):
        inps.append(args[i:i+n_inputs])

    scale, n_samples, seed, steps = args[-4:]
    h = w = 640

    sampler = DDIMSampler(model)
    # sampler = PLMSSampler(model)

    torch.manual_seed(seed)
    start_code = torch.randn(n_samples, 4, h//8, w//8, device=device)
    conds = []

    for b, t, im, s in zip(*inps):
        if b == "Image":
            this_cond = s*get_im_c(im, clip_model)
        elif b == "Text/URL":
            if t.startswith("http"):
                im = get_url_im(t)
                this_cond = s*get_im_c(im, clip_model)
            else:
                this_cond = s*get_txt_c(t, clip_model)
        else:
            this_cond = torch.zeros((1, 768), device=device)
        conds.append(this_cond)
    conds = torch.cat(conds, dim=0).unsqueeze(0)
    conds = conds.tile(n_samples, 1, 1)

    ims = sample(sampler, model, conds, 0*conds, scale, start_code, ddim_steps=steps)
    # return make_row(ims)
    return ims


import gradio as gr
from functools import partial
from itertools import chain

def change_visible(txt1, im1, val):
    outputs = {}
    if val == "Image":
        outputs[im1] = gr.update(visible=True)
        outputs[txt1] = gr.update(visible=False)
    elif val == "Text/URL":
        outputs[im1] = gr.update(visible=False)
        outputs[txt1] = gr.update(visible=True)
    elif val == "Nothing":
        outputs[im1] = gr.update(visible=False)
        outputs[txt1] = gr.update(visible=False)
    return outputs


with gr.Blocks(title="Image Mixer", css=".gr-box {border-color: #8136e2}") as demo:

    gr.Markdown("")
    gr.Markdown(
"""
# Image Mixer

_Created by [Justin Pinkney](https://www.justinpinkney.com) at [Lambda Labs](https://lambdalabs.com/)_

### __Provide one or more images to be mixed together by a fine-tuned Stable Diffusion model.__

![banner-large.jpeg](https://s3.amazonaws.com/moonup/production/uploads/1674039767068-62bd5f951e22ec84279820e8.jpeg)

""")

    btns = []
    txts = []
    ims = []
    strengths = []

    with gr.Row():
        for i in range(n_inputs):
            with gr.Box():
                with gr.Column():
                    btn1 = gr.Radio(
                        choices=["Image", "Text/URL", "Nothing"],
                        label=f"Input {i} type",
                        interactive=True,
                        value="Nothing",
                        )
                    txt1 = gr.Textbox(label="Text or Image URL", visible=False, interactive=True)
                    im1 = gr.Image(label="Image", interactive=True, visible=False, type="pil")
                    strength = gr.Slider(label="Strength", minimum=0, maximum=5, step=0.05, value=1, interactive=True)
    
                    fn = partial(change_visible, txt1, im1)
                    btn1.change(fn=fn, inputs=[btn1], outputs=[txt1, im1])
    
                    btns.append(btn1)
                    txts.append(txt1)
                    ims.append(im1)
                    strengths.append(strength)
    with gr.Row():
        cfg_scale = gr.Slider(label="CFG scale", value=3, minimum=1, maximum=10, step=0.5)
        n_samples = gr.Slider(label="Num samples", value=2, minimum=1, maximum=4, step=1)
        seed = gr.Slider(label="Seed", value=0, minimum=0, maximum=10000, step=1)
        steps = gr.Slider(label="Steps", value=30, minimum=10, maximum=100, step=5)

    with gr.Row():
        submit = gr.Button("Generate")
    output = gr.Gallery().style(grid=[1,2], height="640px")

    inps = list(chain(btns, txts, ims, strengths))
    inps.extend([cfg_scale,n_samples,seed, steps,])
    submit.click(fn=run, inputs=inps, outputs=[output])

    ex = gr.Examples([[
        "Image", "Image", "Text/URL", "Nothing", "Nothing",
        None,None,"wild flowers blooming",None,None,
        "assets/im-examples/vermeer.jpg","assets/im-examples/matisse.jpg",None,None,None,
        1,1,1.8,1,1,
        3.0, 2, 0, 30,
    ]],
    inputs=inps, outputs=[output])

    gr.Markdown(
"""

## Tips

- You can provide between 1 and 5 inputs, these can either be an uploaded image a text prompt or a url to an image file.
- The order of the inputs shouldn't matter, any images will be centre cropped before use.
- Each input has an individual strength parameter which controls how big an influence it has on the output.
- Using only text prompts doesn't work well, make sure there is at least one image or URL to an image.
- The parameters on the bottom row such as cfg scale do the same as for a normal Stable Diffusion model.
- Balancing the different inputs requires tweaking of the strengths, I suggest getting the right balance for a small number of samples and with few steps until you're
happy with the result then increase the steps for better quality.
- Outputs are 640x640 by default.

## How does this work?

This model is based on the [Stable Diffusion Image Variations model](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
but it has been fined tuned to take multiple CLIP image embeddings. During training, up to 5 random crops were taken from the training images and
the CLIP image embeddings were computed, these were then concatenated and used as the conditioning for the model. At inference time we can combine the image
embeddings from multiple images to mix their concepts (and we can also use the text encoder to add text concepts too).

The model was trained on a subset of LAION Improved Aesthetics at a resolution of 640x640 and was trained using 8xA100 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).

""")

demo.launch(share=True)
