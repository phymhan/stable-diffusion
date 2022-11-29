import argparse, os, sys, glob
import re
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torch.nn.functional as F
from torchvision import transforms

from ldm.util import instantiate_from_config

import pdb
st = pdb.set_trace


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
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

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        # nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--prompt_inv",
        type=str,
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps_inv",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=3,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale_inv",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
        # nargs="+",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")
    parser.add_argument("--ckpt_orig_path", type=str, default="download/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument('--name_as_suffix', action="store_true")
    parser.add_argument("--cond_betas1", type=str, default="0.5,0.5", help="single score guidance")
    parser.add_argument("--cond_betas2", type=str, default="0.0,1.0", help="no single score guidance")
    parser.add_argument("--model_based_guidance", action="store_true")
    parser.add_argument("--range_t", type=int, default=-1, help="only perform single score guidance when t >= range_t")
    parser.add_argument("--start_t", type=float, default=-1)
    parser.add_argument("--ref_img_path", type=str, default=None)
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    sys.path.append(os.getcwd())
    
    seed_everything(opt.seed)
    
    # NOTE: preprocess opt before add into extra_config
    if opt.ckpt is not None:
        opt.ckpt = re.split(",|:|;", opt.ckpt)

    opt.cond_betas1 = [float(s) for s in opt.cond_betas1.split(',')]  # NOTE: always assume the first is single score
    opt.cond_betas2 = [float(s) for s in opt.cond_betas2.split(',')]
    assert len(opt.cond_betas1) == len(opt.cond_betas2)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(
        config,
        opt.ckpt[-1],
    )
    #model.embedding_manager.load(opt.embedding_path)
    model.extra_config = vars(opt)  # NOTE: added for convenience
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.name_as_suffix:
        opt.suffix = "+".join([s.split('/')[-3] for s in opt.ckpt]) + opt.suffix

    from ldm.models.diffusion.ddim2 import DDIM2Sampler
    sampler = DDIM2Sampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # NOTE: load reference image
    ref_img = None
    model_kwargs = {}
    assert os.path.exists(opt.ref_img_path)
    if opt.ref_img_path:
        transform = transforms.Compose(
            [
                transforms.Resize((opt.H, opt.W)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        ref_img = transform(Image.open(opt.ref_img_path).convert('RGB')).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        encoder_posterior = model.encode_first_stage(ref_img)
        ref_img_latent = model.get_first_stage_encoding(encoder_posterior).detach()
        if opt.start_t > 0:  # NOTE: SDEdit start from q_sample
            ts = torch.full((batch_size,), opt.start_t, device=device, dtype=torch.long)
            start_code = model.q_sample(ref_img_latent, ts)
        x0 = ref_img_latent

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    uc = None
                    if opt.scale_inv != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if opt.prompt_inv is not None:
                        prompt1 = batch_size * [opt.prompt_inv]
                        c_inv = model.get_learned_conditioning(prompt1)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    sample_latents, intermediates = sampler.ddim_inversion(
                        S=opt.ddim_steps_inv,
                        conditioning=c_inv,
                        batch_size=opt.n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale_inv,
                        unconditional_conditioning=uc,
                        eta=0,
                        x0=x0,
                        **model_kwargs,
                    )
                    sample_latents_x = model.decode_first_stage(sample_latents)
                    sample_latents_x = torch.clamp((sample_latents_x + 1.0) / 2.0, min=0.0, max=1.0)

                    if opt.prompt is not None:
                        if opt.model_based_guidance:
                            prompt2 = [s.strip() for s in opt.prompt.split('[SEP]')]
                        else:
                            prompt2 = batch_size * [opt.prompt]
                        c = model.get_learned_conditioning(prompt2)
                    else:
                        c = c_inv
                    uc = model.get_learned_conditioning(batch_size * [""])
                    recon_samples, recon_intermediates = sampler.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=sample_latents,
                        model_based_CFG=opt.model_based_guidance,
                        **model_kwargs,
                    )
                    # save_image(model.decode_first_stage(recon_samples)/2+0.5, 'test_decode_1_2.png')
                    recon_samples_x = model.decode_first_stage(recon_samples)
                    recon_samples_x = torch.clamp((recon_samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if True:
                    sample_orig_x = model.decode_first_stage(x0)/2+0.5
                    # additionally, save as grid
                    grid = torch.cat([sample_orig_x, sample_latents_x, recon_samples_x], dim=0)
                    grid = make_grid(grid, nrow=n_rows)
                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    prompt1 = opt.prompt_inv
                    prompt2 = opt.prompt if opt.prompt else 'SAME'
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'Org-Inv-Rec_[PROMPT1]_{prompt1.replace(" ", "-")}_[PROMPT2]_{prompt2.replace(" ", "-")}_[SUFFIX]_{opt.suffix}-{grid_count:04}.jpg'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f"Time elapsed {toc-tic:.2f}s\nEnjoy.")


if __name__ == "__main__":
    main()
