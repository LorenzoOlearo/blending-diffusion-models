"""
Blending Stable Diffusion

Author: Lorenzo Olearo
TODO: Test other schedulers
"""

import argparse
import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler

import plots as plots
import utils as utils
from blended_diffusion_pipeline import BlendedDiffusionPipeline


def main():
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(prog="Blending Diffusion Models")
    parser.add_argument("config_path", type=str, help="Path to the config file", default="config.json")
    args = parser.parse_args()
    
    device = "cuda:1"
    
    config = utils.read_config(args.config_path)
    output_path = utils.make_output_dir(config)
    
    vae = AutoencoderKL.from_pretrained(config["model_id"], subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(config["model_id"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config["model_id"], subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(config["model_id"], subfolder="unet")
    scheduler = UniPCMultistepScheduler.from_pretrained(config["model_id"], subfolder="scheduler")
    
    pipeline = BlendedDiffusionPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler
    ).to(device)
    
    # generator = torch.Generator(device=device).manual_seed(config["seed"])
    generator = torch.manual_seed(config["seed"])
    
    prompt_1_latents, prompt_2_latents, blend_latents = pipeline(config=config, generator=generator)
    
    prompt_1_images = utils.decode_images(prompt_1_latents, vae)
    prompt_2_images = utils.decode_images(prompt_2_latents, vae)
    blen_images = utils.decode_images(blend_latents, vae)
    
    plots.save_all_outputs(
        config=config,
        prompt_1_images=prompt_1_images,
        prompt_2_images=prompt_2_images,
        blend_images=blen_images,
        output_path=output_path
    )
     
    utils.save_configuration(args.config_path, output_path)
    
if __name__ == "__main__":
    main()
