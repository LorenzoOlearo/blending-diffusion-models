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
from pipelines.blended_diffusion_pipeline import BlendedDiffusionPipeline
from pipelines.blended_in_unet_pipeline import BlendedInUnetPipeline
from pipelines.blended_interpolated_prompts_pipeline import BlendedInterpolatedPromptsPipeline
from pipelines.blended_alternate_unet_prompts_pipeline import BlendedAlternateUnetPipeline
from models.blended_unet import BlendedUNet2DConditionModel


def main():
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(prog="Blending Diffusion Models")
    parser.add_argument("config_path", type=str, help="Path to the config file", default="config.json")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it exists")
    args = parser.parse_args()
    
    device = "cuda:1"
    
    config = utils.read_config(args.config_path)
    
    vae = AutoencoderKL.from_pretrained(config["model_id"], subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(config["model_id"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config["model_id"], subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(config["model_id"], subfolder="unet")
    unet_base = UNet2DConditionModel.from_pretrained(config["model_id"], subfolder="unet")
    unet_blend = BlendedUNet2DConditionModel.from_pretrained(config["model_id"], subfolder="unet")
    scheduler = UniPCMultistepScheduler.from_pretrained(config["model_id"], subfolder="scheduler")
 
    for blend_method in config["blend_methods"]:
        if blend_method == "blended_diffusion":
            print("Initializing Blended Diffusion Pipeline")
            pipeline = BlendedDiffusionPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                scheduler=scheduler
            ).to(device)
        elif blend_method == "blended_in_unet":
            print("Initializing Blended in UNet Pipeline")
            pipeline = BlendedInUnetPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet_base=unet_base,
                unet_blend=unet_blend,
                scheduler=scheduler
            ).to(device)
        elif blend_method == "blended_interpolated_prompts":
            print("Initializing Blended Interpolated Prompts Pipeline")
            pipeline = BlendedInterpolatedPromptsPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                scheduler=scheduler
            ).to(device)
        elif blend_method == "blended_alternate_unet":
            print("Initializing Blended Alternate UNet Pipeline")
            pipeline = BlendedAlternateUnetPipeline(
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                scheduler=scheduler
            ).to(device)
        else:
            raise ValueError(f"Method {blend_method} not recognized. Available methods: blended_diffusion, blended_in_unet, blended_interpolated_prompts, blended_alternate_unet")
        
        # TEMPORARY: batch_size should be implemented from the latent dimension
        output_paths = []
        for seed in config["seeds"]:
            print(f"Running seed {seed}")
            output_path = utils.make_output_dir(seed, config, blend_method, overwrite=args.overwrite)
            output_paths.append(output_path)
            generator = torch.Generator(device=device).manual_seed(seed)
            # generator = torch.manual_seed(seed)
        
            prompt_1_latents, prompt_2_latents, blend_latents = pipeline(config=config, generator=generator)
        
            prompt_1_images = utils.decode_images(prompt_1_latents, vae)
            prompt_2_images = utils.decode_images(prompt_2_latents, vae)
            blend_images = utils.decode_images(blend_latents, vae)
            
            plots.save_all_outputs(
                config=config,
                prompt_1_images=prompt_1_images,
                prompt_2_images=prompt_2_images,
                blend_images=blend_images,
                output_path=output_path, 
                blend_method=blend_method
            )
            
            utils.save_configuration(args.config_path, output_path)
            
        plots.make_blending_batch_grid(output_paths, blend_method, config) 
    
    
if __name__ == "__main__":
    main()
