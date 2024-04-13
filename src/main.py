"""
Blending Stable Diffusion

Author: Lorenzo Olearo
TODO: Test other schedulers
"""


import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline
from diffusers import UniPCMultistepScheduler

from prompts import Prompt, Blending
import plots
import utils



def reverse(latents, scheduler, unet, text_embeddings, guidance_scale):
    latent = latents[0]
    for t in tqdm(scheduler.timesteps):
        # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes
        latent_model_input = torch.cat([latent] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        
        # Predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_t-1
        latent = scheduler.step(noise_pred, t, latent).prev_sample
        latents.append(latent)
        
    return latents


def reverse_limits(from_t, to_t, latents, scheduler, unet, text_embeddings, guidance_scale):
    
    for t in tqdm(range(from_t, to_t)):
        latent_model_input = torch.cat([latents[t-from_t]] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        
        with torch.no_grad():
            noise_pred = unet(latent_model_input, scheduler.timesteps[t], encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latent = scheduler.step(noise_pred, scheduler.timesteps[t], latents[t-from_t]).prev_sample
        latents.append(latent)
        
    return latents


def _decode_image(latent, vae):
    latent = 1 / 0.18215 * latent
    with torch.no_grad():
        image = vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    
    return image


def decode_images(latents, vae):
    decoded_images = []
    for latent in latents:
        decoded_image = _decode_image(latent, vae)
        decoded_images.append(decoded_image)
        
    return decoded_images


def main():
    
    parser = argparse.ArgumentParser(prog="Blending Diffusion Models")
    parser.add_argument("config_path", type=str, help="Path to the config file", default="config.json")
    args = parser.parse_args()
    
    device = "cuda:1"
    
    seed, prompt_1_config, prompt_2_config, blending_config  = utils.read_config(args.config_path)
    prompt_1 = Prompt(prompt_1_config, device=device, shared=blending_config["shared"]).to(device)
    prompt_2 = Prompt(prompt_2_config, device=device, shared=blending_config["shared"]).to(device)
    
    blend = Blending(blending_config, device)
    generator = torch.manual_seed(seed)
    batch_size = 1
    
    output_path = utils.make_output_dir(seed, prompt_1_config, prompt_2_config, blending_config)
    
    latents = []
    latent_shape = (batch_size, blend.unet.config.in_channels, blend.height // blend.latent_scale, blend.width // blend.latent_scale)
   
    latent = torch.randn(latent_shape, generator=generator)
    latent = latent * blend.scheduler.init_noise_sigma
    latent = latent.to(device)
    latents.append(latent)
    
    latents_1 = reverse_limits(
        from_t = 0,
        to_t = 25,
        latents = latents.copy(),
        scheduler = prompt_1.scheduler,
        unet = prompt_1.unet,
        text_embeddings = prompt_1.get_text_embeddings(),
        guidance_scale = prompt_1.guidance_scale 
    )
    
    latents_2 = reverse_limits(
        from_t = 0,
        to_t = 25,
        latents = latents.copy(),
        scheduler = prompt_2.scheduler,
        unet = prompt_2.unet,
        text_embeddings = prompt_2.get_text_embeddings(),
        guidance_scale = prompt_2.guidance_scale 
    )
    
    decoded_images_1 = decode_images(latents=latents_1, vae=prompt_1.vae)
    decoded_images_2 = decode_images(latents=latents_2, vae=prompt_2.vae)
   
    # decoded_images_1[-1].save(f"{output_path}/final_image-{prompt_1.prompt}.png")
    # decoded_images_2[-1].save(f"{output_path}/final_image-{prompt_2.prompt}.png")
    # decoded_images_1[blend.from_timestep].save(f"{output_path}/intermediate-{prompt_1.prompt}-timestep-{blend.from_timestep}.png")
    plots.save_image(decoded_images_1[-1], f"final_image-{prompt_1.prompt}", output_path)
    plots.save_image(decoded_images_2[-1], f"final_image-{prompt_2.prompt}", output_path)
    plots.save_image(decoded_images_1[blend.from_timestep], f"intermediate-{prompt_1.prompt}-timestep-{blend.from_timestep}", output_path)
    
    latents_blend=reverse_limits(
        from_t = blend.from_timestep,
        to_t = blend.to_timestep, 
        latents = [latents_1[blend.from_timestep]],
        scheduler = blend.scheduler,
        unet = blend.unet,
        text_embeddings = prompt_2.get_text_embeddings(),
        guidance_scale = blend.guidance_scale
    )
    
    decoded_images_blend = decode_images(latents=latents_blend, vae=blend.vae)
    decoded_images_blend[-1].save(f"{output_path}/final_image-{prompt_1.prompt}-BLEND-{prompt_2.prompt}.png")
    
    plots.make_animation(
        decoded_images=decoded_images_blend,
        prompt=f"{prompt_1.prompt}-BLEND-{prompt_2.prompt}",
        output_path=output_path
    )
    
    plots.make_plots(
        image_1 = decoded_images_1[-1],
        image_2 = decoded_images_2[-1],
        image_blend = decoded_images_blend[-1],
        prompt_1 = prompt_1.prompt,
        prompt_2 = prompt_2.prompt,
        output_path = output_path
    )
    
    utils.save_configuration(args.config_path, output_path)
    
if __name__ == "__main__":
    main()
