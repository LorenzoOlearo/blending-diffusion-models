"""
Blending Stable Diffusion

Author: Lorenzo Olearo
TODO: Test other schedulers
"""

import torch
import argparse
from PIL import Image
from tqdm.auto import tqdm

from prompts import Prompt, Blending
import plots as plots
import utils as utils


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
    generator = torch.manual_seed(seed)
    prompt_1 = Prompt(prompt_1_config, device=device, generator=generator, shared=blending_config["shared"]).to(device)
    prompt_2 = Prompt(prompt_2_config, device=device, generator=generator, shared=blending_config["shared"]).to(device)
    
    blend = Blending(blending_config, [prompt_1, prompt_2], generator, device)
    batch_size = 1
    
    output_path = utils.make_output_dir(seed, prompt_1_config, prompt_2_config, blending_config)
    
    latents = []
    latent_shape = (batch_size, blend.unet.config.in_channels, blend.height // blend.latent_scale, blend.width // blend.latent_scale)
   
    latent = torch.randn(latent_shape, generator=generator)
    latent = latent * blend.scheduler.init_noise_sigma
    latent = latent.to(device)
    latents.append(latent)
    
    prompt_1.create_text_embeddings()
    prompt_2.create_text_embeddings()
    
    prompt_1.reverse()
    prompt_2.reverse() 
    
    decoded_images_1 = decode_images(latents=prompt_1.latents, vae=prompt_1.vae)
    decoded_images_2 = decode_images(latents=prompt_2.latents, vae=prompt_2.vae)
   
    plots.save_image(decoded_images_1[-1], f"final_image-{prompt_1.prompt}", output_path)
    plots.save_image(decoded_images_2[-1], f"final_image-{prompt_2.prompt}", output_path)
    plots.save_image(decoded_images_1[blend.from_timestep], f"intermediate-{prompt_1.prompt}-timestep-{blend.from_timestep}", output_path)
    
    blend.reverse_limits(base_latent=prompt_1.latents[blend.from_timestep])
    
    decoded_images_blend = decode_images(latents=blend.latents, vae=blend.vae)
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
