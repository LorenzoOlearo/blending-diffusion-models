"""
Blending Stable Diffusion

Author: Lorenzo Olearo
TODO: Test other schedulers
"""

import argparse

import plots as plots
import utils as utils
from prompts import Prompt, Blending


def main():
    parser = argparse.ArgumentParser(prog="Blending Diffusion Models")
    parser.add_argument("config_path", type=str, help="Path to the config file", default="config.json")
    args = parser.parse_args()
    
    device = "cuda:1"
    
    seed, prompt_1_config, prompt_2_config, blending_config  = utils.read_config(args.config_path)
    
    prompt_1 = Prompt(
        prompt_1_config,
        seed=seed,
        shared_pipeline=blending_config["shared_pipeline"],
        shared_generator=blending_config["shared_generator"],
        device=device
    ).to(device)
    
    prompt_2 = Prompt(
        prompt_2_config,
        seed=seed,
        shared_pipeline=blending_config["shared_pipeline"],
        shared_generator=blending_config["shared_generator"],
        device=device
    ).to(device)
    
    blend = Blending(
        blending_config,
        prompts=[prompt_1, prompt_2],
        seed=seed,
        device=device
    )
        
    output_path = utils.make_output_dir(seed, prompt_1_config, prompt_2_config, blending_config)
    
    prompt_1.create_text_embeddings()
    prompt_2.create_text_embeddings()
    
    prompt_1.reverse()
    prompt_2.reverse() 
    
    decoded_images_1 = utils.decode_images(latents=prompt_1.latents, vae=prompt_1.vae)
    decoded_images_2 = utils.decode_images(latents=prompt_2.latents, vae=prompt_2.vae)
   
    plots.save_image(decoded_images_1[-1], f"final_image-{prompt_1.prompt}", output_path)
    plots.save_image(decoded_images_2[-1], f"final_image-{prompt_2.prompt}", output_path)
    plots.save_image(decoded_images_1[blend.from_timestep], f"intermediate-{prompt_1.prompt}-timestep-{blend.from_timestep}", output_path)
    
    blend.reverse_limits(base_latent=prompt_1.latents[blend.from_timestep])
    
    decoded_images_blend = utils.decode_images(latents=blend.latents, vae=blend.vae)
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
        output_path = output_path,
        p1_t = prompt_1.timesteps,
        p2_t = prompt_2.timesteps,
        blending_from_t = blend.from_timestep,
        blending_to_t = blend.to_timestep
    )
    
    utils.save_configuration(args.config_path, output_path)
    
if __name__ == "__main__":
    main()
