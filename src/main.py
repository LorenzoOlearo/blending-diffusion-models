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


def create_text_embeddings(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, batch_size: int, device: str):
    batch_size = batch_size
    
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    return text_embeddings


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


def make_animation(decoded_images: list, prompt: str, output_path: str):
    fig, ax = plt.subplots()

    plt.title(prompt[0])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)

    ims = []
    for i in range(len(decoded_images)):
        im = ax.imshow(decoded_images[i], animated=True) 
        if i == 0:
            ax.imshow(decoded_images[i])
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat_delay=2000)
    
    save_path = f"{output_path}/denoising-{prompt}.gif"
    ani.save(save_path)
    
    
def make_plots(image_1, image_2, image_blend, prompt_1, prompt_2, output_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image_1)
    ax[0].set_title(prompt_1)
    ax[0].axis("off")
    
    ax[1].imshow(image_blend)
    ax[1].set_title(f"{prompt_1}-BLEND-{prompt_2}")
    ax[1].axis("off")
    
    ax[2].imshow(image_2)
    ax[2].set_title(prompt_2)
    ax[2].axis("off")
    
    plt.savefig(f"{output_path}/blending-{prompt_1}-BLEND-{prompt_2}.png")
    
    
def read_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if not os.path.exists("./out"):
        os.makedirs("./out")
        
    return config["seed"], config["prompt_1_config"], config["prompt_2_config"], config["blending_config"]


def make_output_dir(seed, prompt_1_config, prompt_2_config, blending_config):
    
    output_path = os.path.join("out", str(seed), f"{prompt_1_config['prompt']}-BLEND-{prompt_2_config['prompt']}-scheduler_{blending_config['scheduler']}-model_{blending_config['model_id'].replace('/', '_')}")
    
    if not os.path.exists("./out"):
        os.makedirs("./out")
       
    if not os.path.exists(os.path.join("out", str(seed))):
        os.makedirs(os.path.join("out", str(seed)))
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    return output_path
    


def main():
    
    parser = argparse.ArgumentParser(prog="Blending Diffusion Models")
    parser.add_argument("config_path", type=str, help="Path to the config file", default="config.json")
    args = parser.parse_args()
    
    seed, prompt_1_config, prompt_2_config, blending_config  = read_config(args.config_path)
    prompt_1 = Prompt(prompt_1_config)
    prompt_2 = Prompt(prompt_2_config)
    blend = Blending(blending_config)
    
    output_path = make_output_dir(seed, prompt_1_config, prompt_2_config, blending_config)
    
    device = "cuda:1"
    generator = torch.manual_seed(seed)
    batch_size = 1
  
    # "CompVis/stable-diffusion-v1-4"
    model_id = blending_config["model_id"]
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler_1 = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler_2 = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    scheduler_blend = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    latents = []
    latent_shape = (batch_size, unet.config.in_channels, blend.height // blend.latent_scale, blend.width // blend.latent_scale)
    scheduler_1.set_timesteps(prompt_1.timesteps)
    scheduler_2.set_timesteps(prompt_2.timesteps)
    scheduler_blend.set_timesteps(blend.to_timestep)
   
    # TODO: move these to the prompt class 
    text_embeddings_1 = create_text_embeddings(prompt_1.prompt, tokenizer, text_encoder, batch_size, device)
    text_embeddings_2 = create_text_embeddings(prompt_2.prompt, tokenizer, text_encoder, batch_size, device)
    
    latent = torch.randn(latent_shape, generator=generator)
    latent = latent * scheduler_1.init_noise_sigma
    latent = latent.to(device)
    latents.append(latent)
    
    latents_1 = reverse_limits(
        from_t = 0,
        to_t = 25,
        latents = latents.copy(),
        scheduler = scheduler_1,
        unet = unet,
        text_embeddings = text_embeddings_1,
        guidance_scale = prompt_1.guidance_scale 
    )
    
    latents_2 = reverse_limits(
        from_t = 0,
        to_t = 25,
        latents = latents.copy(),
        scheduler = scheduler_2,
        unet = unet,
        text_embeddings = text_embeddings_2,
        guidance_scale = prompt_2.guidance_scale
    )
    
    decoded_images_1 = decode_images(latents = latents_1, vae = vae)
    decoded_images_2 = decode_images(latents = latents_2, vae = vae)
    decoded_images_1[-1].save(f"{output_path}/final_image-{prompt_1.prompt}.png")
    decoded_images_2[-1].save(f"{output_path}/final_image-{prompt_2.prompt}.png")
    
    decoded_images_1[10].save(f"{output_path}/intermediate-{prompt_1.prompt}-timestep-10.png")
    
    
    from_t = 9
    to_t = 35
    latents_blend = reverse_limits(
        from_t = from_t,
        to_t = to_t, 
        latents = [latents_1[from_t]],
        scheduler = scheduler_blend,
        unet = unet,
        text_embeddings = text_embeddings_2,
        guidance_scale = blend.guidance_scale
    )
    
    decoded_images_blend = decode_images(latents = latents_blend, vae = vae)
    decoded_images_blend[-1].save(f"{output_path}/final_image-{prompt_1.prompt}-BLEND-{prompt_2.prompt}.png")
    
    make_animation(
        decoded_images=decoded_images_blend,
        prompt=f"{prompt_1.prompt}-BLEND-{prompt_2.prompt}",
        output_path=output_path
    )
    
    make_plots(
        image_1=decoded_images_1[-1],
        image_2=decoded_images_2[-1],
        image_blend=decoded_images_blend[-1],
        prompt_1=prompt_1.prompt,
        prompt_2=prompt_2.prompt,
        output_path=output_path
    )
    
    os.system(f"cp {args.config_path} {output_path}/config.json")
    
    
if __name__ == "__main__":
    main()
