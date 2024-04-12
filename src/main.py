import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline
from diffusers import UniPCMultistepScheduler


def create_text_embeddings(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: str):
    batch_size = len(prompt)
    
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
    images = (image * 255).round().astype("uint8")
    image = Image.fromarray(image)
    
    return image


def decode_images(latents, vae):
    decoded_images = []
    for latent in latents:
        decoded_image = _decode_image(latent, vae)
        decoded_images.append(decoded_image)
        
    return decoded_images


def make_animation(decoded_images: list, prompt: str, save_dir: str):
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
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = f"{save_dir}/denoising-{prompt}.gif"
    ani.save(save_path)
    
    
def make_plots(image_1, image_2, image_blend, prompt_1, prompt_2, save_dir="./out"):
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
    
    plt.savefig(f"{save_dir}/blending-{prompt_1}-BLEND-{prompt_2}.png")
    
    

def main():
    device = "cuda:0"

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
    scheduler_1 = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    scheduler_2 = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    scheduler_blend = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    prompt_1 = ["lion"]
    prompt_2 = ["cat"]
    num_inference_steps = 25
    num_inference_steps_blend = 35
    height = 512
    width = 512
    guidance_scale = 7.5
    generator = torch.manual_seed(21)
    batch_size = 1
    latent_shape = (batch_size, unet.config.in_channels, height // 8, width // 8)
    latents = []
    scheduler_1.set_timesteps(num_inference_steps)
    scheduler_2.set_timesteps(num_inference_steps)
    scheduler_blend.set_timesteps(num_inference_steps_blend)
    
    text_embeddings_1 = create_text_embeddings(prompt_1, tokenizer, text_encoder, device)
    text_embeddings_2 = create_text_embeddings(prompt_2, tokenizer, text_encoder, device)
    
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
        guidance_scale = guidance_scale
    )
    
    latents_2 = reverse_limits(
        from_t = 0,
        to_t = 25,
        latents = latents.copy(),
        scheduler = scheduler_2,
        unet = unet,
        text_embeddings = text_embeddings_2,
        guidance_scale = guidance_scale
    )
    
    decoded_images_1 = decode_images(latents = latents_1, vae = vae)
    decoded_images_2 = decode_images(latents = latents_2, vae = vae)
    decoded_images_1[-1].save(f"out/final_image{prompt_1}.png")
    decoded_images_2[-1].save(f"out/final_image{prompt_2}.png")
    
    decoded_images_1[10].save(f"out/final_image-{prompt_1}-AT-10.png")
    
    
    from_t = 9
    to_t = 35
    latents_blend = reverse_limits(
        from_t = from_t,
        to_t = to_t, 
        latents = [latents_1[from_t]],
        scheduler = scheduler_blend,
        unet = unet,
        text_embeddings = text_embeddings_2,
        guidance_scale = guidance_scale
    )
    
    decoded_images_blend = decode_images(latents = latents_blend, vae = vae)
    decoded_images_blend[-1].save(f"out/final_image{prompt_1}-BLEND-{prompt_2}.png")
    
    make_animation(
        decoded_images=decoded_images_blend,
        prompt=f"{prompt_1}-BLEND-{prompt_2}",
        save_dir="./out"
    )
    
    make_plots(
        image_1=decoded_images_1[-1],
        image_2=decoded_images_2[-1],
        image_blend=decoded_images_blend[-1],
        prompt_1=prompt_1,
        prompt_2=prompt_2
    )
    

if __name__ == "__main__":
    main()
