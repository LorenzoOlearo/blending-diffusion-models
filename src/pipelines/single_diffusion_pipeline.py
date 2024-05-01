import torch
from tqdm.auto import tqdm
from diffusers import DiffusionPipeline, UniPCMultistepScheduler


class SingleDiffusionPipeline(DiffusionPipeline):
    
    scheduler_map = {
        "UniPCMultistepScheduler": UniPCMultistepScheduler
    }
    
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder)
        
        
    def __call__(self, prompt, config, generator, prompt_embeddings=None, base_latent=None):
        timesteps = config["timesteps"]
        height = config["height"]
        width = config["width"]
        latent_scale = config["latent_scale"]
        same_base_latent = config["same_base_latent"]
        
        self.scheduler.set_timesteps(timesteps)
        batch_size = 1
        
        latents = []
        if same_base_latent == True and base_latent is not None:
            latents.append(base_latent)
        elif same_base_latent == False:
            latent_shape = (batch_size, self.unet.config.in_channels, height // latent_scale, width // latent_scale)
            latent = torch.randn(latent_shape, generator=generator, device=self.device)
            latent = latent * self.scheduler.init_noise_sigma
            latent = latent.to(self.device)
            latents.append(latent)
        
        if prompt is not None:
            text_embeddings = self.create_text_embeddings(prompt, batch_size)
        elif prompt_embeddings is not None:
            text_embeddings = prompt_embeddings
        else:
            raise ValueError("Prompt or prompt_embeddings must be provided") 
        
        latents = self.reverse(config, latents[-1], text_embeddings)
        
        return latents, text_embeddings
    
    
    def reverse(self, config, base_latent, prompt_embeddings):
        latents = []
        latents.append(base_latent)
       
        for t in tqdm(self.scheduler.timesteps):
            latent = latents[-1]
            # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes
            latent_model_input = torch.cat([latent] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeddings,
                ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config["guidance_scale"] * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
            latents.append(latent)
        
        return latents
    
   
    def create_text_embeddings(self, prompt, batch_size=1):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        return text_embeddings
    