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
        
        
    def __call__(self, prompt, config, generator):
        
        timesteps = config["timesteps"]
        self.scheduler.set_timesteps(timesteps)
        batch_size = 1
        
        latents = []
        latent_shape = (batch_size, self.unet.config.in_channels, config["height"] // config["latent_scale"], config["width"] // config["latent_scale"])
        latent = torch.randn(latent_shape, generator=generator, device=self.device)
        # latent = torch.randn(latent_shape, generator=generator)
        latent = latent * self.scheduler.init_noise_sigma
        latent = latent.to(self.device)
        latents.append(latent)
        
        text_embeddings = self.create_text_embeddings(prompt, batch_size)
        latent = latents[0]
        
        for t in tqdm(self.scheduler.timesteps):
            # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes
            latent_model_input = torch.cat([latent] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            
            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config["guidance_scale"] * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
            latents.append(latent)
        
        return latents, text_embeddings
    
   
    
    def create_text_embeddings(self, prompt, batch_size=1):
        # batch_size = self.batch_size
        
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
    
    
    
    
    