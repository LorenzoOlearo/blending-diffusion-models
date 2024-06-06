import torch
from tqdm.auto import tqdm
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

import plots as plots
import utils as utils
from pipelines.single_diffusion_pipeline import SingleDiffusionPipeline


class SwitchPipeline(DiffusionPipeline):
    
    scheduler_map = {
        "UniPCMultistepScheduler": UniPCMultistepScheduler
    }
    
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler):
        super().__init__()
        
        self.register_modules(unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler)


    @torch.no_grad()
    def __call__(self, config, generator):
        prompt_1 = config["prompt_1"]
        prompt_2 = config["prompt_2"]
        timesteps = config["timesteps"]
        from_timestep = config["from_timestep"]
        to_timestep = config["to_timestep"]
        
        self.scheduler.set_timesteps(timesteps)
        scheduler_1 = UniPCMultistepScheduler().from_config(self.scheduler.config)
        scheduler_2 = UniPCMultistepScheduler().from_config(self.scheduler.config)
        
        pipeline_1 = SingleDiffusionPipeline(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet,
            scheduler=scheduler_1
        ).to(self.device)
        
        pipeline_2 = SingleDiffusionPipeline(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet,
            scheduler=scheduler_2
        ).to(self.device)
        
        latent_shape = (1, self.unet.config.in_channels, config["height"] // config["latent_scale"], config["width"] // config["latent_scale"])
        base_latent = torch.randn(latent_shape, generator=generator, device=self.device)
        base_latent = base_latent * self.scheduler.init_noise_sigma
        base_latent = base_latent.to(self.device)
       
        prompt_1_latents, prompt_1_embeddings = pipeline_1(prompt_1, config, generator, base_latent=base_latent)
        prompt_2_latents, prompt_2_embeddings = pipeline_2(prompt_2, config, generator, base_latent=base_latent)
        
        blend_latents = self.reverse(
            base_latent=prompt_1_latents[from_timestep],
            text_embeddings=prompt_2_embeddings,
            from_timestep=from_timestep,
            to_timestep=to_timestep,
            guidance_scale=config["guidance_scale"]
        )
        
        return prompt_1_latents, prompt_2_latents, blend_latents
        
        
    def reverse(self, base_latent, text_embeddings, from_timestep, to_timestep, guidance_scale):
        latents = []
        latents.append(base_latent)
        
        for t in tqdm(range(from_timestep, to_timestep)):
            latent_model_input = torch.cat([latents[t-from_timestep]] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, self.scheduler.timesteps[t], encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            latent = self.scheduler.step(noise_pred, self.scheduler.timesteps[t], latents[t-from_timestep]).prev_sample
            latents.append(latent)
            
        return latents
            