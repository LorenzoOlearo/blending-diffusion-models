import torch
from tqdm.auto import tqdm
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

from pipelines.single_diffusion_pipeline import SingleDiffusionPipeline
from utils import generate_latent


class UnetPipeline(DiffusionPipeline):
    
    scheduler_map = {
        "UniPCMultistepScheduler": UniPCMultistepScheduler
    }
    
    def __init__(self, vae, tokenizer, text_encoder, unet_base, unet_blend, scheduler):
        super().__init__()
        self.register_modules(unet_base=unet_base, unet_blend=unet_blend, scheduler=scheduler, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder)
        
        
    def __call__(self, config, generator):
        prompt_1 = config["prompt_1"]
        prompt_2 = config["prompt_2"]
        timesteps = config["timesteps"]
        height = config["height"]
        width = config["width"]
        latent_scale = config["latent_scale"]
        blend_ratio = config["blend_ratio"]
        
        self.scheduler.set_timesteps(timesteps)
        scheduler_1 = UniPCMultistepScheduler().from_config(self.scheduler.config)
        scheduler_2 = UniPCMultistepScheduler().from_config(self.scheduler.config)
        
        pipeline_1 = SingleDiffusionPipeline(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet_base,
            scheduler=scheduler_1
        ).to(self.device)
        
        pipeline_2 = SingleDiffusionPipeline(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet_base,
            scheduler=scheduler_2
        ).to(self.device)
       
        base_latent = generate_latent(config, generator, self.unet_base, self.scheduler, self.device) 
        
        if config["same_base_latent"] == True:
            prompt_1_latents, prompt_1_embeddings = pipeline_1(prompt_1, config, generator, base_latent=base_latent)
            prompt_2_latents, prompt_2_embeddings = pipeline_2(prompt_2, config, generator, base_latent=base_latent)
        else:
            prompt_1_latents, prompt_1_embeddings = pipeline_1(prompt_1, config, generator, base_latent=base_latent)
            prompt_2_latents, prompt_2_embeddings = pipeline_2(prompt_2, config, generator)
        
        blend_latents = self.reverse(
            config=config,
            encoder_hidden_states=prompt_1_embeddings,
            decoder_hidden_states=prompt_2_embeddings,
            blend_ratio=blend_ratio,
            generator=generator,
            base_latent=base_latent
        )
        
        return prompt_1_latents, prompt_2_latents, blend_latents
        
        
    def reverse(self, config, encoder_hidden_states, decoder_hidden_states, blend_ratio, generator=None, base_latent=None):
        latents = []
        
        if config["same_base_latent"] == True and base_latent is not None:
            latents.append(base_latent)
        elif config["same_base_latent"] == False and generator is not None:
            base_latent = generate_latent(config, generator, self.unet_base, self.scheduler, self.device)
            latents.append(base_latent)
        else:
            raise ValueError("base_latent or generator must be provided")
       
        for t in tqdm(self.scheduler.timesteps):
            latent = latents[-1]
            # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes
            latent_model_input = torch.cat([latent] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            
            with torch.no_grad():
                noise_pred = self.unet_blend(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_hidden_states=decoder_hidden_states,
                    blend_ratio=blend_ratio
                ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config["guidance_scale"] * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
            latents.append(latent)
        
        return latents
    
