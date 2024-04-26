import torch
from tqdm.auto import tqdm
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

from pipelines.single_diffusion_pipeline import SingleDiffusionPipeline


class BlendedInUnetPipeline(DiffusionPipeline):
    
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
        
        prompt_1_latents, prompt_1_embeddings = pipeline_1(prompt_1, config, generator)
        prompt_2_latents, prompt_2_embeddings = pipeline_2(prompt_2, config, generator)
        
        batch_size = 1 
        latents = []
        latent_shape = (batch_size, self.unet_blend.config.in_channels, config["height"] // config["latent_scale"], config["width"] // config["latent_scale"])
        latent = torch.randn(latent_shape, generator=generator, device=self.device)
        # latent = torch.randn(latent_shape, generator=generator)
        latent = latent * self.scheduler.init_noise_sigma
        latent = latent.to(self.device)
        latents.append(latent)
        
        blend_latents = self.reverse_blend_unet(config, latents, prompt_1_embeddings, prompt_2_embeddings)
        
        return prompt_1_latents, prompt_2_latents, blend_latents
        
        
    def reverse_blend_unet(self, config, latents, encoder_hidden_states, decoder_hidden_states):
        latent = latents[0]
        
        for t in tqdm(self.scheduler.timesteps):
            # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes
            latent_model_input = torch.cat([latent] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            
            with torch.no_grad():
                noise_pred = self.unet_blend(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_hidden_states=decoder_hidden_states
                ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config["guidance_scale"] * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
            latents.append(latent)
        
        return latents
        
    
   
    
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
    
    
    
    
    