import torch
import numpy as np
from tqdm.auto import tqdm
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

from pipelines.single_diffusion_pipeline import SingleDiffusionPipeline


class BlendedInterpolatedPromptsPipeline(DiffusionPipeline):
    
    scheduler_map = {
        "UniPCMultistepScheduler": UniPCMultistepScheduler
    }
    
    def __init__(self, vae, tokenizer, text_encoder, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder)
        
        
    def __call__(self, config, generator):
        prompt_1 = config["prompt_1"]
        prompt_2 = config["prompt_2"]
        timesteps = config["timesteps"]
        interpolation_scale = config["blended_interpolated_prompts_scale"]
        
        self.scheduler.set_timesteps(timesteps)
        scheduler_1 = UniPCMultistepScheduler().from_config(self.scheduler.config)
        scheduler_2 = UniPCMultistepScheduler().from_config(self.scheduler.config)
        scheduler_blend = UniPCMultistepScheduler().from_config(self.scheduler.config)
        
        prompt_1_embeddings = self.create_text_embeddings(prompt_1)
        prompt_2_embeddings = self.create_text_embeddings(prompt_2)
        blended_prompts = ((1 - interpolation_scale) * prompt_1_embeddings) + (interpolation_scale * prompt_2_embeddings)
        
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
        
        pipeline_blend = SingleDiffusionPipeline(
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet,
            scheduler=scheduler_blend
        ).to(self.device)
        
        prompt_1_latents, prompt_1_embeddings = pipeline_1(prompt_1, config, generator)
        prompt_2_latents, prompt_2_embeddings = pipeline_2(prompt_2, config, generator)
        blend_latents, _ = pipeline_blend(None, config, generator, prompt_embeddings=blended_prompts, base_latent=prompt_1_latents[0])
        
        return prompt_1_latents, prompt_2_latents, blend_latents
    
    
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
    
    
    