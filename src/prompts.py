import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from tqdm.auto import tqdm


class Prompt(nn.Module):
    vae = None 
    tokenizer = None
    text_encoder = None
    unet = None
    generator = None
    
    scheduler_map = {
        "UniPCMultistepScheduler": UniPCMultistepScheduler
    }
    
    def __init__(self, prompt_config, device, seed, shared_pipeline, shared_generator):
        super(Prompt, self).__init__()
        self.device = device
        self.prompt = prompt_config['prompt']
        self.timesteps = prompt_config['timesteps']
        self.model_id = prompt_config['model_id']
        self.height = prompt_config['height']
        self.width = prompt_config['width']
        self.guidance_scale = prompt_config['guidance_scale']
        self.latent_scale = prompt_config['latent_scale']
        
        # TODO: Find a sensible way of managing batch size
        self.batch_size = 1
        
        if (not shared_pipeline):
            self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
            self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        else:
            if Prompt.vae is None:
                Prompt.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
                Prompt.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
                Prompt.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
                Prompt.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
            self.vae = Prompt.vae
            self.tokenizer = Prompt.tokenizer
            self.text_encoder = Prompt.text_encoder
            self.unet = Prompt.unet
            self.generator = Prompt.generator
            
        if (not shared_generator):
            self.generator = torch.manual_seed(seed)
        else:
            if Prompt.generator is None:
                Prompt.generator = torch.manual_seed(seed)
            self.generator = Prompt.generator
        
        self.scheduler = Prompt.scheduler_map[prompt_config['scheduler']].from_pretrained(self.model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(self.timesteps)
        
        self.text_embeddings = None
        
        self.latents = []
        latent_shape = (self.batch_size, self.unet.config.in_channels, self.height // self.latent_scale, self.width // self.latent_scale)
        latent = torch.randn(latent_shape, generator=self.generator)
        latent = latent * self.scheduler.init_noise_sigma
        latent = latent.to(device)
        self.latents.append(latent)
        
       
        
    def create_text_embeddings(self):
        batch_size = self.batch_size
        
        text_input = self.tokenizer(
            self.prompt,
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
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
    
    def reverse(self):
        latent = self.latents[0]
        for t in tqdm(self.scheduler.timesteps):
            # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes
            latent_model_input = torch.cat([latent] * 2)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            
            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=self.text_embeddings).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample
            self.latents.append(latent)
            
            
    
        
class Blending(Prompt):
    
    def __init__(self, blending_config, prompts, seed, device):
        super(Blending, self).__init__(blending_config, device, seed=seed, shared_pipeline=blending_config["shared_pipeline"], shared_generator=blending_config["shared_generator"])
        self.from_timestep = blending_config['from_timestep']
        self.to_timestep = blending_config['to_timestep']
        self.scheduler.set_timesteps(self.to_timestep)
        self.prompts = prompts
        self.latents = []
        self.base_latent = None
        
        
    def reverse_limits(self, base_latent):
        self.base_latent = base_latent
        self.latents.append(self.base_latent)
        
        for t in tqdm(range(self.from_timestep, self.to_timestep)):
            latent_model_input = torch.cat([self.latents[t-self.from_timestep]] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, self.scheduler.timesteps[t], encoder_hidden_states=self.prompts[1].text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latent = self.scheduler.step(noise_pred, self.scheduler.timesteps[t], self.latents[t-self.from_timestep]).prev_sample
            self.latents.append(latent)
        