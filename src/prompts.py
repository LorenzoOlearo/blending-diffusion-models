import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler


class Prompt(nn.Module):
    vae = None 
    tokenizer = None
    text_encoder = None
    unet = None
    
    scheduler_map = {
        "UniPCMultistepScheduler": UniPCMultistepScheduler
    }
    
    def __init__(self, prompt_config, device, shared):
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
        
        if (not shared):
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
        
        self.scheduler = Prompt.scheduler_map[prompt_config['scheduler']].from_pretrained(self.model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(self.timesteps)
        
        
    def get_text_embeddings(self):
        return self._create_text_embeddings()
        
        
    def _create_text_embeddings(self):
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
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        return text_embeddings
   
    
        
class Blending(Prompt):
    
    def __init__(self, blending_config, device):
        super(Blending, self).__init__(blending_config, device, shared=blending_config["shared"])
        self.from_timestep = blending_config['from_timestep']
        self.to_timestep = blending_config['to_timestep']
        self.scheduler.set_timesteps(self.to_timestep)
        