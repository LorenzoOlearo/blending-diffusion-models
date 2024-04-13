import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler


class Prompt:
    
    def __init__(self, prompt_config, device):
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
        
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        
        # TODO: Load scheduler according to the prompt_config
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
        
        self.scheduler.set_timesteps(self.timesteps)
        
        self.text_embeddings = self._create_text_embeddings()
        
        
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
   
    
        
class Blending:
    
    def __init__(self, blending_config, device):
        self.model_id = blending_config['model_id']
        self.height = blending_config['height']
        self.width = blending_config['width']
        self.guidance_scale = blending_config['guidance_scale']
        self.latent_scale = blending_config['latent_scale']
        self.from_timestep = blending_config['from_timestep']
        self.to_timestep = blending_config['to_timestep']
        self.latent_scale = blending_config['latent_scale']
        
        # TODO: Load scheduler according to the blending_config
        # self.scheduler = blending_config['scheduler']
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        
        # self.scheduler = UniPCMultistepScheduler.from_pretrained(blending_config['scheduler'], subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder="unet")
        
        self.vae.to(device)
        self.text_encoder.to(device)
        self.unet.to(device)
        
        self.scheduler.set_timesteps(self.to_timestep)
        