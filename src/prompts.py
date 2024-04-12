class Prompt:
    
    def __init__(self, prompt_config):
        self.prompt = prompt_config['prompt']
        self.timesteps = prompt_config['timesteps']
        self.scheduler = prompt_config['scheduler']
        self.model_id = prompt_config['model_id']
        self.height = prompt_config['height']
        self.width = prompt_config['width']
        self.guidance_scale = prompt_config['guidance_scale']
        self.latent_scale = prompt_config['latent_scale']
        
        
class Blending:
    
    def __init__(self, blending_config):
        self.scheduler = blending_config['scheduler']
        self.model_id = blending_config['model_id']
        self.height = blending_config['height']
        self.width = blending_config['width']
        self.guidance_scale = blending_config['guidance_scale']
        self.latent_scale = blending_config['latent_scale']
        self.from_timestep = blending_config['from_timestep']
        self.to_timestep = blending_config['to_timestep']
        self.latent_scale = blending_config['latent_scale']
        