import os
import json
import shutil
import torch
from PIL import Image


def decode_image(latent, vae):
    latent = 1 / 0.18215 * latent
    with torch.no_grad():
        image = vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    
    return image


def decode_images(latents, vae):
    decoded_images = []
    for latent in latents:
        decoded_image = decode_image(latent, vae)
        decoded_images.append(decoded_image)
        
    return decoded_images


def read_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config


def get_additional_parameters(config, blend_method):
    additional_parameters_string =  None
    if blend_method == "SWITCH":
        from_timestep = config["from_timestep"]
        to_timestep = config["to_timestep"]
        additional_parameters_string = f"from_{from_timestep}-to_{to_timestep}"
    elif blend_method == "UNET":
        pass
    elif blend_method == "TEXTUAL":
        interpolation_scale = config["TEXTUAL_scale"]
        interpolation_scale = str(interpolation_scale).replace(".", "-")
        additional_parameters_string = f"scale_{interpolation_scale}"
    elif blend_method == "ALTERNATE":
        pass
    else: 
        raise ValueError(f"Method {blend_method} not recognized.")
    
    return additional_parameters_string


def make_output_dir(seed, config, blend_method, overwrite):
    scheduler_name = config["scheduler"]
    model_id = config["model_id"].replace("/", "-")
    prompt_1 = config["prompt_1"]
    prompt_2 = config["prompt_2"]
    
    additional_parameters = get_additional_parameters(config, blend_method)
    
    output_path = "./out"
    output_path = os.path.join(output_path, f"{prompt_1}-BLEND-{prompt_2}")
    output_path = os.path.join(output_path, blend_method)
    if additional_parameters is not None:
        output_path = os.path.join(output_path, additional_parameters)
    output_path = os.path.join(output_path, str(seed))
    if additional_parameters is not None:
        output_path = os.path.join(output_path, f"[{blend_method}]-[{additional_parameters}]-[{scheduler_name}]-[{model_id}]")
    else:
        output_path = os.path.join(output_path, f"[{blend_method}]-[{scheduler_name}]-[{model_id}]")
   
    if os.path.exists(output_path) and not overwrite:
        overwrite = input(f"Output directory {output_path} already exists. Do you want to overwrite it? (y/N): ")
        if overwrite.lower() == "y":
            print("Overwriting...")
            shutil.rmtree(output_path)
        else:
            name = input("Would you like to append a name to this run? (Leave blank for progressive numbering): ")
            if name == "":
                name = 1
                while os.path.exists(f"{output_path}-{name}"):
                    name += 1
            output_path = f"{output_path}-{name}"
            
    print(f"Output directory: {output_path}")
    os.makedirs("./out", exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    return output_path


def save_configuration(config_path, output_path):
    with open("config.json", "r") as f:
        config = json.load(f)
        
    with open(f"{output_path}/config.json", "w") as f:
        json.dump(config, f, indent=4)
        