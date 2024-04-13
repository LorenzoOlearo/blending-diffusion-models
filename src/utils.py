import os
import json


def read_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    if not os.path.exists("./out"):
        os.makedirs("./out")
        
    return config["seed"], config["prompt_1_config"], config["prompt_2_config"], config["blending_config"]


def make_output_dir(seed, prompt_1_config, prompt_2_config, blending_config):
    
    output_path = os.path.join("out", str(seed), f"{prompt_1_config['prompt']}-BLEND-{prompt_2_config['prompt']}-scheduler_{blending_config['scheduler']}-model_{blending_config['model_id'].replace('/', '_')}")
    
    if not os.path.exists("./out"):
        os.makedirs("./out")
       
    if not os.path.exists(os.path.join("out", str(seed))):
        os.makedirs(os.path.join("out", str(seed)))
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    return output_path


def save_configuration(config_path, output_path):
    
    # os.system(f"cp {args.config_path} {output_path}/config.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    with open(f"{output_path}/config.json", "w") as f:
        json.dump(config, f, indent=4)
        