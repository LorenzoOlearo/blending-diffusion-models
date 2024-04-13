import os
import json


def read_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config["seed"], config["prompt_1_config"], config["prompt_2_config"], config["blending_config"]


def make_output_dir(seed, prompt_1_config, prompt_2_config, blending_config):
    output_path = os.path.join("out", str(seed), f"[{prompt_1_config['prompt']}-BLEND-{prompt_2_config['prompt']}]-[from_{blending_config['from_timestep']}-to_{blending_config['to_timestep']}]-[{blending_config['scheduler']}]-[{blending_config['model_id'].replace('/', '_')}]")
    
    if os.path.exists(output_path):
        i = 1
        while os.path.exists(f"{output_path}_{i}"):
            i += 1
        output_path = f"{output_path}_{i}"
    
    os.makedirs("./out", exist_ok=True)
    os.makedirs(os.path.join("out", str(seed)), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    return output_path


def save_configuration(config_path, output_path):
    
    # os.system(f"cp {args.config_path} {output_path}/config.json")
    
    with open("config.json", "r") as f:
        config = json.load(f)
        
    with open(f"{output_path}/config.json", "w") as f:
        json.dump(config, f, indent=4)
        