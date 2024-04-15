import os
import json
import shutil


def read_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    return config["seed"], config["prompt_1_config"], config["prompt_2_config"], config["blending_config"]

# Output path:
# out/{prompt_1}-BLEND-{prompt_2}/seed/[{prompt_1}-BLEND-{prompt_2}]-[from_{from_timestep}]-[to_{to_timestep}]-[{scheduler}]-[{model_id}]-[p1_{prompt_1_timesteps}]-[p2_{prompt_2_timesteps}
def make_output_dir(seed, prompt_1_config, prompt_2_config, blending_config):
    output_path = "./out"
    output_path = os.path.join(output_path, f"{prompt_1_config['prompt']}-BLEND-{prompt_2_config['prompt']}")
    output_path = os.path.join(output_path, str(seed))
    output_path = os.path.join(output_path, f"[{prompt_1_config['prompt']}-BLEND-{prompt_2_config['prompt']}]-[from_{blending_config['from_timestep']}]-[to_{blending_config['to_timestep']}]-[{blending_config['scheduler']}]-[{blending_config['model_id'].replace('/', '-')}]-[p1_{prompt_1_config['timesteps']}]-[p2_{prompt_2_config['timesteps']}]")  
    
    if os.path.exists(output_path):
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
        