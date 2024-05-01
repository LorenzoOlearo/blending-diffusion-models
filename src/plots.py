import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import numpy as np
import os


FINAL_IMAGE_PREFIX = "final"
FINAL_IMAGE_BLEND_PREFIX = "blend"
INTERMEDIATE_IMAGE_PREFIX = "intermediate"


def save_all_outputs(config, prompt_1_images, prompt_2_images, blend_images, blend_method, output_path):
    prompt_1 = config["prompt_1"]
    prompt_2 = config["prompt_2"]
    timesteps = config["timesteps"]
    final_image_1 = prompt_1_images[-1]
    final_image_2 = prompt_2_images[-1]
    final_image_blend = blend_images[-1]
   
    additional_parameters = get_additional_parameters_string(config, blend_method)
    
    final_image_1.save(f"{output_path}/{prompt_1}.png")
    final_image_2.save(f"{output_path}/{prompt_2}.png")
    final_image_blend.save(f"{output_path}/{prompt_1}-BLEND-{prompt_2}.png")
    
    make_animation(
        decoded_images=blend_images,
        prompt=f"{prompt_1}-BLEND-{prompt_2}",
        output_path=output_path
    )
    
    make_plots(
        image_1 = final_image_1,
        image_2 = final_image_2,
        image_blend = final_image_blend,
        prompt_1 = prompt_1,
        prompt_2 = prompt_2,
        timesteps = timesteps,
        blend_method=blend_method,
        additional_parameters = additional_parameters,
        output_path = output_path
    )
    
    
def get_additional_parameters_string(config, blend_method):
    additional_parameters_string =  None
    if blend_method == "blended_diffusion":
        from_timestep = config["from_timestep"]
        to_timestep = config["to_timestep"]
        additional_parameters_string = f"from_{from_timestep}-to_{to_timestep}"
    elif blend_method == "blended_in_unet":
        pass
    elif blend_method == "blended_interpolated_prompts":
        interpolation_scale = config["blended_interpolated_prompts_scale"]
        interpolation_scale = str(interpolation_scale).replace(".", "-")
        additional_parameters_string = f"scale_{interpolation_scale}"
    elif blend_method == "blended_alternate_unet":
        pass
    else: 
        raise ValueError(f"Method {blend_method} not recognized.")
    
    return additional_parameters_string


def make_animation(decoded_images: list, prompt: str, output_path: str):
    plt.close()
    fig, ax = plt.subplots()

    plt.title(prompt[0])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)

    ims = []
    for i in range(len(decoded_images)):
        im = ax.imshow(decoded_images[i], animated=True) 
        if i == 0:
            ax.imshow(decoded_images[i])
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat_delay=2000)
    
    save_path = f"{output_path}/denoising-{prompt}.gif"
    ani.save(save_path)
    
    
def make_plots(image_1, image_2, image_blend, prompt_1, prompt_2, timesteps, blend_method, additional_parameters, output_path):
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image_1)
    ax[0].set_title(prompt_1+f"\ntimesteps: {timesteps}")
    ax[0].axis("off")
    
    ax[1].imshow(image_blend)
    if additional_parameters is not None:
        ax[1].set_title(f"{prompt_1}-BLEND-{prompt_2}-{blend_method}-{additional_parameters}")
    else:
        ax[1].set_title(f"{prompt_1}-BLEND-{prompt_2}-{blend_method}")
    ax[1].axis("off")
    
    ax[2].imshow(image_2)
    ax[2].set_title(prompt_2+f"\ntimesteps: {timesteps}")
    ax[2].axis("off")
   
    plt.savefig(f"{output_path}/comparison-{prompt_1}-BLEND-{prompt_2}.png")
    
    
def make_blending_batch_grid(output_paths, blend_method, config):
    seeds = config["seeds"]
    scheduler_name = config["scheduler"]
    model_id = config["model_id"].replace("/", "-")
    prompt_1 = config["prompt_1"]
    prompt_2 = config["prompt_2"]
    additional_parameters = get_additional_parameters_string(config, blend_method)
   
    rows = [] 
    for folder in output_paths:
        image_1 = plt.imread(os.path.join(folder, f"{prompt_1}.png"))
        image_2 = plt.imread(os.path.join(folder, f"{prompt_2}.png"))
        blend_image = plt.imread(os.path.join(folder, f"{prompt_1}-BLEND-{prompt_2}.png"))
        rows.append([image_1, blend_image, image_2])
        
    plt.close()
    fig, axs = plt.subplots(len(rows), 3, figsize=(20, 32))
    if additional_parameters is not None:
        fig.suptitle(f'{prompt_1}-BLEND-{prompt_2}-[{blend_method}-{additional_parameters}]', fontsize=20)
    else:
        fig.suptitle(f'{prompt_1}-BLEND-{prompt_2}-[{blend_method}]', fontsize=20)
    
    for i, row in enumerate(rows):
        axs[i, 0].set_title(f'prompt 1: {prompt_1}')
        axs[i, 1].set_title(f'Seed: {seeds[i]}')
        axs[i, 2].set_title(f'prompt 2: {prompt_2}')
        axs[i, 0].imshow(row[0])
        axs[i, 0].axis("off")
        axs[i, 1].imshow(row[1])
        axs[i, 1].axis("off")
        axs[i, 2].imshow(row[2])
        axs[i, 2].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    save_path = "./out"
    save_path = os.path.join(save_path, f"{prompt_1}-BLEND-{prompt_2}")
    save_path = os.path.join(save_path, blend_method)
    if additional_parameters is not None: 
        save_path = os.path.join(save_path, additional_parameters)
    save_path = os.path.join(save_path, f"results-[{blend_method}]-[{additional_parameters}]-[{model_id}]-[{scheduler_name}].png")
    
    plt.savefig(save_path)
    
    
def make_blend_comparison_grid(config):
    seeds = config["seeds"]
    scheduler_name = config["scheduler"]
    model_id = config["model_id"].replace("/", "-")
    prompt_1 = config["prompt_1"]
    prompt_2 = config["prompt_2"]
    timesteps = config["timesteps"]
    
    # Set the order of the blend methods 
    blend_methods = ["blended_diffusion", "blended_in_unet", "blended_interpolated_prompts", "blended_alternate_unet"]
   
    # Remove the blend methods that are not in the config 
    if "blended_diffusion" not in blend_methods:
        blend_methods.remove("blended_diffusion")
    if "blended_in_unet" not in blend_methods:
        blend_methods.remove("blended_in_unet")
    if "blended_interpolated_prompts" not in blend_methods:
        blend_methods.remove("blended_interpolated_prompts")
    if "blended_alternate_unet" not in blend_methods:
        blend_methods.remove("blended_alternate_unet")
    
    grid = [] 
    for blend_method in blend_methods:
        additional_parameters = get_additional_parameters_string(config, blend_method)
        input_folder = "./out"
        input_folder = os.path.join(input_folder, f"{prompt_1}-BLEND-{prompt_2}")
        input_folder = os.path.join(input_folder, blend_method)
        if additional_parameters is not None:
            input_folder = os.path.join(input_folder, additional_parameters)
        blend_column = []
        images_1 = []
        images_2 = []
        tmp_input_folder = input_folder
        for seed in seeds:
            input_folder = tmp_input_folder
            input_folder = os.path.join(input_folder, str(seed))
            if additional_parameters is not None:
                input_folder = os.path.join(input_folder, f"[{blend_method}]-[{additional_parameters}]-[{scheduler_name}]-[{model_id}]")
            else:
                input_folder = os.path.join(input_folder, f"[{blend_method}]-[{scheduler_name}]-[{model_id}]")
            blend_column.append(plt.imread(os.path.join(input_folder, f"{prompt_1}-BLEND-{prompt_2}.png")))
            images_1.append(plt.imread(os.path.join(input_folder, f"{prompt_1}.png")))
            images_2.append(plt.imread(os.path.join(input_folder, f"{prompt_2}.png")))
        grid.append(blend_column)
        
    plt.close()
    fig, axs = plt.subplots(len(seeds), len(blend_methods)+2, figsize=(30, 30))
    fig.suptitle(f'{prompt_1}-BLEND-{prompt_2}', fontsize=20)
    
    for i, seed in enumerate(seeds):
        axs[i, 0].set_title(f'{prompt_1}-seed-{seed}')
        axs[i, 1].set_title(f'{prompt_2}-seed-{seed}')
        axs[i, 0].imshow(images_1[i])
        axs[i, 0].axis("off")
        axs[i, 1].imshow(images_2[i])
        axs[i, 1].axis("off")
        for j, blend_method in enumerate(blend_methods):
            axs[i, j+2].set_title(f'{blend_method}')
            axs[i, j+2].imshow(grid[j][i])
            axs[i, j+2].axis("off")
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    save_path = "./out"
    save_path = os.path.join(save_path, f"{prompt_1}-BLEND-{prompt_2}")
    save_path = os.path.join(save_path, "comparison")
    
    plt.savefig(save_path)
    