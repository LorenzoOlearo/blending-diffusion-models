import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import numpy as np
import os

def save_image(image, filename, output_path):
    image.save(f"{output_path}/{filename}.png")
    

def make_animation(decoded_images: list, prompt: str, output_path: str):
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
    
    
def make_plots(image_1, image_2, image_blend, prompt_1, prompt_2, output_path, p1_t, p2_t, blending_from_t, blending_to_t):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # Set title and timesteps on new line
    ax[0].imshow(image_1)
    ax[0].set_title(prompt_1+f"\ntimesteps: {p1_t}")
    ax[0].axis("off")
    
    
    ax[1].imshow(image_blend)
    ax[1].set_title(f"{prompt_1}-BLEND-{prompt_2}"+f"\ntimesteps from {blending_from_t} to {blending_to_t}")
    ax[1].axis("off")
    
    ax[2].imshow(image_2)
    ax[2].set_title(prompt_2+f"\ntimesteps: {p2_t}")
    ax[2].axis("off")
    
    plt.savefig(f"{output_path}/blending-{prompt_1}-BLEND-{prompt_2}.png")
    

# TODO: refactor considering the new config structure and the different methods of blending
def save_all_outputs(config, prompt_1_images, prompt_2_images, blend_images, blend_method, output_path):
    prompt_1 = config["prompt_1"]
    prompt_2 = config["prompt_2"]
    timesteps = config["timesteps"]
    from_timestep = config["from_timestep"]
    to_timestep = config["to_timestep"]
    
    save_image(prompt_1_images[-1], f"final_image-{prompt_1}", output_path)
    save_image(prompt_2_images[-1], f"final_image-{prompt_2}", output_path)
    save_image(prompt_1_images[from_timestep], f"intermediate-{prompt_1}-timestep-{from_timestep}", output_path)
    
    blend_images[-1].save(f"{output_path}/final_image-{prompt_1}-BLEND-{prompt_2}.png")
    
    make_animation(
        decoded_images=blend_images,
        prompt=f"{prompt_1}-BLEND-{prompt_2}",
        output_path=output_path
    )
    
    make_plots(
        image_1 = prompt_1_images[-1],
        image_2 = prompt_2_images[-1],
        image_blend = blend_images[-1],
        prompt_1 = prompt_1,
        prompt_2 = prompt_2,
        output_path = output_path,
        p1_t = timesteps,
        p2_t = timesteps,
        blending_from_t = from_timestep,
        blending_to_t = to_timestep
    )
    
    
def make_blending_batch_grid(output_paths, blend_method, config):
    seeds = config["seeds"]
    scheduler_name = config["scheduler"]
    model_id = config["model_id"].replace("/", "-")
    prompt_1 = config["prompt_1"]
    prompt_2 = config["prompt_2"]
    timesteps = config["timesteps"]
    from_timestep = config["from_timestep"]
    to_timestep = config["to_timestep"]
   
    rows = [] 
    for folder in output_paths:
        image_1 = plt.imread(os.path.join(folder, f"final_image-{prompt_1}.png"))
        image_2 = plt.imread(os.path.join(folder, f"final_image-{prompt_2}.png"))
        blend_image = plt.imread(os.path.join(folder, f"final_image-{prompt_1}-BLEND-{prompt_2}.png"))
        rows.append([image_1, blend_image, image_2])
        
    plt.close()
    fig, axs = plt.subplots(len(rows), 3, figsize=(20, 32))
    
    if blend_method == "blended_diffusion":
        fig.suptitle(f'{blend_method}-{prompt_1}-BLEND-{prompt_2}-from_{from_timestep}-to_{to_timestep}-[{scheduler_name}]-[{model_id}]', fontsize=20)
    elif blend_method == "blended_in_unet":
        fig.suptitle(f'{blend_method}-{prompt_1}-BLEND-{prompt_2}-[{scheduler_name}]-[{model_id}]-p1_{timesteps}-p2_{timesteps}', fontsize=20)
    elif blend_method == "blended_interpolated_prompts":
        interpolation_scale = config["blended_interpolated_prompts_scale"]
        fig.suptitle(f'{blend_method}-{interpolation_scale}-{prompt_1}-BLEND-{prompt_2}-[{scheduler_name}]-[{model_id}]-p1_{timesteps}-p2_{timesteps}', fontsize=20)
    
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
    save_path = os.path.join(save_path, f"[from_{from_timestep}]-[to_{to_timestep}]")
    save_path = os.path.join(save_path, f"results-[from_{from_timestep}]-[to_{to_timestep}]-[{scheduler_name}]-[{model_id}]")
    
    output_path = "./out"
    output_path = os.path.join(output_path, f"{prompt_1}-BLEND-{prompt_2}")
    output_path = os.path.join(output_path, blend_method)
    if blend_method == "blended_diffusion":
        output_path = os.path.join(output_path, f"[from_{from_timestep}]-[to_{to_timestep}]")
        output_path = os.path.join(output_path, f"{blend_method}-batch_grid.png")
    elif blend_method == "blended_in_unet":
        output_path = os.path.join(output_path, f"{blend_method}-batch_grid.png")
    elif blend_method == "blended_interpolated_prompts":
        interpolation_scale = config["blended_interpolated_prompts_scale"]
        output_path = os.path.join(output_path, str(interpolation_scale))
        output_path = os.path.join(output_path, f"{blend_method}-batch_grid.png")
    elif blend_method == "blended_switch_unet":
        output_path = os.path.join(output_path, f"[{blend_method}]-[{scheduler_name}]-[{model_id}]-[p1_{timesteps}]-[p2_{timesteps}]")
    else:
        raise ValueError(f"Method {blend_method} not recognized.")
       
    plt.savefig(output_path)
    