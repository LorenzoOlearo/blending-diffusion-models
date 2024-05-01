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
    
    final_image_1.save(f"{output_path}/{FINAL_IMAGE_PREFIX}-{prompt_1}.png")
    final_image_2.save(f"{output_path}/{FINAL_IMAGE_PREFIX}-{prompt_2}.png")
    if additional_parameters is not None:
        final_image_blend.save(f"{output_path}/{FINAL_IMAGE_BLEND_PREFIX}-{additional_parameters}-{prompt_1}-BLEND-{prompt_2}.png")
    else:
        final_image_blend.save(f"{output_path}/{FINAL_IMAGE_BLEND_PREFIX}-{prompt_1}-BLEND-{prompt_2}.png")
    
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
   
    if additional_parameters is not None:
        plt.savefig(f"{output_path}/blending-{blend_method}-{additional_parameters}-{prompt_1}-BLEND-{prompt_2}.png") 
    else:
        plt.savefig(f"{output_path}/blending-{blend_method}-{prompt_1}-BLEND-{prompt_2}.png")
    
    
def make_blending_batch_grid(output_paths, blend_method, config):
    seeds = config["seeds"]
    scheduler_name = config["scheduler"]
    model_id = config["model_id"].replace("/", "-")
    prompt_1 = config["prompt_1"]
    prompt_2 = config["prompt_2"]
    additional_parameters = get_additional_parameters_string(config, blend_method)
   
    rows = [] 
    for folder in output_paths:
        image_1 = plt.imread(os.path.join(folder, f"{FINAL_IMAGE_PREFIX}-{prompt_1}.png"))
        image_2 = plt.imread(os.path.join(folder, f"{FINAL_IMAGE_PREFIX}-{prompt_2}.png"))
        if additional_parameters is not None:
            blend_image = plt.imread(os.path.join(folder, f"{FINAL_IMAGE_BLEND_PREFIX}-{additional_parameters}-{prompt_1}-BLEND-{prompt_2}.png"))
        else:
            blend_image = plt.imread(os.path.join(folder, f"{FINAL_IMAGE_BLEND_PREFIX}-{prompt_1}-BLEND-{prompt_2}.png"))
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
       