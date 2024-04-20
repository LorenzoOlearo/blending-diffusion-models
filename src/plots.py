import matplotlib.pyplot as plt
import matplotlib.animation as animation

import utils


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
    

def save_all_outputs(config, prompt_1_images, prompt_2_images, blend_images, output_path):
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
    