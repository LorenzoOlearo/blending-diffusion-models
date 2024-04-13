import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    
    
def make_plots(image_1, image_2, image_blend, prompt_1, prompt_2, output_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image_1)
    ax[0].set_title(prompt_1)
    ax[0].axis("off")
    
    ax[1].imshow(image_blend)
    ax[1].set_title(f"{prompt_1}-BLEND-{prompt_2}")
    ax[1].axis("off")
    
    ax[2].imshow(image_2)
    ax[2].set_title(prompt_2)
    ax[2].axis("off")
    
    plt.savefig(f"{output_path}/blending-{prompt_1}-BLEND-{prompt_2}.png")