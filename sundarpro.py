import torch  # type: ignore
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model with optimizations
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,  # Use mixed precision
    revision="fp16"
).to(device)

# Use a faster scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Enable memory optimizations
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()

# Function to generate an image from text input
def generate_text_image(prompt):
    print(f"Generating image for: {prompt}")

    # Generate image with optimized settings
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=25).images[0]  # Reduced steps for faster generation

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    # Save the image
    image.save("output_image.png")

# Get user input
if __name__ == "__main__":
    prompt = input("Enter a description for the image: ")
    generate_text_image(prompt)
