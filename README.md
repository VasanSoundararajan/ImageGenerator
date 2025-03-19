# Stable Diffusion Image Generator

This project uses the Stable Diffusion model to generate high-quality images from text descriptions using optimized settings for faster inference.

## Features
- Utilizes the **Stable Diffusion v1.4** model for text-to-image generation.
- Implements **mixed precision (fp16)** for improved performance on compatible GPUs.
- Optimized with:
  - **`EulerAncestralDiscreteScheduler`** for faster image generation.
  - **Memory-efficient attention** for reduced GPU usage.
  - **Attention slicing** for better memory management.
- Generates and saves the generated image as `output_image.png`.

---

## Requirements
### Prerequisites
Ensure you have the following libraries installed:
- `torch`
- `diffusers`
- `transformers`
- `matplotlib`

You can install them via:
```sh
pip install torch diffusers transformers matplotlib
```

### Hardware Requirement
- **GPU (NVIDIA recommended)** for optimal performance.
- Supports CPU, but performance may be slower.

---

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/stable-diffusion-image-generator.git
   cd stable-diffusion-image-generator
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage
1. Run the Python script:
   ```sh
   python main.py
   ```
2. Enter a text prompt when prompted:
   ```
   Enter a description for the image: A futuristic city skyline at sunset
   ```
3. The generated image will be displayed and saved as `output_image.png`.

---

## Sample Output
```
Enter a description for the image: A fantasy castle on a mountain peak
Generating image for: A fantasy castle on a mountain peak
[Image displayed here]
```

---

## Future Improvements
- Add support for batch image generation.
- Implement advanced prompt engineering for enhanced results.
- Introduce additional scheduler options for diverse artistic styles.

---

## License
This project is licensed under the MIT License.

