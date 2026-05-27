import torch
from torchvision.utils import save_image
from tqdm import tqdm
import os


from unet import UNet
from text_encoder import TextEncoder
from diffusion import ForwardDiffusion

@torch.no_grad() 
def generate_image(prompt, model_path="checkpoints/unet_sanity_check.pt", image_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating on device: {device}")
    
    diffusion = ForwardDiffusion(timesteps=1000)
    text_encoder = TextEncoder(device=device)
    
    print(f"Loading model weights from {model_path}...")
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=128, text_emb_dim=512).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"WARNING: Checkpoint {model_path} not found. Generating with untrained random weights!")
        
    model.eval() 
    
    print(f"Prompt: '{prompt}'")
    text_embeddings = text_encoder.encode([prompt])
    
    x = torch.randn(1, 3, image_size, image_size).to(device)
    
    print("Starting denoising loop...")
    for i in tqdm(reversed(range(diffusion.timesteps)), total=diffusion.timesteps):
        t = torch.tensor([i]).to(device)
        
        # a. Predict the noise
        predicted_noise = model(x, t, text_embeddings)
        
        # b. Grab the pre-calculated math constants for this timestep
        alpha = diffusion.alphas[i].to(device)
        alpha_cumprod = diffusion.alphas_cumprod[i].to(device)
        beta = diffusion.betas[i].to(device)
        
        # c. Calculate the denoised image (x_{t-1}) using the DDPM formula
        if i > 0:
            noise = torch.randn_like(x) 
        else:
            noise = torch.zeros_like(x) 
            
        # The core DDPM Reverse Equation
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise

    # 6. Save the final image
    # Un-normalize from [-1, 1] back to [0, 1] for saving
    x = (x.clamp(-1, 1) + 1) / 2
    
    output_filename = "generated_sample.png"
    save_image(x, output_filename)
    print(f"\nSuccess! Image saved as '{output_filename}'")

if __name__ == "__main__":
    test_prompt = "A photo of a person face with Smiling, Eyeglasses, Male, Young"
    generate_image(test_prompt)