import torch
from torchvision.utils import save_image, make_grid
from unet import UNet
from text_encoder import TextEncoder
from diffusion import ForwardDiffusion
import os

@torch.no_grad()
def generate_readme_grid(model_path="checkpoints/unet_sanity_check.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating showcase grid on {device}...")
    
    diffusion = ForwardDiffusion(timesteps=1000)
    text_encoder = TextEncoder(device=device)
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=128, text_emb_dim=512).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    # 4 distinct prompts to show model variance
    prompts = [
        "A photo of a person face with Smiling, Eyeglasses, Male",
        "A photo of a person face with Bangs, No Beard, Young, Female",
        "A photo of a person face with Black Hair, Heavy Makeup",
        "A photo of a person face with Goatee, Bushy Eyebrows, Male"
    ]
    
    text_embeddings = text_encoder.encode(prompts)
    x = torch.randn(4, 3, 64, 64).to(device) # Batch size of 4
    
    for i in reversed(range(diffusion.timesteps)):
        t = torch.tensor([i] * 4).to(device) # Timesteps for the whole batch
        predicted_noise = model(x, t, text_embeddings)
        
        alpha = diffusion.alphas[i].to(device)
        alpha_cumprod = diffusion.alphas_cumprod[i].to(device)
        beta = diffusion.betas[i].to(device)
        
        noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise

    # Process and save the grid
    x = (x.clamp(-1, 1) + 1) / 2
    grid = make_grid(x, nrow=2, padding=2, normalize=False)
    
    os.makedirs("assets", exist_ok=True)
    save_image(grid, "assets/showcase_grid.png")
    print("Saved to assets/showcase_grid.png")

if __name__ == "__main__":
    generate_readme_grid()