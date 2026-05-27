import torch
import math

class ForwardDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        
        # Defining the variance schedule (betas)
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x_0, t):
        # Generate random pure Gaussian noise
        noise = torch.randn_like(x_0)
        
        # Extract the scaling factors for the specific timesteps in this batch
        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        # core forward diffusion equation
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return x_t, noise

if __name__ == "__main__":
    import os
    from torchvision.utils import save_image
    from torch.utils.data import DataLoader
    from dataset import CelebADataset 
    
    DATA_DIR = "../archive/img_align_celeba/img_align_celeba"
    CSV_PATH = "../archive/list_attr_celeba.csv"
    
    if os.path.exists(DATA_DIR):
        print("Loading a sample to test noise injection...")
        dataset = CelebADataset(image_dir=DATA_DIR, attr_csv_path=CSV_PATH, image_size=64)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        clean_image, prompt = next(iter(dataloader))
        
        diffusion = ForwardDiffusion(timesteps=1000)
        
        # Let's test specific timesteps to see the degradation
        timesteps_to_test = torch.tensor([0, 250, 500, 750, 999])
        noisy_images = []
        
        for t in timesteps_to_test:
            # We pass the image and a batch of timesteps (just one in this case)
            x_t, _ = diffusion.add_noise(clean_image, torch.tensor([t]))
            
            # Un-normalize from [-1, 1] back to [0, 1] for saving/viewing
            x_t_viewable = (x_t.clamp(-1, 1) + 1) / 2
            noisy_images.append(x_t_viewable)
            
        # Concatenate them side-by-side and save
        all_images = torch.cat(noisy_images, dim=0)
        save_image(all_images, "forward_diffusion_test.png", nrow=5)
        
        print(f"Test complete! Prompt: '{prompt[0]}'")
        print("Check your project folder for 'forward_diffusion_test.png' to see the signal degrade.")
    else:
        print("Dataset not found. Run from the project root.")