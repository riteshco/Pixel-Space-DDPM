import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


from dataset import CelebADataset
from diffusion import ForwardDiffusion
from unet import UNet
from text_encoder import TextEncoder

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on device: {device}")
    
    batch_size = 8
    epochs = 1
    learning_rate = 1e-4
    image_size = 64
    
    data_dir = "../archive/img_align_celeba/img_align_celeba"
    csv_path = "../archive/list_attr_celeba.csv"
    
    print("Initializing components...")
    dataset = CelebADataset(data_dir, csv_path, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    diffusion = ForwardDiffusion(timesteps=1000)
    text_encoder = TextEncoder(device=device)
    
    model = UNet(in_channels=3, out_channels=3, time_emb_dim=128, text_emb_dim=512).to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() 
    
    print(f"Starting training for {epochs} epoch(s)...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (clean_images, prompts) in enumerate(progress_bar):
            # Move images to GPU
            clean_images = clean_images.to(device)
            
            # a. Encode text prompts
            text_embeddings = text_encoder.encode(prompts)
            
            # b. Sample random timesteps for this batch
            # shape: (batch_size,) - values between 0 and 999
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # c. Add noise to the clean images (Forward process)
            noisy_images, actual_noise = diffusion.add_noise(clean_images, t)
            noisy_images = noisy_images.to(device)
            actual_noise = actual_noise.to(device)
            
            # d. Predict the noise!
            optimizer.zero_grad()
            predicted_noise = model(noisy_images, t, text_embeddings)
            
            # e. Calculate Loss (MSE between prediction and reality)
            loss = criterion(predicted_noise, actual_noise)
            
            # f. Backpropagate and update weights
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # FOR TESTING:
            """ if batch_idx >= 50: 
                print("\nStopping early for sanity check. The pipeline works!")
                break """
                
        avg_loss = epoch_loss / min(51, len(dataloader))
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Save the model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/unet_sanity_check.pt")
    print("Model saved to checkpoints/unet_sanity_check.pt")

if __name__ == "__main__":
    train()