import math
import torch
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        
        # Calculate the frequencies (similar to high-frequency hardware clocks)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU() 
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        # If time embedding is provided, we project it to match out_channels
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

    def forward(self, x, t=None):
        # First conv layer
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        
        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            time_emb = time_emb[(...,) + (None,) * 2] 
            h = h + time_emb
            
        # Second conv layer
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        return h

if __name__ == "__main__":
    print("Testing U-Net Foundations...")
    
    time_dim = 32
    time_embedder = SinusoidalPositionEmbeddings(time_dim)
    dummy_times = torch.tensor([10, 50, 100, 999]) 
    t_emb = time_embedder(dummy_times)
    
    print(f"Time Embedding Input Shape (Batch): {dummy_times.shape}")
    print(f"Time Embedding Output Shape: {t_emb.shape} - Expected: (4, 32)")
    
    in_ch = 3   # e.g., RGB image
    out_ch = 64 
    block = Block(in_channels=in_ch, out_channels=out_ch, time_emb_dim=time_dim)
    
    # Simulate a noisy image batch: [Batch, Channels, Height, Width]
    dummy_image = torch.randn(4, 3, 64, 64) 
    
    # Pass image and time through the block
    out_tensor = block(dummy_image, t_emb)
    
    print(f"Block Input Image Shape: {dummy_image.shape}")
    print(f"Block Output Shape: {out_tensor.shape} - Expected: (4, 64, 64, 64)")
    print("If shapes match expectations, our hardware blocks are wired correctly!")