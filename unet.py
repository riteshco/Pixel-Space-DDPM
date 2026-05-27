import math
import torch
from torch import nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
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

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None

    def forward(self, x, t=None):
        h = self.act1(self.norm1(self.conv1(x)))
        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            time_emb = time_emb[(...,) + (None,) * 2]
            h = h + time_emb
        h = self.act2(self.norm2(self.conv2(h)))
        return h

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding processor
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Downsampling path (Encoder)
        self.down1 = Block(64, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2) # 64x64 -> 32x32
        
        self.down2 = Block(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2) # 32x32 -> 16x16
        
        self.down3 = Block(128, 256, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2) # 16x16 -> 8x8
        
        # Bottleneck
        self.mid1 = Block(256, 256, time_emb_dim)
        self.mid2 = Block(256, 256, time_emb_dim)
        
        # Upsampling path (Decoder) with Corrected Skip Connections
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.up_block1 = Block(128 + 256, 128, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.up_block2 = Block(64 + 128, 64, time_emb_dim)
        
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) 
        self.up_block3 = Block(64 + 64, 64, time_emb_dim)
        
        # Final output projection
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x0 = self.init_conv(x)
        
        # Down
        d1 = self.down1(x0, t_emb)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2, t_emb)
        p3 = self.pool3(d3)
        
        # Bottleneck
        m = self.mid1(p3, t_emb)
        m = self.mid2(m, t_emb)
        
        # Up + Skip Connections
        u1 = self.up1(m)
        u1 = torch.cat([u1, d3], dim=1) # 16x16 + 16x16
        u1 = self.up_block1(u1, t_emb)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1) # 32x32 + 32x32
        u2 = self.up_block2(u2, t_emb)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1) # 64x64 + 64x64
        u3 = self.up_block3(u3, t_emb)
        
        out = self.out_conv(u3)
        return out

if __name__ == "__main__":
    print("Testing Full U-Net Architecture...")
    
    model = UNet()
    
    dummy_noisy_images = torch.randn(2, 3, 64, 64)
    dummy_timesteps = torch.tensor([10, 500])
    
    predicted_noise = model(dummy_noisy_images, dummy_timesteps)
    
    print(f"Input Noisy Image Shape: {dummy_noisy_images.shape}")
    print(f"Input Timesteps Shape: {dummy_timesteps.shape}")
    print(f"Predicted Noise Output Shape: {predicted_noise.shape}")
    
    if dummy_noisy_images.shape == predicted_noise.shape:
        print("SUCCESS: The U-Net output shape perfectly matches the input image shape. It is ready to predict noise!")
    else:
        print("ERROR: Shapes do not match.")