import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CelebADataset(Dataset):
    def __init__(self, image_dir, attr_csv_path, image_size=64):
        self.image_dir = image_dir
        
        print("Loading attributes CSV...")
        self.attr_df = pd.read_csv(attr_csv_path)
        
        # The first column is 'image_id', the rest are attributes
        self.image_names = self.attr_df['image_id'].values
        self.attributes = self.attr_df.drop(columns=['image_id'])
        self.attr_names = self.attributes.columns.tolist()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),             
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_names)
    
    def generate_prompt(self, attr_row):
        """Converts binary attributes into a text prompt."""
        positive_attrs = []
        for attr_name, value in zip(self.attr_names, attr_row):
            if value == 1:
                clean_name = attr_name.replace("_", " ")
                positive_attrs.append(clean_name)
        
        if not positive_attrs:
            return "A photo of a person face"
            
        return "A photo of a person face with " + ", ".join(positive_attrs)

    def __getitem__(self, idx):
        # Get Image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        tensor_image = self.transform(image)
        
        # Get Text Prompt
        attr_row = self.attributes.iloc[idx].values
        prompt = self.generate_prompt(attr_row)
        
        return tensor_image, prompt

if __name__ == "__main__":
    DATA_DIR = "/home/ritesh/Work_2026/ai_ml/projects/archive/img_align_celeba/img_align_celeba"
    CSV_PATH = "/home/ritesh/Work_2026/ai_ml/projects/archive/list_attr_celeba.csv"
    
    if not os.path.exists(DATA_DIR) or not os.path.exists(CSV_PATH):
        print("WARNING: Please ensure your image folder and CSV file are in the correct paths.")
    else:
        IMAGE_SIZE = 64 
        BATCH_SIZE = 4
        
        dataset = CelebADataset(image_dir=DATA_DIR, attr_csv_path=CSV_PATH, image_size=IMAGE_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        print(f"Total images found: {len(dataset)}")
        
        for images, prompts in dataloader:
            print("\n--- Batch Info ---")
            print(f"Image Batch Shape: {images.shape}")
            print(f"Max pixel value: {images.max().item():.4f}") 
            print(f"Min pixel value: {images.min().item():.4f}") 
            
            print("\n--- Generated Prompts ---")
            for i in range(BATCH_SIZE):
                print(f"{i+1}: {prompts[i]}")
            break