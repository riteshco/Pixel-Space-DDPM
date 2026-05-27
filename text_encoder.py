import torch
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder:
    def __init__(self, device="cpu"):
        self.device = device
        model_id = "openai/clip-vit-base-patch32"
        
        print(f"Downloading/Loading CLIP model ({model_id})...")
        print("This might take a minute on the very first run to download the weights.")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.model = CLIPTextModel.from_pretrained(model_id).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def encode(self, texts):
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        """We grab the 'pooler_output' which is a single condensed vector 
        representing the entire meaning of the sentence. """
        text_embeddings = outputs.pooler_output
        return text_embeddings

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    encoder = TextEncoder(device=device)
    
    sample_prompts = [
        "A photo of a person face with Eyeglasses, Smiling",
        "A photo of a person face with Bangs, No Beard, Young"
    ]
    
    print(f"\nEncoding {len(sample_prompts)} prompts...")
    embeddings = encoder.encode(sample_prompts)
    
    print("\n--- Output Info ---")
    print(f"Input Prompts: {sample_prompts}")
    print(f"Embedding Tensor Shape: {embeddings.shape}")
    print(f"Expected Shape: [2, 512] (Batch Size, Embedding Dimension)")
    
    if embeddings.shape == (len(sample_prompts), 512):
        print("\nSUCCESS: Text has been successfully translated into mathematics!")