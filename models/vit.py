"""
Vision Transformers

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size  # Assuming 3 channels (RGB)

        # Patch embeddings and position embeddings
        self.patch_embeddings = nn.Linear(patch_dim, dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        # Classification head
        self.fc = nn.Linear(dim, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, 3 * self.patch_size * self.patch_size)  # (B, N, patch_dim)
        
        # Patch embedding
        x = self.patch_embeddings(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, dim)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        # Transformer
        x = self.transformer(x)
        
        # Classification head (use the class token output)
        cls_output = x[:, 0]
        out = self.fc(cls_output)
        
        return out

if __name__ == '__main__':
    # Example usage
    img = torch.randn(1, 3, 224, 224)  # Batch of 1, 3 channels (RGB), 224x224 image
    model = VisionTransformer()
    logits = model(img)
    print(logits.shape)  # Should output torch.Size([1, 1000])