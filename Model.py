import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze operation (global average pooling)
        channel_weights = self.avg_pool(x).view(b, c)
        
        # Excitation operation (fully connected layers)
        channel_weights = self.fc1(channel_weights)
        
        channel_weights = self.relu(channel_weights)
        
        channel_weights = self.fc2(channel_weights)
        
        channel_weights = self.sigmoid(channel_weights)
        
        # Reshape channel weights for broadcasting
        channel_weights = channel_weights.view(b, c, 1, 1)
        
        # Scale the input features
        x = x * channel_weights
        
        return x

class EfficientCNN(nn.Module):
    def __init__(self, in_channels=10):
        super(EfficientCNN, self).__init__()
        
        # 3D Convolutional Layers
        self.conv3d_layer1 = nn.Conv3d(in_channels, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn3d1 = nn.BatchNorm3d(16)
        self.relu3d1 = nn.ReLU()

        self.conv3d_layer2 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn3d2 = nn.BatchNorm3d(32)
        self.relu3d2 = nn.ReLU()

        # 1x1 Convolutional Layer
        self.conv1x1 = nn.Conv3d(32, 3, kernel_size=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        # 3D Convolutional Layers
        x = self.relu3d1(self.bn3d1(self.conv3d_layer1(x)))
        x = self.relu3d2(self.bn3d2(self.conv3d_layer2(x)))

        # 1x1 Convolutional Layer
        x = self.conv1x1(x)

         # Remove the second dimension (size 10) using slice operation
        x = x[:, :, 0, :, :]
        
        return x

# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
    
class VisionTransformerv2(nn.Module):
    def __init__(self,
                 img_size=5,
                 num_channels=3,  # Number of channels in the input hyperspectral image
                 patch_size=1,
                 embedding_dim=768,
                 dropout=0.1,
                 mlp_size=3072,
                 num_transformer_layers=12,
                 num_heads=2,
                 num_classes=16):
        super().__init__()

        # Assert image size is divisible by patch size
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, "Image size must be divisble by patch size."

        # 1. Create patch embedding
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # 2. Create class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

        # 3. Create positional embedding
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # N = (H // P) * (W // P)
        
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

        # 4. Create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)

        # 5. Create stack Transformer Encoder layers (stacked single layers)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                                  nhead=num_heads,
                                                                                                  dim_feedforward=mlp_size,
                                                                                                  activation="gelu",
                                                                                                  batch_first=True,
                                                                                                  norm_first=True),
                                                                                                  num_layers=num_transformer_layers)

        # 7. Create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def reset_parameters(self):
        # Initialize parameters of the layers in your model
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            else:
                print("parameter reset failed")

    def forward(self, x):
      # Get some dimensions from x
      batch_size = x.shape[0]

      # Create the patch embedding
      x = self.patch_embedding(x)
      
      # First, expand the class token across the batch size
      class_token = self.class_token.expand(batch_size, -1, -1) # "-1" means infer the dimension

      # Prepend the class token to the patch embedding
      x = torch.cat((class_token, x), dim=1)
      
      # Add the positional embedding to patch embedding with class token
      x = self.positional_embedding + x
      
      # Dropout on patch + positional embedding
      x = self.embedding_dropout(x)

      # Pass embedding through Transformer Encoder stack
      x = self.transformer_encoder(x)

      # Pass 0th index of x through MLP head
      x = self.mlp_head(x[:, 0])

      return x

# Define the custom image classification model
class CombinedModel(nn.Module):
    def __init__(self, num_channels, reduction_ratio, num_classes):
        super(CombinedModel, self).__init__()
        self.w = 0
        self.h = 0

        self.se_block = SEBlock(in_channels=num_channels, reduction_ratio=reduction_ratio)
        
        self.CNNmodel = EfficientCNN(num_channels)

        # Vision Transformer Model
        self.vit_model = self.build_vit_model(num_classes, self.w, self.h)

    def build_vit_model(self, num_classes, w, h):
        # Instantiate the ViT model for pixel-level classification
        img_size = (w, h)
        num_bands = 3  # Adjusted to match the number of bands in the Indian Pines dataset
        patch_size = 1  # Set patch_size to 1 for pixel-level classification

        embedding_dim = 100
        dropout = 0.1
        mlp_size = 3072
        num_transformer_layers = 4
        num_heads = 2
        
        model = VisionTransformerv2(img_size, num_bands, patch_size,
                                    embedding_dim, dropout, mlp_size,
                                    num_transformer_layers, num_heads,
                                    num_classes)
        
        return model

    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [batch, width, channels, height]

        x = self.se_block(x)

        # 3D Convolutional Layers
        x = x.unsqueeze(2)  # Add a new dimension at index 2

        # Reshape and expand the input tensor to the required shape
        x = x.expand(-1, -1, x.shape[1], -1, -1)

        x = self.CNNmodel(x)

        self.h, self.w = x.shape[2], x.shape[3]

        # Vision Transformer Model
        vit_output = self.vit_model(x)

        return vit_output