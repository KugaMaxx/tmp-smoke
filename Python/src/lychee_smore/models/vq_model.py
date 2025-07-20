import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (batch_size, embedding_dim, sequence_length)
        Returns:
            quantized: quantized vectors
            loss: VQ loss
            perplexity: perplexity of the quantization
        """
        # Convert to (batch_size, sequence_length, embedding_dim)
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances between input and embedding vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find closest embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert back to (batch_size, embedding_dim, sequence_length)
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return quantized, loss, perplexity


class Encoder1D(nn.Module):
    """1D Encoder for VQ-VAE"""
    
    def __init__(self, in_channels: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        
        modules = []
        
        # Build encoder layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        # Final convolution to latent dimension
        modules.append(
            nn.Sequential(
                nn.Conv1d(hidden_dims[-1], latent_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(latent_dim)
            )
        )
        
        self.encoder = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder1D(nn.Module):
    """1D Decoder for VQ-VAE"""
    
    def __init__(self, latent_dim: int, hidden_dims: list, out_channels: int):
        super().__init__()
        
        # Reverse the hidden dimensions for decoder
        hidden_dims = list(reversed(hidden_dims))
        
        modules = []
        
        # Build decoder layers
        in_channels = latent_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim
        
        # Final layer to output channels
        modules.append(
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dims[-1], out_channels, kernel_size=3, stride=1, padding=1)
            )
        )
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class VQConfig(PretrainedConfig):
    """
    [`VQConfig`] is the configuration class to store the configuration of a [`VQModel`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    
    model_type = "vq_vae"
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_dims: list = [64, 128, 256],
        latent_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost


class VQModel(PreTrainedModel):
    """
    VQ-VAE Model for 1D data
    """
    
    config_class = VQConfig
    
    def __init__(self, config: VQConfig):
        super().__init__(config)
        
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.hidden_dims = config.hidden_dims
        self.latent_dim = config.latent_dim
        self.num_embeddings = config.num_embeddings
        self.commitment_cost = config.commitment_cost
        
        # Build encoder and decoder
        self.encoder = Encoder1D(self.in_channels, self.hidden_dims, self.latent_dim)
        self.decoder = Decoder1D(self.latent_dim, self.hidden_dims, self.out_channels)
        
        # Vector quantizer
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.latent_dim, self.commitment_cost)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to output"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VQ-VAE
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
            
        Returns:
            Dictionary containing:
                - reconstructed: Reconstructed output
                - perplexity: Quantization perplexity
                - encoded: Encoded latent representation
                - quantized: Quantized latent representation
        """
        # Encode
        encoded = self.encode(x)
        
        # Vector quantization
        quantized, vq_loss, perplexity = self.vq_layer(encoded)
        
        # Decode
        reconstructed = self.decode(quantized)
        
        return {
            'reconstructed': reconstructed,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'encoded': encoded,
            'quantized': quantized
        }
    
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Get quantization codes for input"""
        with torch.no_grad():
            encoded = self.encode(x)
            
            # Convert to (batch_size, sequence_length, embedding_dim)
            encoded = encoded.permute(0, 2, 1).contiguous()
            flat_input = encoded.view(-1, self.latent_dim)
            
            # Calculate distances and get codes
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                        + torch.sum(self.vq_layer.embedding.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.vq_layer.embedding.weight.t()))
            
            codes = torch.argmin(distances, dim=1)
            return codes.view(encoded.shape[0], encoded.shape[1])
    
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from quantization codes"""
        # Get embeddings from codes
        embeddings = self.vq_layer.embedding(codes)
        
        # Convert to (batch_size, embedding_dim, sequence_length)
        embeddings = embeddings.permute(0, 2, 1).contiguous()
        
        # Decode
        return self.decode(embeddings)