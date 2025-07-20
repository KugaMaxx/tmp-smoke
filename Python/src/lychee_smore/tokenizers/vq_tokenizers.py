import os
import re
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn

from tokenizers import AddedToken
from transformers.tokenization_utils import PreTrainedTokenizer
from diffusers.configuration_utils import FrozenDict

from ..models import VQConfig, VQModel


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Reversible Instance Normalization (RevIN) for input normalization.

        Args:
            num_features (int): Number of features to normalize.
            eps (float): Small value to avoid division by zero.
            affine (bool): Whether to use learnable affine parameters.
        Reference:
            https://github.com/ts-kim/RevIN/blob/master/RevIN.py
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class VQTokenizer(PreTrainedTokenizer):
    """
    A tokenizer that uses a VQ-VAE (Vector Quantized Variational Autoencoder) for tokenization.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vq_config: Optional[Union[Dict[str, Any], VQConfig]] = None,
        vq_model: VQModel = None,
        text_length: int = 256,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  
        *args, **kwargs
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token

        super().__init__(
            text_length=text_length,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

        if vq_model is None:
            # If no vq_config is provided, use the default VQConfig
            if vq_config is None:
                self.vq_config = VQConfig()
                self.vq_model = VQModel(self.vq_config)

            # If vq_config is already a VQConfig instance, use it directly
            elif isinstance(vq_config, VQConfig):
                self.vq_config = vq_config
                self.vq_model = VQModel(self.vq_config)
            
            # If vq_config is a dictionary, create a VQConfig from it
            elif isinstance(vq_config, dict):
                self.vq_config = VQConfig(**vq_config)
                self.vq_model = VQModel(self.vq_config)
        
            # Otherwise, raise an error for invalid type
            else:
                raise ValueError("vq_config must be a VQConfig instance or a dictionary.")
            
        else:
            self.vq_model = vq_model
            self.vq_config = self.vq_model.config

        # Set VQ model to evaluation mode to ensure consistent tokenization
        self.vq_model.eval()

        # Set RevIN for input normalization
        self.revin = RevIN(text_length, eps=1e-5, affine=False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs
    ):
        """
        Load a tokenizer from a pretrained model.
        """
        # Load VQ model and config
        vq_save_directory = os.path.join(pretrained_model_name_or_path, "vq")
        if os.path.exists(vq_save_directory):
            kwargs["vq_model"] = VQModel.from_pretrained(vq_save_directory)
            kwargs["vq_config"] = VQConfig.from_pretrained(vq_save_directory)
        else:
            raise ValueError(f"Subfolder 'vq' does not exist in {pretrained_model_name_or_path}.")
        
        # Call the parent's from_pretrained method to handle the tokenizer loading
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        return tokenizer

    def save_pretrained(self, save_directory: Union[FrozenDict, Dict[str, Any]], **kwargs):
        """
        Save the tokenizer and VQ model to a directory.
        """
        # Call parent's save_pretrained method first
        super().save_pretrained(save_directory, **kwargs)

        # Create sub-directory for VQ model
        vq_save_directory = os.path.join(save_directory, "vq")
        os.makedirs(vq_save_directory, exist_ok=True)
        
        # Save VQ model
        self.vq_model.save_pretrained(vq_save_directory)

        # Save VQ config
        self.vq_config.save_pretrained(vq_save_directory)

    def _tokenize(self, text: Union[str, list]) -> torch.Tensor:
        """
        Tokenizes a time-series string or array into quantized codes.
        Supports input as a string (semicolon/comma separated), list, or torch.Tensor.
        """
        if isinstance(text, str):
            # Use regex to split by semicolon (ignoring surrounding whitespace)
            text = re.split(r'\s*;\s*', text.strip())

            # Extract numbers from each dimension string
            inputs = [[float(num) for num in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', dim)] for dim in text if dim]
            inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = self.revin(inputs, mode='norm')  # Normalize inputs

        elif isinstance(text, (list, tuple)):
            # Convert list/tuple to tensor and add batch dimension if needed
            inputs = torch.tensor(text, dtype=torch.float32)

        else:
            raise ValueError("Input must be a string or list.")

        # Get quantization codes
        inputs = inputs.unsqueeze(0).to(self.vq_model.device)
        tokens = self.vq_model.get_codes(inputs).tolist()

        return tokens

    @property
    def vocab_size(self):
        """Return the vocabulary size of the VQ model."""
        return 1
        # return getattr(self.vq_model, 'num_embeddings', 0)

    def _convert_token_to_id(self, token):
        """Convert a token to its corresponding ID."""
        # For VQ tokenizer, tokens are already IDs
        try:
            return int(token)
        except (ValueError, TypeError):
            return self.unk_token_id

    def _convert_id_to_token(self, index):
        """Convert an ID to its corresponding token."""
        # For VQ tokenizer, IDs are the tokens
        return str(index)

    def get_vocab(self):
        """
        Return the vocabulary as a dictionary.
        Key: decoded data (as string representation)
        Value: codebook index/combination
        """
        vocab = {'1': 1}
            
        return vocab

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary to a file as required by the transformers PreTrainedTokenizer interface.
        For VQTokenizer, we just create a dummy vocab file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vq_vocab.txt"
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token in self.get_vocab():
                f.write(str(token) + "\n")
        return (vocab_file,)
