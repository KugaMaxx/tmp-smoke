from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from dalle_pytorch import OpenAIDiscreteVAE, DALLE


class DALLEConfig(PretrainedConfig):
    """
    Configuration class for DALL-E model.
    """
    model_type = "dalle"
    
    def __init__(
        self,
        text_seq_len = 24,          # text sequence length
        num_text_tokens = 10000,    # vocab size for text
        dim: int = 512,             # model dimension
        depth = 6,                  # should aim to be 64
        heads = 16,                 # attention heads
        dim_head = 64,              # attention head dimension
        attn_dropout = 0.1,         # attention dropout
        ff_dropout = 0.1,           # feedforward dropout
        **kwargs
    ):
        self.text_seq_len = text_seq_len
        self.num_text_tokens = num_text_tokens
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        
        super().__init__(**kwargs)


class DALLEModel(PreTrainedModel):
    """
    Wrapper class for DALLE-pytorch.
    https://github.com/lucidrains/DALLE-pytorch
    """
    config_class = DALLEConfig
    base_model_prefix = "dalle"

    def __init__(self, config: DALLEConfig):
        super().__init__(config)

        # Store config parameters
        self.config = config
        self.text_seq_len = config.text_seq_len
        self.num_text_tokens = config.num_text_tokens
        self.dim = config.dim
        self.depth = config.depth
        self.heads = config.heads
        self.dim_head = config.dim_head
        self.attn_dropout = config.attn_dropout
        self.ff_dropout = config.ff_dropout

        # # Numbering tokenizer is more useful for our case
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pretrained VAE on 256x256 images
        self.vae = OpenAIDiscreteVAE()

        # Initialize DALL-E model
        self.dalle = DALLE(
            dim = self.dim,
            vae = self.vae,
            num_text_tokens = self.num_text_tokens,
            text_seq_len = self.text_seq_len,
            depth = self.depth,
            heads = self.heads,
            dim_head = self.dim_head,
            attn_dropout = self.attn_dropout,
            ff_dropout = self.ff_dropout
        )

    def forward(
        self,
        inputs: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:

        # Normalize pixel values to [0, 1]
        pixel_values = (pixel_values + 1) / 2

        # Because pretrained VAE provided is only trained on 256x256 images
        # we need to downsample to the desired size
        pixel_values = F.interpolate(pixel_values, size=(256, 256), mode='bilinear', align_corners=False)

        # Numbering tokenizer
        text_seq_len = 24
        hist_len = int(text_seq_len / inputs.size(1))

        assert hist_len > 0, f"text_seq_len ({text_seq_len}) must be >= number of sensors {inputs.size(1)}"

        text = inputs[..., :hist_len].contiguous().view(inputs.size(0), -1)
        text = (text * 1E4).long()

        # Inference
        loss = self.dalle(text, pixel_values, return_loss = True)

        # Image will be None if training to promote efficiency
        if self.dalle.training:
            image = None
        else:
            image = self.dalle.generate_images(text)
            image = (image * 2) - 1
            image = F.interpolate(
                image,
                size=(pixel_values.size(2), pixel_values.size(3)),
                mode="bilinear",
                align_corners=False,
            )

        return {
            'outputs': image,
            'loss': loss
        }