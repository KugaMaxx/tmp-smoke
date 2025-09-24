from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2Tokenizer
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from dalle_pytorch import OpenAIDiscreteVAE, DALLE


def prepare_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "field":
        model_config = FieldConfig()
        model = FieldModel(model_config)
        return model
    elif model_name == "adlstm":
        model_config = ADLSTMConfig()
        model = ADLSTM(model_config)
        return model
    elif model_name == "dalle":
        model_config = DALLEConfig()
        model = DALLEModel(model_config)
        return model
    else:
        raise ValueError(f"Model {model_name} not recognized.")


class FieldConfig(PretrainedConfig):
    """
    Configuration class for FieldModel.
    """
    model_type = "field_model"
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        lstm_hiddens: list = [32, 64],
        lstm_num_layers: list = [1, 2],
        dropout: float = 0.2,
        upsample_dims: list = [32, 128, 96, 64, 48, 32],
        **kwargs
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hiddens = lstm_hiddens
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.upsample_dims = upsample_dims
        
        super().__init__(**kwargs)


class FieldModel(PreTrainedModel):
    """
    Field model for time series to image generation.
    https://journals.sagepub.com/doi/abs/10.1177/1420326X251331180
    """
    config_class = FieldConfig

    def __init__(self, config: FieldConfig):
        super().__init__(config)

        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_num_layers = config.lstm_num_layers
        self.dropout = config.dropout
        self.upsample_dims = config.upsample_dims

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        assert len(self.lstm_hiddens) == len(self.lstm_num_layers), "lstm_hiddens and lstm_num_layers must have the same length"

        current_input_size = self.in_channels
        for i, (h_size, n_layers) in enumerate(zip(self.lstm_hiddens, self.lstm_num_layers)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=current_input_size,
                    hidden_size=h_size,
                    num_layers=n_layers,
                    batch_first=True
                )
            )
            current_input_size = h_size

        # Create upsampling layers
        modules = []
        current_channels = 1
        for i, hidden_channels in enumerate(self.upsample_dims):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(0.2, True),
                )
            )
            current_channels = hidden_channels

        modules.append(
            nn.Sequential(
                nn.Conv2d(self.upsample_dims[-1], self.out_channels, kernel_size=5, stride=1, padding=2, bias=True),  # 使用5x5卷积进一步平滑
                nn.Tanh()
            )
        )

        self.upsamplers = nn.Sequential(*modules)

    def forward(
        self,
        inputs: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        # inputs: (batch_size, num_sensors, sequence_length)
        x = inputs.transpose(1, 2)

        # Pass through all LSTM layers
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)

        # Get last timestep 
        x = x[:, -1, :]  # (batch_size, final_lstm_hidden)

        # Reshape for CNN input
        hidden_size = (
            int(pixel_values.size(2) / (2 ** len(self.upsample_dims))),
            int(pixel_values.size(3) / (2 ** len(self.upsample_dims)))
        )
        
        if hidden_size[0] * hidden_size[1] < x.size(1):
            # Truncate if too large
            x = x[:, :hidden_size[0] * hidden_size[1]]
            print(f"Warning: Truncating LSTM output from {x.size(1)} to {hidden_size[0] * hidden_size[1]} to fit the feature map size.")

        elif hidden_size[0] * hidden_size[1] > x.size(1):
            # Zero-pad if too small
            pad_size = hidden_size[0] * hidden_size[1] - x.size(1)
            pad = torch.zeros(x.size(0), pad_size, device=x.device)
            x = torch.cat([x, pad], dim=1)
            print(f"Warning: Padding LSTM output from {x.size(1)} to {hidden_size[0] * hidden_size[1]} to fit the feature map size.")

        x = x.view(x.size(0), 1, hidden_size[0], hidden_size[1])

        # Upsampling path
        outputs = self.upsamplers(x)
        
        # Calculate loss if pixel_values provided
        loss = F.mse_loss(outputs, pixel_values)

        # Return dictionary
        return {
            'outputs': outputs,
            'loss': loss
        }


class ADLSTMConfig(PretrainedConfig):
    """
    Configuration class for ADLSTM model.
    """
    model_type = "adlstm"
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        lstm_hiddens: list = [32, 64],
        lstm_num_layers: list = [1, 1],
        hidden_dims: list = [256, 512, 1024],
        upsample_dims: list = [128, 96, 64, 32],
        **kwargs
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hiddens = lstm_hiddens
        self.lstm_num_layers = lstm_num_layers
        self.hidden_dims = hidden_dims
        self.upsample_dims = upsample_dims
        
        super().__init__(**kwargs)


class ADLSTM(PreTrainedModel):
    """
    ADLSTM model for time series to image generation.
    https://www.sciencedirect.com/science/article/pii/S1474034625000102
    """
    config_class = ADLSTMConfig
    
    def __init__(self, config: ADLSTMConfig):
        super().__init__(config)
        
        # Store config parameters
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_num_layers = config.lstm_num_layers
        self.hidden_dims = config.hidden_dims
        self.upsample_dims = config.upsample_dims

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        assert len(self.lstm_hiddens) == len(self.lstm_num_layers), "lstm_hiddens and lstm_num_layers must have the same length"

        current_input_size = self.in_channels
        for i, (h_size, n_layers) in enumerate(zip(self.lstm_hiddens, self.lstm_num_layers)):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=current_input_size,
                    hidden_size=h_size,
                    num_layers=n_layers,
                    batch_first=True
                )
            )
            current_input_size = h_size

        # Create Dense layers
        modules = []

        current_input_size = self.lstm_hiddens[-1]
        for h_size in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(current_input_size, h_size),
                    nn.BatchNorm1d(h_size),
                    nn.Tanh()
                )
            )
            current_input_size = h_size

        self.denses = nn.Sequential(*modules)

        # Create upsampling layers
        modules = []
        current_channels = 1
        for i, hidden_channels in enumerate(self.upsample_dims):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(0.2, True),
                )
                # # More robust to avoid checkerboard artifacts
                # nn.Sequential(
                #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                #     nn.Conv2d(current_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                #     nn.LeakyReLU(0.2, True),
                # )
            )
            current_channels = hidden_channels

        modules.append(
            nn.Sequential(
                nn.Conv2d(self.upsample_dims[-1], self.out_channels, kernel_size=5, stride=1, padding=2, bias=True),  # 使用5x5卷积进一步平滑
                nn.Tanh()
            )
        )

        self.upsamplers = nn.Sequential(*modules)

    def forward(
        self,
        inputs: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        # inputs: (batch_size, num_sensors, sequence_length)
        x = inputs.transpose(1, 2)

        # Pass through all LSTM layers
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)

        # Get last timestep 
        x = x[:, -1, :]  # (batch_size, final_lstm_hidden)

        # Pass through all Dense layers
        x = self.denses(x)

        # Reshape for CNN input
        hidden_size = (
            int(pixel_values.size(2) / (2 ** len(self.upsample_dims))),
            int(pixel_values.size(3) / (2 ** len(self.upsample_dims)))
        )
        
        if hidden_size[0] * hidden_size[1] < x.size(1):
            # Truncate if too large
            x = x[:, :hidden_size[0] * hidden_size[1]]
            print(f"Warning: Truncating LSTM output from {x.size(1)} to {hidden_size[0] * hidden_size[1]} to fit the feature map size.")

        elif hidden_size[0] * hidden_size[1] > x.size(1):
            # Zero-pad if too small
            pad_size = hidden_size[0] * hidden_size[1] - x.size(1)
            pad = torch.zeros(x.size(0), pad_size, device=x.device)
            x = torch.cat([x, pad], dim=1)
            print(f"Warning: Padding LSTM output from {x.size(1)} to {hidden_size[0] * hidden_size[1]} to fit the feature map size.")

        x = x.view(x.size(0), 1, hidden_size[0], hidden_size[1])

        # Upsampling path
        outputs = self.upsamplers(x)
        
        # Calculate loss if pixel_values provided
        loss = F.mse_loss(outputs, pixel_values)

        # Return dictionary
        return {
            'outputs': outputs,
            'loss': loss
        }


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
