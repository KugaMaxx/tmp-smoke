from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput


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