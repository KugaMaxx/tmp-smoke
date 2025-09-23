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

    if model_name == "adlstm":
        model_config = ADLSTMConfig()
        model = ADLSTM(model_config)
        return model
    elif model_name == "dalle":
        model_config = DALLEConfig()
        model = DALLEModel(model_config)
        return model
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    

# class FieldModel(nn.Module):
#     def __init__(self, input_size=3, lstm_hiddens=[512, 1024], num_layers=[1, 2], dropout=0.2):
#         super().__init__()
        
#         # 增加LSTM的hidden_size以获得更丰富的特征表示
#         self.lstm1 = nn.LSTM(input_size=3, hidden_size=512, num_layers=1, batch_first=True)
#         self.lstm2 = nn.LSTM(input_size=512, hidden_size=1024, num_layers=2, batch_first=True, dropout=dropout)

#         # 从32x32开始上采样，减少层数从8层到4层
#         self.main = nn.ModuleList([
#             # 第1层: 32x32 -> 64x64，保持较多通道数
#             nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, True),
#             ),
            
#             # 第2层: 64x64 -> 128x128
#             nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, True),
#             ),
            
#             # 第3层: 128x128 -> 256x256
#             nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, True),
#             ),
            
#             # 第4层: 256x256 -> 512x512，最终输出层
#             nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.2, True),
#             # 最终输出层：生成3通道图像
#             nn.Conv2d(16, 3, kernel_size=5, stride=1, padding=2, bias=True),  # 使用5x5卷积进一步平滑
#             nn.Tanh()  # 确保输出在[-1,1]范围内
#             ),
#         ])

#     def forward(self, inputs, pixel_values):
#         x = inputs.transpose(1, 2)  # (batch_size, sequence_length, num_sensors)

#         x, _ = self.lstm1(x)  # LSTM第一层
#         x, _ = self.lstm2(x)  # LSTM第二层
        
#         x = x[:, -1, :]  # 取最后一个time step的输出 (batch_size, 1024)
        
#         # 直接将LSTM输出reshape为特征图，不使用全连接层
#         # 将1024维reshape为合理的特征图尺寸，例如 (32, 32) 的单通道特征图
#         # 这里我们需要选择一个合适的初始尺寸，让1024能够整除
#         # 1024 = 32 * 32, 所以我们可以reshape为 (1, 32, 32)
#         x = x.view(x.size(0), 1, 32, 32)
        
#         # 首先扩展通道数到32通道
#         x = F.interpolate(x.repeat(1, 32, 1, 1), size=(32, 32), mode='bilinear', align_corners=False)
        
#         # 依次通过每个上采样模块 (4层而不是8层)
#         for layer in self.main:
#             x = layer(x)
        
#         outputs = x

#         return {
#             'outputs': outputs,
#             'loss': F.mse_loss(outputs, pixel_values)
#         }


class ADLSTMConfig(PretrainedConfig):
    """
    Configuration class for ADLSTM model.
    """
    model_type = "adlstm"
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        lstm_hidden_size: list = [512, 1024],
        lstm_num_layers: list = [1, 1],
        hidden_dims: list = [256, 512, 1024, 2048, 8192],
        upsample_dims: list = [32, 128, 96, 64, 48, 32],
        **kwargs
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.hidden_dims = hidden_dims
        self.upsample_dims = upsample_dims
        
        super().__init__(**kwargs)


class ADLSTM(PreTrainedModel):
    """
    ADLSTM model for time series to image generation.
    """
    config_class = ADLSTMConfig
    
    def __init__(self, config: ADLSTMConfig):
        super().__init__(config)
        
        # Store config parameters
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm_num_layers = config.lstm_num_layers
        self.hidden_dims = config.hidden_dims
        self.upsample_dims = config.upsample_dims

        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        assert len(self.lstm_hidden_size) == len(self.lstm_num_layers), "lstm_hidden_size and lstm_num_layers must have the same length"

        current_input_size = self.in_channels
        for i, (h_size, n_layers) in enumerate(zip(self.lstm_hidden_size, self.lstm_num_layers)):
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

        current_input_size = self.lstm_hidden_size[-1]
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
        current_channels = self.upsample_dims[0]
        for i, hidden_channels in enumerate(self.upsample_dims[1:]):
            modules.append(
                nn.Sequential(
                    # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    # nn.Conv2d(current_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
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

        # Pass through all Dense layers
        x = self.denses(x)

        # Reshape for CNN input
        hidden_size = (
            int(pixel_values.size(2) / (2 ** (len(self.upsample_dims) - 1))),
            int(pixel_values.size(3) / (2 ** (len(self.upsample_dims) - 1)))
        )
        x = x.view(x.size(0), -1, hidden_size[0], hidden_size[1])

        # Adjust channels if necessary
        if self.upsample_dims[0] > x.size(1):
            pad_channels = self.upsample_dims[0] - x.size(1)
            pad = torch.zeros(x.size(0), pad_channels, x.size(2), x.size(3), device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif self.upsample_dims[0] < x.size(1):
            x = x[:, :self.upsample_dims[0], :, :]

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
        image_size: int = 512,
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
        self.image_size = image_size
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
        self.image_size = config.image_size
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

            # Because pretrained VAE provided is only trained on 256x256 images
            # we need to upsample to the desired size
            if self.image_size != 256:
                image = F.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        return {
            'outputs': image,
            'loss': loss
        }
