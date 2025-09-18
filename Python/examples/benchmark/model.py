import torch
import torch.nn as nn
import torch.nn.functional as F


class ADLSTMFire(nn.Module):
    def __init__(self, input_features=3, lstm_hidden1=100, lstm_hidden2=10, dropout_rate=0.1):
        super(ADLSTMFire, self).__init__()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_features, lstm_hidden1, batch_first=True, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(lstm_hidden1, lstm_hidden2, batch_first=True, dropout=dropout_rate)
        
        # Dense layers with BatchNorm
        self.dense1 = nn.Linear(lstm_hidden2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.dense2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.dense3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        # 增加中间层来缓和维度跳跃，使用更大的初始特征图
        self.dense4 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        # 添加额外的全连接层来增大特征图
        self.dense5 = nn.Linear(512, 8192)  # 8192 = 32 * 16 * 16 (从16x16开始，减少上采样层数)
        self.bn5 = nn.BatchNorm1d(8192)
        self.dropout2 = nn.Dropout(0.2)
        
        # 更平滑的上采样路径，减少棋盘效应
        # 从 32x16x16 开始，逐步上采样到 512x512x3
        # 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512 (5层而不是6层)
        
        # 第一层：上采样+卷积 16x16 -> 32x32
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(128)
        
        # 第二层：结合上采样+卷积减少棋盘效应 32x32 -> 64x64
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(96)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # 第三层：上采样+卷积 64x64 -> 128x128
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(64)
        
        # 第四层：上采样+卷积 128x128 -> 256x256
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1)
        self.bn_conv4 = nn.BatchNorm2d(48)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # 第五层：上采样+卷积 256x256 -> 512x512
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1)
        self.bn_conv5 = nn.BatchNorm2d(32)
        
        # 最终输出层：直接生成3通道输出
        self.conv_final = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # 对于LeakyReLU使用合适的初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                # LSTM权重初始化
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # 设置forget gate bias为1（LSTM常见做法）
                        n = param.size(0)
                        param.data[(n//4):(n//2)].fill_(1)

    def forward(self, inputs, pixel_values):
        # (batch_size, num_sensors, sequence_length) -> (batch_size, sequence_length, num_sensors)
        x = inputs.transpose(1, 2)

        # LSTM part
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = lstm_out2[:, -1, :]  # Take the last timestep output
        
        # Dense layers - use LeakyReLU for better gradient flow
        x = F.leaky_relu(self.bn1(self.dense1(lstm_out2)), 0.2)
        x = F.leaky_relu(self.bn2(self.dense2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.dense3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.dense4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.dense5(x)), 0.2)
        x = self.dropout2(x)
        
        # Reshape for CNN input (32, 16, 16) - 32 channels, 16x16 spatial
        x = x.view(-1, 32, 16, 16)
        
        # 改进的上采样路径，减少棋盘效应
        # 第一层：上采样+卷积 16x16 -> 32x32
        x = self.upsample1(x)
        x = F.leaky_relu(self.bn_conv1(self.conv1(x)), 0.2)
        
        # 第二层：上采样+卷积 32x32 -> 64x64
        x = self.upsample2(x)
        x = F.leaky_relu(self.bn_conv2(self.conv2(x)), 0.2)
        x = self.dropout3(x)
        
        # 第三层：上采样+卷积 64x64 -> 128x128
        x = self.upsample3(x)
        x = F.leaky_relu(self.bn_conv3(self.conv3(x)), 0.2)
        
        # 第四层：上采样+卷积 128x128 -> 256x256
        x = self.upsample4(x)
        x = F.leaky_relu(self.bn_conv4(self.conv4(x)), 0.2)
        x = self.dropout4(x)
        
        # 第五层：上采样+卷积 256x256 -> 512x512
        x = self.upsample5(x)
        x = F.leaky_relu(self.bn_conv5(self.conv5(x)), 0.2)
        
        # 最终输出层：直接生成3通道输出，使用tanh确保输出在[-1,1]范围
        outputs = torch.tanh(self.conv_final(x))
        
        return {
            'outputs': outputs,
            'loss': F.mse_loss(outputs, pixel_values)
        }
