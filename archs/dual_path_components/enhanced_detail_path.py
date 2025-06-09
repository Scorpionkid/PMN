import torch
import torch.nn as nn
import torch.nn.functional as F
from ..restormer_components.te_mdta import TextureEnhancedMDTA as TE_MDTA

class SobelFilter(nn.Module):
    """Sobel edge detection filter"""
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Define Sobel filter
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32).reshape(1, 1, 3, 3).repeat(1, 1, 1, 1)

        # Register as buffer, so it will be saved in the model state, but not optimized as a parameter
        self.register_buffer('kernel_x', self.sobel_x)
        self.register_buffer('kernel_y', self.sobel_y)

    def forward(self, x):
        # Ensure the input dimension is correct
        b, c, h, w = x.shape
        x_reshaped = x.view(b*c, 1, h, w)

        # Apply Sobel filter
        edge_x = F.conv2d(x_reshaped, self.kernel_x, padding=1)
        edge_y = F.conv2d(x_reshaped, self.kernel_y, padding=1)

        # Calculate gradient magnitude
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)

        # Restore the original dimension
        return edge.view(b, c, h, w)

class DilatedConvChain(nn.Module):
    """Dilated convolution chain, using convolution cascades with different dilation rates to increase the receptive field"""
    def __init__(self, channels, dilated_rates=None):
        super(DilatedConvChain, self).__init__()
        if dilated_rates is None:
            dilated_rates = [1, 2, 4, 8]

        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=r, dilation=r)
            for r in dilated_rates
        ])
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        res = x
        for conv in self.dilated_convs:
            res = self.activation(conv(res) + res)  # Residual connection
        return res

class EnhancedDetailPath(nn.Module):
    def __init__(self, channels, num_heads=1):
        super(EnhancedDetailPath, self).__init__()

        # 保留原有的DilatedConvChain
        self.dilated_convs = DilatedConvChain(channels)

        # 保留原有的Sobel边缘检测器
        self.edge_detector = SobelFilter()
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1)

        # 新增：MDTA用于全局上下文建模
        self.mdta = TE_MDTA(channels, num_heads=num_heads)

        # 保留原有的注意力生成机制
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, noise_map=None, texture_mask=None):
        # 空洞卷积链处理
        res = self.dilated_convs(x)

        # MDTA增强全局上下文
        global_context = self.mdta(res, texture_mask)

        # 边缘检测
        edge_map = self.edge_detector(x)
        edge_feat = self.edge_conv(edge_map)

        # 生成注意力图 - 结合边缘特征和全局上下文
        attention = self.attention(torch.cat([global_context, edge_feat], dim=1))

        # 噪声自适应调整
        if self.use_noise_map and noise_map is not None:
            noise_factor = torch.sigmoid(4.0 * noise_map - 2.0)
            # 高噪声区域降低纹理敏感度
            attention = attention * (1.0 - noise_factor)

        output = x * attention + x

        return output