import torch
import torch.nn as nn
from ..restormer_components.te_mdta import TextureEnhancedMDTA as TE_MDTA

class AttentionGuidedFusion(nn.Module):
    """TE_MDTA引导的双路径融合层，直接利用噪声图"""
    def __init__(self, channels, num_heads=2, use_noise_map=False, use_texture_mask=False):
        super(AttentionGuidedFusion, self).__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask

        # 保持与路径中相同的设计模式 - 使用TE_MDTA分析特征
        self.context_attn = TE_MDTA(channels, num_heads=num_heads)


        # 融合权重生成器 - 使用上下文特征和路径差异信息

        self.fusion_weight = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//2, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # self.fusion_weight = nn.Sequential(
        #     nn.Conv2d(channels, channels//2, 3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(channels//2, channels//4, 3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(channels//4, 1, 3, padding=1),
        #     nn.Sigmoid()
        # )

        # 噪声系数参数 - 可学习参数用于调整噪声影响
        if use_noise_map:
            self.noise_scale = nn.Parameter(torch.tensor(3.0))  # 初始缩放系数
            # self.noise_bias = nn.Parameter(torch.tensor(0.2))   # 初始偏置系数

    def forward(self, detail_path, denoise_path, features, noise_map=None, texture_mask=None):
        """
        Args:
            detail_path: 细节路径输出
            denoise_path: 降噪路径输出
            features: 原始特征
            noise_map: 噪声图
            texture_mask: 纹理掩码
        """
        path_diff = torch.abs(detail_path - denoise_path)

        context_features = self.context_attn(features, texture_mask)

        weight_input = torch.cat([context_features, path_diff], dim=1)
        alpha = self.fusion_weight(weight_input)

        if self.use_noise_map and noise_map is not None:
            # 限制系数在合理范围内
            scale = torch.clamp(self.noise_scale, 1.0, 10.0)
            # bias = torch.clamp(self.noise_bias, 0.0, 0.5)

            # 使用简单的函数映射噪声到权重调整系数
            # 公式: 1.0 - (scale * noise_map - bias).clamp(0, 1 - bias)
            # 高噪声产生小系数，低噪声产生接近1的系数
            # 修复: 使用两步clamp，避免参数类型不匹配
            # noise_effect = scale * noise_map - bias
            # noise_effect = torch.clamp(noise_effect, min=0.0)  # 先应用最小值限制
            # max_val = 1.0 - bias  # 计算最大值限制
            # noise_effect = torch.clamp(noise_effect, max=max_val)  # 再应用最大值限制
            # noise_weight = 1.0 - noise_effect

            # 应用噪声权重调整 - 高噪声区域偏向降噪路径
            noise_weight = torch.exp(-scale * noise_map)
            alpha = alpha * noise_weight

        return alpha * detail_path + (1.0 - alpha) * denoise_path