import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseLevelNetwork(nn.Module):
    """Noise level estimation network"""
    def __init__(self, in_channels=4):
        super(NoiseLevelNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class AdaptiveUnsharpMask(nn.Module):
    """Adaptive sharpening module"""
    def __init__(self, in_channels=4):
        super(AdaptiveUnsharpMask, self).__init__()

        # Sharpening strength prediction
        self.sharpness_strength = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Predict sharpening strength
        strength = self.sharpness_strength(x)

        # Generate blurred version
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # Calculate high-frequency component
        high_freq = x - blurred

        # Add sharpening
        sharpened = x + high_freq * strength

        return sharpened

class EnhancedSharpnessModule(nn.Module):
    """增强型多尺度自适应锐化模块"""
    def __init__(self, in_channels=4, scales=[5, 3]):
        super(EnhancedSharpnessModule, self).__init__()
        self.scales = scales

        # 每个尺度的锐化强度预测
        self.scale_strengths = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, len(scales), 3, padding=1),
            nn.Sigmoid()
        )

        # 阈值调整因子预测
        self.threshold_modulator = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 获取多尺度锐化强度和阈值调整因子
        strengths = self.scale_strengths(x)
        thresh_factor = 1.0 + self.threshold_modulator(x)  # 范围[1.0-2.0]

        # 逐尺度锐化处理
        result = x
        for i, kernel_size in enumerate(self.scales):
            padding = kernel_size // 2
            blurred = F.avg_pool2d(result, kernel_size=kernel_size, stride=1, padding=padding)

            high_freq = result - blurred

            local_mean = torch.mean(torch.abs(high_freq), dim=1, keepdim=True)
            local_threshold = local_mean * thresh_factor

            # 自适应调整锐化强度 - 防止过度锐化已锐利区域
            adaptive_factor = torch.sigmoid(1.0 - torch.abs(high_freq) / (local_threshold + 1e-6))
            effective_strength = strengths[:, i:i+1] * adaptive_factor

            result = result + high_freq * effective_strength

        return result


class SharpnessRecovery(nn.Module):
    def __init__(self, in_channels=4, use_noise_map=False, use_texture_mask=False, sharpness_texture_boost=0.3):
        super(SharpnessRecovery, self).__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask
        if not use_noise_map:
            self.noise_estimator = NoiseLevelNetwork(in_channels)
        # self.adaptive_sharp = AdaptiveUnsharpMask(in_channels)
        self.adaptive_sharp = EnhancedSharpnessModule(in_channels, scales=[5, 3])

        # control sharpened strength
        self.safe_mode = True

        if use_texture_mask:
            self.texture_enhance = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.GroupNorm(2, 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(8, 1, 3, padding=1),
                nn.Sigmoid()
            )
            self.texture_boost = nn.Parameter(torch.tensor(sharpness_texture_boost))

    def forward(self, x, noise_map=None, texture_mask=None):

        if self.use_texture_mask and texture_mask is not None:
            boost_factor = torch.clamp(self.texture_boost, 0.1, 0.9)


            if self.use_noise_map and noise_map is not None:
                # 噪声调制因子：0.3到1.0范围，不会完全消除锐化
                noise_factor = 0.3 + 0.7 * torch.exp(-4.0 * noise_map)

                # scheme1:
                # 使用网络增强纹理掩码的表现力
                enhanced_texture = self.texture_enhance(texture_mask)
                base_sharpness = (texture_mask + enhanced_texture) / 2.0
                boosted_sharpness = base_sharpness * boost_factor
                sharp_mask = boosted_sharpness * noise_factor

                # scheme2:
                # sharp_mask = texture_mask * boost_factor * noise_factor

            sharp_mask = torch.clamp(sharp_mask, 0.05, 0.95)

        elif self.use_noise_map and noise_map is not None:
            # Use simple inversion and cropping functions.
            sharp_mask = torch.clamp(1.0 - noise_map * 5.0, 0.0, 1.0)
        else:
            noise_level = self.noise_estimator(x)
            sharp_mask = 1.0 - noise_level

        # Generate sharpness mask
        # sharpen in low noise regions, keep in high noise regions
        sharpened = self.adaptive_sharp(x)

        if self.safe_mode:
            # 计算锐化幅度
            sharp_diff = sharpened - x
            max_diff = torch.max(torch.abs(sharp_diff))

            # 如果锐化效果过强，进行缩放
            if max_diff > 0.5:  # 阈值可以根据需要调整
                scale_factor = 0.5 / max_diff
                sharpened = x + sharp_diff * scale_factor
        result = sharpened * sharp_mask + x * (1.0 - sharp_mask)
        result = torch.clamp(result, min=0.0)
        return result