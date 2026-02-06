"""
MultiScaleVGGFreqDiscriminator - 多尺度VGG频域判别器
用于DualPathNet对抗训练，结合空间域和频域特征判别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """VGG特征提取器，用于多尺度判别"""
    def __init__(self, feature_layers=[2, 7, 12, 21, 30], use_bn=False, use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        
        if use_bn:
            model = models.vgg19_bn(pretrained=True)
        else:
            model = models.vgg19(pretrained=True)
        
        # 如果输入是4通道(RAW RGGB)，修改第一层
        if use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406, 0.406]).view(1, 4, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225, 0.225]).view(1, 4, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        
        self.feature_layers = feature_layers
        self.features = nn.Sequential(*list(model.features.children())[:max(feature_layers) + 1])
        
        # 修改第一层以接受4通道输入
        first_conv = self.features[0]
        self.features[0] = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        
        # 将前3个通道的权重复制过来，第4个通道用绿色通道的平均值
        with torch.no_grad():
            self.features[0].weight[:, :3] = first_conv.weight
            self.features[0].weight[:, 3] = first_conv.weight[:, 1]  # 使用绿色通道
        
        # 冻结VGG参数
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """提取多尺度VGG特征"""
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features


class FrequencyDiscriminatorHead(nn.Module):
    """频域判别头 - 分析频谱特性"""
    def __init__(self, in_channels, nf=64):
        super(FrequencyDiscriminatorHead, self).__init__()
        
        # 频谱特征提取
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, nf, 3, 1, 1),  # *2 因为有实部和虚部
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 判别头
        self.classifier = nn.Sequential(
            nn.Conv2d(nf, nf // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf // 2, 1, 3, 1, 1)
        )
    
    def forward(self, x):
        """
        输入: spatial domain feature
        输出: frequency domain discrimination result
        """
        # FFT变换到频域
        freq = torch.fft.rfft2(x, norm='ortho')
        
        # 分离实部和虚部
        freq_real = freq.real
        freq_imag = freq.imag
        
        # 拼接实部和虚部
        freq_concat = torch.cat([freq_real, freq_imag], dim=1)
        
        # 频域特征提理
        freq_feat = self.freq_conv(freq_concat)
        
        # 判别
        out = self.classifier(freq_feat)
        
        return out


class SpatialDiscriminatorHead(nn.Module):
    """空间域判别头 - 基于VGG特征判别"""
    def __init__(self, in_channels, nf=64):
        super(SpatialDiscriminatorHead, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, 3, 2, 1),  # downsample
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf // 2, 1, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.conv(x)


class MultiScaleVGGFreqDiscriminator(nn.Module):
    """
    多尺度VGG频域判别器
    
    特点:
    1. 多尺度输入: 处理原始图像和下采样版本
    2. VGG特征提取: 使用预训练VGG提取特征
    3. 双路径判别:
       - 空间域路径: 基于VGG特征判别
       - 频域路径: FFT频谱分析判别
    4. 多层判别: 在不同尺度和不同VGG层进行判别
    
    Args:
        in_channels: 输入通道数 (默认4, 对应RAW RGGB)
        nf: 基础特征数 (默认64)
        num_scales: 尺度数量 (默认3)
        vgg_layers: VGG特征层 (默认[2, 7, 12, 21, 30])
        use_freq_discriminator: 是否使用频域判别 (默认True)
    """
    def __init__(self, 
                 in_channels=4, 
                 nf=64, 
                 num_scales=3,
                 vgg_layers=[2, 7, 12, 21, 30],
                 use_freq_discriminator=True):
        super(MultiScaleVGGFreqDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        self.use_freq_discriminator = use_freq_discriminator
        
        # VGG特征提取器
        self.vgg_extractor = VGGFeatureExtractor(
            feature_layers=vgg_layers,
            use_bn=False,
            use_input_norm=True
        )
        
        # 获取VGG各层的通道数
        vgg_channels = {
            2: 64,    # conv1_2
            7: 128,   # conv2_2
            12: 256,  # conv3_2
            21: 512,  # conv4_2
            30: 512   # conv5_2
        }
        
        # 为每个VGG层创建判别头
        self.spatial_heads = nn.ModuleDict()
        self.freq_heads = nn.ModuleDict()
        
        for layer_idx in vgg_layers:
            layer_name = f'layer_{layer_idx}'
            feat_channels = vgg_channels[layer_idx]
            
            # 空间域判别头
            self.spatial_heads[layer_name] = SpatialDiscriminatorHead(
                feat_channels, nf
            )
            
            # 频域判别头 (可选)
            if use_freq_discriminator:
                self.freq_heads[layer_name] = FrequencyDiscriminatorHead(
                    feat_channels, nf
                )
        
        # 多尺度下采样
        self.downsample = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 4, H, W]
        
        Returns:
            results: 字典，包含不同尺度和不同层的判别结果
            {
                'scale_0': {
                    'spatial_layer_2': [B, 1, H1, W1],
                    'freq_layer_2': [B, 1, H1, W1],
                    ...
                },
                'scale_1': {...},
                ...
            }
        """
        results = {}
        
        current_x = x
        for scale_idx in range(self.num_scales):
            scale_name = f'scale_{scale_idx}'
            results[scale_name] = {}
            
            # 提取VGG特征
            vgg_features = self.vgg_extractor(current_x)
            
            # 对每个VGG层进行判别
            for feat_idx, feat in enumerate(vgg_features):
                layer_name = f'layer_{self.vgg_extractor.feature_layers[feat_idx]}'
                
                # 空间域判别
                spatial_out = self.spatial_heads[layer_name](feat)
                results[scale_name][f'spatial_{layer_name}'] = spatial_out
                
                # 频域判别
                if self.use_freq_discriminator:
                    freq_out = self.freq_heads[layer_name](feat)
                    results[scale_name][f'freq_{layer_name}'] = freq_out
            
            # 下采样到下一个尺度
            if scale_idx < self.num_scales - 1:
                current_x = self.downsample(current_x)
        
        return results


class GANLoss(nn.Module):
    """GAN损失，支持多种GAN类型"""
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        
        if gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'wgan':
            self.loss = None
        elif gan_type == 'hinge':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN type {gan_type} not implemented')
    
    def get_target_tensor(self, input, target_is_real):
        """生成目标标签张量"""
        if target_is_real:
            target_val = self.real_label_val
        else:
            target_val = self.fake_label_val
        
        return torch.full_like(input, target_val)
    
    def forward(self, input, target_is_real):
        """
        计算GAN损失
        
        Args:
            input: 判别器输出
            target_is_real: 是否为真实样本
        
        Returns:
            loss: GAN损失
        """
        if self.gan_type == 'wgan':
            if target_is_real:
                loss = -input.mean()
            else:
                loss = input.mean()
        elif self.gan_type == 'hinge':
            if target_is_real:
                loss = F.relu(1.0 - input).mean()
            else:
                loss = F.relu(1.0 + input).mean()
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = self.loss(input, target_tensor)
        
        return loss


def gradient_penalty(discriminator, real_data, fake_data, device='cuda'):
    """
    计算梯度惩罚 (用于WGAN-GP)
    
    Args:
        discriminator: 判别器模型
        real_data: 真实数据
        fake_data: 生成数据
        device: 设备
    
    Returns:
        gp: 梯度惩罚
    """
    batch_size = real_data.size(0)
    
    # 随机插值
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    # 判别器输出
    disc_interpolates = discriminator(interpolates)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 梯度惩罚
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


# 简化的判别器 (如果计算资源有限)
class SimpleMultiScaleDiscriminator(nn.Module):
    """
    简化版多尺度判别器 (不使用VGG)
    适用于快速实验和资源受限的情况
    """
    def __init__(self, in_channels=4, nf=64, num_scales=3):
        super(SimpleMultiScaleDiscriminator, self).__init__()
        
        self.num_scales = num_scales
        
        # 为每个尺度创建判别器
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(
                self._make_discriminator(in_channels, nf)
            )
        
        self.downsample = nn.AvgPool2d(2, 2)
    
    def _make_discriminator(self, in_channels, nf):
        """创建单个判别器网络"""
        return nn.Sequential(
            # 64x64
            nn.Conv2d(in_channels, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8
            nn.Conv2d(nf * 4, nf * 8, 4, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出
            nn.Conv2d(nf * 8, 1, 4, 1, 1)
        )
    
    def forward(self, x):
        """前向传播"""
        results = {}
        
        current_x = x
        for scale_idx, disc in enumerate(self.discriminators):
            scale_name = f'scale_{scale_idx}'
            results[scale_name] = disc(current_x)
            
            # 下采样
            if scale_idx < self.num_scales - 1:
                current_x = self.downsample(current_x)
        
        return results


if __name__ == '__main__':
    # 测试代码
    print("Testing MultiScaleVGGFreqDiscriminator...")
    
    # 创建判别器
    discriminator = MultiScaleVGGFreqDiscriminator(
        in_channels=4,
        nf=64,
        num_scales=3,
        vgg_layers=[2, 7, 12],
        use_freq_discriminator=True
    )
    
    # 测试输入
    x = torch.randn(2, 4, 256, 256)
    
    # 前向传播
    results = discriminator(x)
    
    print("\nDiscriminator output structure:")
    for scale_name, scale_results in results.items():
        print(f"\n{scale_name}:")
        for output_name, output_tensor in scale_results.items():
            print(f"  {output_name}: {output_tensor.shape}")
    
    print("\n✓ Discriminator test passed!")
    
    # 测试简化版判别器
    print("\nTesting SimpleMultiScaleDiscriminator...")
    simple_disc = SimpleMultiScaleDiscriminator(
        in_channels=4,
        nf=64,
        num_scales=3
    )
    
    simple_results = simple_disc(x)
    print("\nSimple Discriminator output structure:")
    for scale_name, output_tensor in simple_results.items():
        print(f"  {scale_name}: {output_tensor.shape}")
    
    print("\n✓ Simple Discriminator test passed!")