"""
PBNet: Physics-Based Network - 全BayerConvBlock版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import *


class LightweightPhysicsBlock(nn.Module):
    """物理约束块 - 修正版本"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # reduction ratio解释：
        # 输入channels → channels//reduction → channels
        # 例如：256 → 32 → 256，大幅减少参数量
        mid_channels = max(channels // reduction, 8)
        
        self.noise_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),  # 压缩阶段
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1),  # 恢复阶段
            nn.Sigmoid()
        )
        
        self.residual_conv = nn.Conv2d(channels, channels, 1)

    def _downsample_noise_map(self, noise_map, target_size):
        current_h, current_w = noise_map.shape[-2:]
        target_h, target_w = target_size
        
        # 计算需要的池化倍数
        scale_h = current_h // target_h
        scale_w = current_w // target_w
        
        if scale_h > 1 or scale_w > 1:
            # 使用平均池化缩小
            kernel_size = max(scale_h, scale_w)
            noise_map = F.avg_pool2d(noise_map, kernel_size=kernel_size, stride=kernel_size)
            
            # 如果还不匹配，进行自适应池化
            if noise_map.shape[-2:] != target_size:
                noise_map = F.adaptive_avg_pool2d(noise_map, target_size)
        
        return noise_map
        
    def forward(self, x, noise_map=None):
        # 通过reduction实现轻量化的通道注意力
        weight = self.noise_modulator(x)  # [B, C, 1, 1]
        residual = torch.tanh(self.residual_conv(x * weight))
        
        if noise_map is not None:
            noise_map = self._downsample_noise_map(noise_map, x.shape[-2:])
            noise_strength = torch.sqrt(noise_map.clamp(min=1e-6))
            residual = residual * noise_strength
        
        return x + residual * 0.1


class BayerConvBlock(nn.Module):
    """Bayer卷积块 - 使用InstanceNorm"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_lightweight=False):
        super().__init__()
        
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        
        self.bayer_conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size, stride=stride, 
            padding=kernel_size//2, 
            groups=4  # Bayer通道分组
        )
        
        if use_lightweight:
            self.fusion = nn.Conv2d(out_channels, out_channels, 1, groups=4)
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, groups=4),
                nn.Conv2d(out_channels, out_channels, 1)
            )
        
        # 修正：使用InstanceNorm而不是BatchNorm
        # 原因：去噪任务中每个样本的噪声特性可能差异很大
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        out = self.bayer_conv(x)
        out = self.fusion(out)
        out = self.norm(out)  # 每个样本独立归一化
        out = self.relu(out)
        return out


class PBNet_AllBayer(nn.Module):
    """全BayerConvBlock版本的PBNet"""
    def __init__(self, args=None):
        super().__init__()
        
        self.args = args or {
            'nframes': 1,
            'in_nc': 4,
            'out_nc': 4,
            'nf': 32,
            'res': True,
            'use_physics': True,
            'camera_type': 'SonyA7S2'
        }
        
        nframes = self.args['nframes']
        nf = self.args['nf']
        in_nc = self.args['in_nc']
        out_nc = self.args['out_nc']
        
        self.use_physics = self.args.get('use_physics', True)
        
        # 编码器 - 全部使用BayerConvBlock
        # 前几层使用标准版，深层使用轻量版
        self.conv1_1 = BayerConvBlock(in_nc * nframes, nf, use_lightweight=False)
        self.conv1_2 = BayerConvBlock(nf, nf, use_lightweight=False)
        
        self.conv2_1 = BayerConvBlock(nf, nf*2, use_lightweight=False)
        self.conv2_2 = BayerConvBlock(nf*2, nf*2, use_lightweight=False)
        
        self.conv3_1 = BayerConvBlock(nf*2, nf*4, use_lightweight=True)
        self.conv3_2 = BayerConvBlock(nf*4, nf*4, use_lightweight=True)
        
        self.conv4_1 = BayerConvBlock(nf*4, nf*8, use_lightweight=True)
        self.conv4_2 = BayerConvBlock(nf*8, nf*8, use_lightweight=True)
        
        self.conv5_1 = BayerConvBlock(nf*8, nf*16, use_lightweight=True)
        self.conv5_2 = BayerConvBlock(nf*16, nf*16, use_lightweight=True)
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        
        # 物理约束模块
        if self.use_physics:
            self.physics1 = LightweightPhysicsBlock(nf)
            self.physics3 = LightweightPhysicsBlock(nf*4)
            self.physics5 = LightweightPhysicsBlock(nf*16)
        
        # 解码器 - 也全部使用BayerConvBlock
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6_1 = BayerConvBlock(nf*16, nf*8, use_lightweight=True)
        self.conv6_2 = BayerConvBlock(nf*8, nf*8, use_lightweight=True)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7_1 = BayerConvBlock(nf*8, nf*4, use_lightweight=True)
        self.conv7_2 = BayerConvBlock(nf*4, nf*4, use_lightweight=True)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8_1 = BayerConvBlock(nf*4, nf*2, use_lightweight=False)
        self.conv8_2 = BayerConvBlock(nf*2, nf*2, use_lightweight=False)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9_1 = BayerConvBlock(nf*2, nf, use_lightweight=False)
        self.conv9_2 = BayerConvBlock(nf, nf, use_lightweight=False)
        
        # 最后的输出层使用标准卷积
        self.conv10_1 = nn.Conv2d(nf, out_nc, 1)
        
        self.res = self.args.get('res', True)
        
    def forward(self, x, noise_map=None):
        """
        x: 输入RGGB图像 [B, 4, H, W] - 通道顺序为[R,G,B,G] 
        noise_map: 噪声图 [B, 1, H, W]
        """
        # 编码路径
        conv1 = self.conv1_1(x)
        conv1 = self.conv1_2(conv1)
        
        if self.use_physics and noise_map is not None:
            conv1 = self.physics1(conv1, noise_map)
        
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        
        if self.use_physics and noise_map is not None:
            conv3 = self.physics3(conv3, noise_map)
        
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv5)
        
        if self.use_physics and noise_map is not None:
            conv5 = self.physics5(conv5, noise_map)
        
        # 解码路径
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6_1(up6)
        conv6 = self.conv6_2(conv6)
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7_1(up7)
        conv7 = self.conv7_2(conv7)
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8_1(up8)
        conv8 = self.conv8_2(conv8)
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9_1(up9)
        conv9 = self.conv9_2(conv9)
        
        conv10 = self.conv10_1(conv9)
        
        if self.res:
            out = conv10 + x
        else:
            out = conv10
            
        return out


def test_all_bayer_params():
    """测试全BayerConvBlock版本的参数量"""
    
    configs = [
        {'nf': 28, 'name': '全Bayer-28通道'},
        {'nf': 32, 'name': '全Bayer-32通道'},
        {'nf': 36, 'name': '全Bayer-36通道'},
        {'nf': 40, 'name': '全Bayer-40通道'},
    ]
    
    print("参数量对比:")
    print("-" * 50)
    
    for config in configs:
        args = {
            'nframes': 1,
            'in_nc': 4,
            'out_nc': 4,
            'nf': config['nf'],
            'res': True,
            'use_physics': True,
            'camera_type': 'SonyA7S2'
        }
        
        model = PBNet_AllBayer(args)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"{config['name']}: {total_params/1e6:.2f}M 参数")
        
        # 测试前向传播
        x = torch.randn(1, 4, 128, 128)
        noise_map = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            output = model(x, noise_map)
            print(f"  输入: {x.shape} → 输出: {output.shape}")
    
    print("-" * 50)
    print("建议: 使用32通道版本，参数量约10-11M，在目标范围内")



class PBNetLoss(nn.Module):
    """PBNet的损失函数，集成物理约束"""
    
    def __init__(self, camera_type='SonyA7S2', lambda_physics=0.1):
        super().__init__()
        from losses import Unet_Loss
        self.base_loss = Unet_Loss()
        self.lambda_physics = lambda_physics
        self.camera_type = camera_type
        self.l2_loss = nn.MSELoss()
        
    def compute_physics_loss(self, output, noisy, noise_params=None, iso=None):
        """计算物理约束损失"""
        # 计算残差
        residual = noisy - output
        
        # 获取噪声参数
        K = 0.1  # 默认值
        sigma_read = 0.01
        
        if noise_params is not None:
            K = noise_params.get('K', noise_params.get('Kmax', K))
            sigma_read = noise_params.get('sigGs', sigma_read)
        elif iso is not None:
            # 从ISO估算参数
            try:
                from data_process.process import get_camera_noisy_params_max
                params = get_camera_noisy_params_max(f"{self.camera_type}_{int(iso)}")
                if params:
                    K = params.get('Kmax', K)
                    sigma_read = params.get('sigGs', sigma_read)
            except:
                pass
        
        # 转换为张量
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, device=output.device, dtype=output.dtype)
        if not isinstance(sigma_read, torch.Tensor):
            sigma_read = torch.tensor(sigma_read, device=output.device, dtype=output.dtype)
        
        # Poisson-Gaussian模型：Var = K * I + sigma_read^2
        expected_var = K * output.clamp(min=0) + sigma_read ** 2
        
        # 计算局部方差
        kernel_size = 5
        pad = kernel_size // 2
        residual_sq = residual ** 2
        actual_var = F.avg_pool2d(residual_sq, kernel_size, stride=1, padding=pad)
        
        # 物理一致性损失
        physics_loss = self.l2_loss(actual_var, expected_var)
        
        return physics_loss
    
    def forward(self, output, target, noisy=None, noise_params=None, iso=None):
        """计算总损失"""
        # 基础L1损失
        l1_loss = self.base_loss(output, target)
        
        # 物理约束损失
        if self.lambda_physics > 0 and noisy is not None:
            physics_loss = self.compute_physics_loss(output, noisy, noise_params, iso)
            total_loss = l1_loss + self.lambda_physics * physics_loss
            return total_loss, {'l1': l1_loss.item(), 'physics': physics_loss.item()}
        else:
            return l1_loss, {'l1': l1_loss.item(), 'physics': 0}
        
# 对比不同reduction ratio的参数量影响
def compare_reduction_ratios():
    """对比不同reduction ratio的参数量"""
    channels = 128
    
    print("Reduction Ratio对参数量的影响:")
    print("-" * 40)
    
    for ratio in [4, 8, 16]:
        mid_channels = max(channels // ratio, 8)
        
        # 计算参数量
        compress_params = channels * mid_channels
        expand_params = mid_channels * channels
        total_params = compress_params + expand_params
        
        print(f"Ratio {ratio}: {channels}→{mid_channels}→{channels}")
        print(f"  参数量: {total_params:,} ({total_params/1000:.1f}K)")
        print(f"  相比直接连接减少: {(1 - total_params/(channels*channels))*100:.1f}%")
        print()


# 对比BatchNorm vs InstanceNorm
def compare_normalization():
    """展示BatchNorm和InstanceNorm的差异"""
    batch_size = 4
    channels = 64
    height, width = 32, 32
    
    # 模拟不同噪声程度的输入
    clean = torch.randn(1, channels, height, width) * 0.1
    noisy = torch.randn(1, channels, height, width) * 0.5  
    very_noisy = torch.randn(1, channels, height, width) * 1.0
    extreme_noisy = torch.randn(1, channels, height, width) * 2.0
    
    batch_input = torch.cat([clean, noisy, very_noisy, extreme_noisy], dim=0)
    
    # BatchNorm vs InstanceNorm
    bn = nn.BatchNorm2d(channels)
    in_norm = nn.InstanceNorm2d(channels, affine=True)
    
    with torch.no_grad():
        bn_output = bn(batch_input)
        in_output = in_norm(batch_input)
        
        print("归一化方式对比:")
        print("-" * 30)
        print("输入统计 (每个样本的std):")
        for i in range(batch_size):
            print(f"  样本{i}: {torch.std(batch_input[i]):.3f}")
        
        print("\nBatchNorm输出 (强制batch内统一):")
        for i in range(batch_size):
            print(f"  样本{i}: {torch.std(bn_output[i]):.3f}")
            
        print("\nInstanceNorm输出 (保持样本独立性):")
        for i in range(batch_size):
            print(f"  样本{i}: {torch.std(in_output[i]):.3f}")

# 兼容性函数
def UNetSeeInDark_Physics(args=None):
    return PBNet_AllBayer(args)


if __name__ == '__main__':
    test_all_bayer_params()
    compare_reduction_ratios()
    print("="*50)
    compare_normalization()