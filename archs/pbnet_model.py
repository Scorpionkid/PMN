"""
PBNet: Physics-Based Network for Raw Image Denoising
基于PMN框架的物理约束网络实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import *
from typing import Optional, Dict, Any

# 从PMN框架导入噪声处理函数
try:
    from data_process.process import get_camera_noisy_params_max, sample_params_max
    from data_process.noise_map import generate_noise_map
except:
    print("Warning: PMN noise processing modules not found")


class PhysicsConstraintBlock(nn.Module):
    """物理约束块 - 使用PMN的噪声参数"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        
        # 噪声参数调制网络
        self.noise_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels + 1, channels // reduction, 1),  # +1 for noise map
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 残差生成
        self.residual_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # 深度卷积
            nn.Conv2d(channels, channels, 1),  # 逐点卷积
            nn.Tanh()
        )
        
    def forward(self, x, noise_map=None):
        """
        x: 特征图 [B, C, H, W]
        noise_map: 噪声图 [B, 1, H, W] (从PMN的generate_noise_map获得)
        """
        if noise_map is None:
            return x
        
        # 调整噪声图大小以匹配特征图
        if noise_map.shape[-2:] != x.shape[-2:]:
            noise_map = F.interpolate(noise_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # 拼接特征和噪声图
        combined = torch.cat([x, noise_map], dim=1)
        
        # 生成调制权重
        weight = self.noise_modulator(combined)
        
        # 生成物理约束的残差
        residual = self.residual_conv(x * weight)
        
        # 使用噪声图调制残差强度
        noise_strength = torch.sqrt(noise_map.clamp(min=1e-6))
        residual = residual * noise_strength
        
        return x + residual * 0.1  # 小幅残差


class BayerConvBlock(nn.Module):
    """Bayer-aware卷积块 - 针对RGGB模式"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # RGGB通道分组处理
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        group_in = in_channels // 4
        group_out = out_channels // 4
        
        # 对每个Bayer通道独立处理
        self.conv_r = nn.Conv2d(group_in, group_out, 3, stride=stride, padding=1)
        self.conv_g1 = nn.Conv2d(group_in, group_out, 3, stride=stride, padding=1)
        self.conv_g2 = nn.Conv2d(group_in, group_out, 3, stride=stride, padding=1)
        self.conv_b = nn.Conv2d(group_in, group_out, 3, stride=stride, padding=1)
        
        # 跨通道融合
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # 分离RGGB通道
        B, C, H, W = x.shape
        x_reshape = x.view(B, 4, C//4, H, W)
        
        # 独立处理每个通道
        r = self.conv_r(x_reshape[:, 0])
        g1 = self.conv_g1(x_reshape[:, 1])
        g2 = self.conv_g2(x_reshape[:, 2])
        b = self.conv_b(x_reshape[:, 3])
        
        # 合并
        out = torch.cat([r, g1, g2, b], dim=1)
        
        # 跨通道融合
        out = self.fusion(out)
        out = self.norm(out)
        out = self.relu(out)
        
        return out


class PBNet(nn.Module):
    """Physics-Based Network - 基于PMN框架的物理约束网络"""
    def __init__(self, args=None):
        super().__init__()
        
        # 默认参数
        self.args = args or {
            'nframes': 1,
            'in_nc': 4,      # RGGB
            'out_nc': 4,     # 保持RGGB输出
            'nf': 32,
            'res': True,
            'use_physics': True,
            'use_bayer_conv': True,
            'camera_type': 'SonyA7S2'
        }
        
        nframes = self.args['nframes']
        nf = self.args['nf']
        in_nc = self.args['in_nc']
        out_nc = self.args['out_nc']
        
        self.use_physics = self.args.get('use_physics', True)
        self.use_bayer_conv = self.args.get('use_bayer_conv', True)
        self.camera_type = self.args.get('camera_type', 'SonyA7S2')
        
        # 使用Bayer-aware卷积或标准卷积
        ConvBlock = BayerConvBlock if self.use_bayer_conv else nn.Conv2d
        
        # 编码器
        if self.use_bayer_conv:
            self.conv1_1 = BayerConvBlock(in_nc * nframes, nf)
            self.conv1_2 = BayerConvBlock(nf, nf)
        else:
            self.conv1_1 = nn.Conv2d(in_nc * nframes, nf, 3, padding=1)
            self.conv1_2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2_1 = nn.Conv2d(nf, nf*2, 3, padding=1)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3_1 = nn.Conv2d(nf*2, nf*4, 3, padding=1)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4_1 = nn.Conv2d(nf*4, nf*8, 3, padding=1)
        self.conv4_2 = nn.Conv2d(nf*8, nf*8, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5_1 = nn.Conv2d(nf*8, nf*16, 3, padding=1)
        self.conv5_2 = nn.Conv2d(nf*16, nf*16, 3, padding=1)
        
        # 物理约束模块（在不同尺度应用）
        if self.use_physics:
            self.physics1 = PhysicsConstraintBlock(nf)
            self.physics2 = PhysicsConstraintBlock(nf*4)
            self.physics3 = PhysicsConstraintBlock(nf*16)
        
        # 解码器
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf*16, nf*8, 3, padding=1)
        self.conv6_2 = nn.Conv2d(nf*8, nf*8, 3, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf*8, nf*4, 3, padding=1)
        self.conv7_2 = nn.Conv2d(nf*4, nf*4, 3, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(nf*4, nf*2, 3, padding=1)
        self.conv8_2 = nn.Conv2d(nf*2, nf*2, 3, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9_1 = nn.Conv2d(nf*2, nf, 3, padding=1)
        self.conv9_2 = nn.Conv2d(nf, nf, 3, padding=1)
        
        self.conv10_1 = nn.Conv2d(nf, out_nc, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        # 残差连接
        self.res = self.args.get('res', True)
        
    def forward(self, x, noise_params=None, iso=None):
        """
        x: 输入RGGB图像 [B, 4, H, W]
        noise_params: 噪声参数字典 (从PMN获取)
        iso: ISO值 [B] (用于生成噪声图)
        """
        # 生成噪声图（使用PMN的噪声模型）
        noise_map = None
        if self.use_physics and (noise_params is not None or iso is not None):
            try:
                # 使用PMN的噪声图生成
                noise_map = generate_noise_map(
                    x, 
                    noise_params=noise_params,
                    iso=iso,
                    camera_name=self.camera_type
                )
                
                # 转换为合适的尺寸
                if isinstance(noise_map, np.ndarray):
                    noise_map = torch.from_numpy(noise_map).float()
                if noise_map.dim() == 3:
                    noise_map = noise_map.unsqueeze(1)  # 添加通道维度
                    
                # 对RGGB取平均得到单通道噪声图
                if noise_map.shape[1] == 4:
                    noise_map = noise_map.mean(dim=1, keepdim=True)
                    
            except Exception as e:
                print(f"Warning: Failed to generate noise map: {e}")
                noise_map = None
        
        # 编码路径
        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        
        # 应用物理约束
        if self.use_physics and noise_map is not None:
            conv1 = self.physics1(conv1, noise_map)
        
        pool1 = self.pool1(conv1)
        
        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)
        
        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        
        # 应用物理约束
        if self.use_physics and noise_map is not None:
            # 下采样噪声图
            noise_map_down2 = F.avg_pool2d(noise_map, 4)
            conv3 = self.physics2(conv3, noise_map_down2)
        
        pool3 = self.pool3(conv3)
        
        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)
        
        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))
        
        # 应用物理约束
        if self.use_physics and noise_map is not None:
            # 下采样噪声图
            noise_map_down4 = F.avg_pool2d(noise_map, 16)
            conv5 = self.physics3(conv5, noise_map_down4)
        
        # 解码路径
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))
        
        conv10 = self.conv10_1(conv9)
        
        # 残差连接
        if self.res:
            out = conv10 + x
        else:
            out = conv10
            
        return out
    
    def get_noise_params(self, data):
        """从数据中提取噪声参数（兼容PMN框架）"""
        if 'noise_params' in data:
            return data['noise_params']
        
        # 尝试从ISO获取
        if 'ISO' in data:
            iso = data['ISO']
            if hasattr(iso, 'item'):
                iso = iso.item()
            
            # 使用PMN的噪声参数获取函数
            try:
                params = sample_params_max(
                    camera_type=self.camera_type,
                    iso=iso
                )
                return params
            except:
                return None
        
        return None


class PBNet_DSC(PBNet):
    """PBNet with Dark Shading Correction - 集成暗影校正"""
    def __init__(self, args=None):
        super().__init__(args)
        # 此版本假设暗影校正已在预处理中完成
        self.name = 'PBNet_DSC'


# 兼容性函数
def UNetSeeInDark_Physics(args=None):
    """创建PBNet，兼容原始UNet接口"""
    return PBNet(args)


# 测试函数
def test_pbnet_pmn():
    """测试PBNet与PMN框架的集成"""
    import torch
    
    # 创建模型
    args = {
        'nframes': 1,
        'in_nc': 4,
        'out_nc': 4,
        'nf': 32,
        'res': True,
        'use_physics': True,
        'use_bayer_conv': True,
        'camera_type': 'SonyA7S2'
    }
    
    model = PBNet(args)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 4, 128, 128)  # RGGB输入
    iso = torch.tensor([1600, 3200])
    
    # 前向传播
    with torch.no_grad():
        output = model(x, iso=iso)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    # 测试噪声参数传递
    noise_params = {
        'K': 0.5,
        'sigGs': 0.01,
        'sigTL': 0.02,
        'wp': 16383,
        'bl': 512
    }
    
    with torch.no_grad():
        output = model(x, noise_params=noise_params)
        print("Test with noise params passed!")
    
    return model


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


if __name__ == '__main__':
    test_pbnet_pmn()