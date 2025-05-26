"""
PBNet: Physics-Based Network for Raw Image Denoising
Author: [hgh]
Date: 2025
Description: A lightweight physics-constrained network for raw image denoising
             that incorporates Bayer-aware processing and ISO-adaptive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，用于减少参数量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LightweightBayerPreprocess(nn.Module):
    """轻量级Bayer预处理模块"""
    def __init__(self):
        super().__init__()
        # Bayer pattern masks for RGGB
        self.register_buffer('r_mask', torch.tensor([[1,0],[0,0]], dtype=torch.float32))
        self.register_buffer('g_mask', torch.tensor([[0,1],[0,1]], dtype=torch.float32))
        self.register_buffer('b_mask', torch.tensor([[0,0],[1,0]], dtype=torch.float32))
    
    def forward(self, x):
        """
        输入: [B, C, H, W] - 原始Bayer图像
        输出: [B, C*4, H//2, W//2] - 展开的RGGB通道
        """
        B, C, H, W = x.shape
        # 展开2x2块
        x_unfold = F.unfold(x, kernel_size=2, stride=2)  # [B, C*4, H*W/4]
        x_unfold = x_unfold.view(B, C, 4, H//2, W//2)   # [B, C, 4, H//2, W//2]
        
        # 提取RGGB通道
        r = x_unfold[:, :, 0:1, :, :]  # R
        g1 = x_unfold[:, :, 1:2, :, :] # G1
        b = x_unfold[:, :, 2:3, :, :] # G2
        g2 = x_unfold[:, :, 3:4, :, :]  # B
        
        # 合并绿色通道
        g = (g1 + g2) / 2
        
        # 拼接为 [B, C*4, H//2, W//2]
        out = torch.cat([r, g1, b, g2], dim=2)
        out = out.view(B, C*4, H//2, W//2)
        return out


class PhysicsConstraintModule(nn.Module):
    """物理约束模块"""
    def __init__(self, channels, iso_levels=10):
        super().__init__()
        self.channels = channels
        self.iso_levels = iso_levels
        
        # ISO自适应参数
        self.iso_embed = nn.Embedding(iso_levels, channels//8)
        self.iso_proj = nn.Sequential(
            nn.Conv2d(channels//8, channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 1),
            nn.Sigmoid()
        )
        
        # 噪声级别预测
        self.noise_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//16, 1, 1),
            nn.Sigmoid()
        )
        
        # 物理约束的残差生成
        self.residual_gen = nn.Sequential(
            DepthwiseSeparableConv(channels, channels//2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels//2, channels)
        )
        
        # 系统增益参数（可学习）
        self.register_buffer('system_gain', torch.linspace(0.1, 2.0, iso_levels))
        
    def forward(self, x, iso_idx=None):
        """
        x: 输入特征 [B, C, H, W]
        iso_idx: ISO索引 [B]
        """
        if iso_idx is None:
            return x
        
        B, C, H, W = x.shape
        
        # ISO自适应调制
        iso_feat = self.iso_embed(iso_idx)  # [B, C//8]
        iso_feat = iso_feat.view(B, -1, 1, 1)
        iso_weight = self.iso_proj(iso_feat)  # [B, C, 1, 1]
        
        # 噪声级别估计
        noise_level = self.noise_predictor(x)  # [B, 1, 1, 1]
        
        # 获取当前ISO的系统增益
        gain = self.system_gain[iso_idx].view(B, 1, 1, 1)
        
        # 生成物理约束的残差
        residual = self.residual_gen(x * iso_weight)
        
        # 应用物理约束：残差应与噪声级别和系统增益相关
        physics_residual = residual * noise_level * torch.sqrt(gain)
        
        return x + 0.1 * physics_residual  # 小幅度残差修正


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, use_depthwise=False):
        super().__init__()
        if use_depthwise and in_channels == out_channels:
            self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class PBNet(nn.Module):
    """
    Physics-Based Network for Raw Image Denoising
    结合物理约束和Bayer-aware处理的轻量级去噪网络
    """
    def __init__(self, args=None):
        super().__init__()
        
        # 默认参数
        if args is None:
            args = {
                'nframes': 1,
                'in_nc': 4,      # Bayer RGGB
                'out_nc': 3,     # RGB输出
                'nf': 32,        # 基础通道数
                'res': True,     # 是否使用残差连接
                'iso_levels': 10,
                'use_physics': True,
                'use_bayer_preprocess': True
            }
        
        self.args = args
        nframes = args['nframes']
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']
        
        # Bayer预处理（可选）
        self.use_bayer_preprocess = args.get('use_bayer_preprocess', True)
        if self.use_bayer_preprocess:
            self.bayer_preprocess = LightweightBayerPreprocess()
            actual_in_nc = in_nc * 4  # RGGB展开
        else:
            actual_in_nc = in_nc * nframes
        
        # 编码器
        self.conv1 = ConvBlock(actual_in_nc, nf)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = ConvBlock(nf, nf*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(nf*2, nf*4, use_depthwise=True)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = ConvBlock(nf*4, nf*8, use_depthwise=True)
        self.pool4 = nn.MaxPool2d(2)
        
        # 瓶颈层 + 物理约束
        self.conv5 = ConvBlock(nf*8, nf*16, use_depthwise=True)
        
        # 物理约束模块（在多个尺度应用）
        self.use_physics = args.get('use_physics', True)
        if self.use_physics:
            self.physics_module1 = PhysicsConstraintModule(nf*16, args.get('iso_levels', 10))
            self.physics_module2 = PhysicsConstraintModule(nf*8, args.get('iso_levels', 10))
            self.physics_module3 = PhysicsConstraintModule(nf*4, args.get('iso_levels', 10))
        
        # 解码器
        self.up6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6 = ConvBlock(nf*16, nf*8, use_depthwise=True)  # concat后是nf*16
        
        self.up7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7 = ConvBlock(nf*8, nf*4, use_depthwise=True)
        
        self.up8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8 = ConvBlock(nf*4, nf*2)
        
        self.up9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9 = ConvBlock(nf*2, nf)
        
        # 输出层
        self.conv10 = nn.Conv2d(nf, out_nc, 1)
        
        # 残差连接（可选）
        self.res = args.get('res', True)
        if self.res:
            if self.use_bayer_preprocess:
                self.res_conv = nn.Conv2d(in_nc*4, out_nc, 1)
            else:
                self.res_conv = nn.Conv2d(in_nc*nframes, out_nc, 1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, iso_idx=None):
        """
        x: 输入张量 [B, C, H, W]
        iso_idx: ISO索引 [B] (可选)
        """
        # Bayer预处理
        if self.use_bayer_preprocess:
            x = self.bayer_preprocess(x)
        
        # 保存输入用于残差连接
        input_x = x
        
        # 编码路径
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        # 瓶颈层
        conv5 = self.conv5(pool4)
        
        # 应用物理约束
        if self.use_physics and iso_idx is not None:
            conv5 = self.physics_module1(conv5, iso_idx)
        
        # 解码路径
        up6 = self.up6(conv5)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        if self.use_physics and iso_idx is not None:
            conv6 = self.physics_module2(conv6, iso_idx)
        
        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        if self.use_physics and iso_idx is not None:
            conv7 = self.physics_module3(conv7, iso_idx)
        
        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        
        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)
        
        # 输出
        out = self.conv10(conv9)
        
        # 残差连接
        if self.res:
            out = out + self.res_conv(input_x)
        
        return out
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self):
        """获取模型信息"""
        param_count = self.count_parameters()
        info = {
            'name': 'PBNet',
            'parameters': param_count,
            'parameters_str': f'{param_count/1e6:.2f}M',
            'use_physics': self.use_physics,
            'use_bayer_preprocess': self.use_bayer_preprocess,
            'base_channels': self.args['nf']
        }
        return info


class PBNetLoss(nn.Module):
    """PBNet的损失函数，包含物理一致性约束"""
    def __init__(self, args=None):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # 权重
        self.lambda_l1 = args.get('lambda_l1', 1.0)
        self.lambda_physics = args.get('lambda_physics', 0.1)
        self.lambda_perceptual = args.get('lambda_perceptual', 0.0)
        
        # 物理参数
        self.register_buffer('system_gain', torch.linspace(0.1, 2.0, 10))
        self.register_buffer('read_noise_var', torch.tensor(0.01))
    
    def compute_physics_loss(self, output, noisy, iso_idx):
        """计算物理一致性损失"""
        # 获取系统增益
        gain = self.system_gain[iso_idx].view(-1, 1, 1, 1)
        
        # 计算残差
        residual = noisy - output
        
        # 理论噪声方差（简化的Poisson-Gaussian模型）
        expected_var = gain * output.clamp(min=1e-3) + self.read_noise_var
        
        # 实际噪声方差（局部估计）
        kernel_size = 5
        residual_sq = residual ** 2
        actual_var = F.avg_pool2d(residual_sq, kernel_size, stride=1, padding=kernel_size//2)
        
        # 物理一致性损失
        physics_loss = self.l2_loss(actual_var, expected_var)
        
        return physics_loss
    
    def forward(self, output, target, noisy=None, iso_idx=None):
        """
        output: 网络输出
        target: 真实值
        noisy: 噪声输入（用于物理约束）
        iso_idx: ISO索引
        """
        losses = {}
        
        # 主要重建损失
        l1_loss = self.l1_loss(output, target)
        losses['l1'] = l1_loss
        
        # 物理一致性损失
        if noisy is not None and iso_idx is not None and self.lambda_physics > 0:
            physics_loss = self.compute_physics_loss(output, noisy, iso_idx)
            losses['physics'] = physics_loss
        else:
            losses['physics'] = torch.tensor(0.0).to(output.device)
        
        # 总损失
        total_loss = self.lambda_l1 * losses['l1'] + self.lambda_physics * losses['physics']
        losses['total'] = total_loss
        
        return losses


def create_pbnet(args=None):
    """创建PBNet模型"""
    model = PBNet(args)
    return model


def test_pbnet():
    """测试PBNet模型"""
    # 测试参数
    args = {
        'nframes': 1,
        'in_nc': 4,
        'out_nc': 3,
        'nf': 32,
        'iso_levels': 10,
        'use_physics': True,
        'use_bayer_preprocess': True
    }
    
    # 创建模型
    model = create_pbnet(args)
    
    # 打印模型信息
    info = model.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Parameters: {info['parameters_str']}")
    print(f"Use Physics: {info['use_physics']}")
    print(f"Use Bayer Preprocess: {info['use_bayer_preprocess']}")
    
    # 测试前向传播
    batch_size = 2
    height, width = 128, 128
    x = torch.randn(batch_size, 4, height, width)
    iso_idx = torch.tensor([0, 1])  # ISO索引
    
    # 前向传播
    with torch.no_grad():
        output = model(x, iso_idx)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 测试损失函数
    loss_fn = PBNetLoss(args)
    target = torch.randn_like(output)
    noisy = x[:, :3, :, :] if x.shape[1] > 3 else x[:, :1, :, :].repeat(1, 3, 1, 1)
    
    losses = loss_fn(output, target, noisy, iso_idx)
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")


if __name__ == '__main__':
    test_pbnet()
