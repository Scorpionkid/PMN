import torch
import torch.nn as nn
import torch.nn.functional as F
import noise_map_processor as nmp

from .dual_path_components import (
    DynamicFusion, AGF,
    WaveletUpsample, DiscreteWaveletUpsample,
    SharpnessRecovery,
    RAWTextureDetector,
    EnhancedDenoisePath,
    EnhancedDetailPath
)


import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== NAFNet核心组件 ==========
class SimpleGate(nn.Module):
    """NAFNet的核心：移除非线性激活函数，使用简单门控"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    """NAFNet的简化通道注意力"""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        return x * self.weight(F.adaptive_avg_pool2d(x, 1))

# ========== 轻量化路径组件 ==========
# ========== 轻量化路径组件 ==========
class SobelFilter(nn.Module):
    """Sobel edge detection filter - 保留原始实现"""
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

class OptimizedDilatedConvChain(nn.Module):
    """优化的空洞卷积链 - 保留核心功能，优化激活函数"""
    def __init__(self, channels, dilated_rates=None):
        super(OptimizedDilatedConvChain, self).__init__()
        if dilated_rates is None:
            dilated_rates = [1, 2, 4, 8]

        # 保留空洞卷积结构，但优化激活函数
        self.dilated_convs = nn.ModuleList()
        for r in dilated_rates:
            conv_block = nn.Sequential(
                nn.Conv2d(channels, channels*2, 3, padding=r, dilation=r),  # 为SimpleGate准备2倍通道
                SimpleGate(),  # 用NAFNet的SimpleGate替代LeakyReLU
                nn.Conv2d(channels, channels, 1)  # 恢复原始通道数
            )
            self.dilated_convs.append(conv_block)

    def forward(self, x):
        res = x
        for conv_block in self.dilated_convs:
            res = conv_block(res) + res  # 保留残差连接
        return res

class LightweightDetailPath(nn.Module):
    """保留核心细节组件的轻量化细节路径"""
    def __init__(self, channels, num_heads=1, dilated_rates=None):
        super().__init__()
        
        # 保留原有的优化空洞卷积链 - 细节保留的核心
        self.dilated_convs = OptimizedDilatedConvChain(channels, dilated_rates)
        
        # 保留原有的Sobel边缘检测器 - 细节检测的核心
        self.edge_detector = SobelFilter()
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1)
        
        # NAFNet风格的轻量化全局上下文处理（替代复杂的TE-MDTA）
        self.global_context = nn.Sequential(
            nn.Conv2d(channels, channels*2, 1),  # 为SimpleGate准备
            SimpleGate(),
            SimplifiedChannelAttention(channels)  # 简化的通道注意力
        )
        
        # 简化的注意力生成机制（保留原有逻辑但减少参数）
        self.attention_gen = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, 1),  # global_context + edge_feat
            SimpleGate(),  # NAFNet的核心替代Sigmoid
            nn.Conv2d(channels, channels, 1)
        )
        
    def forward(self, x, noise_map=None, texture_mask=None):
        # 保留：空洞卷积链处理 - 增加感受野，保留细节
        res = self.dilated_convs(x)
        
        # NAFNet优化：轻量化全局上下文建模
        global_context = self.global_context(res)
        
        # 保留：Sobel边缘检测 - 细节检测的核心
        edge_map = self.edge_detector(x)
        edge_feat = self.edge_conv(edge_map)
        
        # NAFNet优化：简化的注意力生成
        attn_input = torch.cat([global_context, edge_feat], dim=1)
        attention = torch.sigmoid(self.attention_gen(attn_input))
        
        # 保留原有的噪声自适应调整逻辑
        if noise_map is not None:
            noise_factor = torch.sigmoid(4.0 * noise_map - 2.0)
            # 高噪声区域降低纹理敏感度
            attention = attention * (1.0 - noise_factor)
        
        # 保留原有的输出逻辑
        output = x * attention + x
        
        return output

class LightweightDenoisePath(nn.Module):
    """基于NAFNet思想的轻量化降噪路径"""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        
        # 简化的上下文处理（移除复杂的TE-MDTA）
        self.context_proc = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels//2),
            nn.Conv2d(channels, channels*2, 1),  # 为SimpleGate准备
            SimpleGate(),
            nn.Conv2d(channels, channels, 1)
        )
        
        # 简化的内容分支
        self.content_branch = nn.Conv2d(channels, channels, 1)
        
        # NAFNet风格的门控生成
        self.gate_gen = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, 1),
            SimpleGate(),
            nn.Conv2d(channels, 1, 1)
        )
        
        # 简化通道注意力
        self.sca = SimplifiedChannelAttention(channels)
        
    def forward(self, x, noise_map=None, texture_mask=None):
        # 简化的上下文处理
        context = self.context_proc(x)
        
        # 内容处理
        content = self.content_branch(x)
        
        # NAFNet风格的门控生成
        gate_input = torch.cat([context, content], dim=1)
        gate = torch.sigmoid(self.gate_gen(gate_input))
        
        # 应用简化通道注意力
        result = self.sca(content * gate)
        
        # 噪声调制（简化版本）
        if noise_map is not None:
            noise_weight = torch.sigmoid(3.0 * noise_map - 1.5)
            result = result * (0.3 + 0.7 * noise_weight)
        
        return x + result

class LightweightAGF(nn.Module):
    """基于NAFNet思想的轻量化AGF模块"""
    def __init__(self, channels, use_noise_map=False, use_texture_mask=False):
        super().__init__()
        self.use_noise_map = use_noise_map
        self.use_texture_mask = use_texture_mask
        
        # 简化的融合权重生成
        self.fusion_gen = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, 1),  # detail + denoise + features
            SimpleGate(),  # NAFNet的核心组件
            nn.Conv2d(channels, 1, 1)
        )
        
        # 简化的通道注意力用于特征增强
        self.sca = SimplifiedChannelAttention(channels)
        
        # 可学习的噪声缩放参数
        if use_noise_map:
            self.noise_scale = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, detail_path, denoise_path, features, noise_map=None, texture_mask=None):
        # 计算路径差异（保持原有逻辑）
        path_diff = torch.abs(detail_path - denoise_path)
        
        # 应用简化通道注意力增强特征
        enhanced_features = self.sca(features)
        
        # NAFNet风格的融合权重生成
        fusion_input = torch.cat([enhanced_features, path_diff], dim=1)
        alpha = torch.sigmoid(self.fusion_gen(fusion_input))
        
        # 噪声调制（简化但保持效果）
        if self.use_noise_map and noise_map is not None:
            scale = torch.clamp(self.noise_scale, 1.0, 5.0)
            noise_weight = torch.exp(-scale * noise_map)
            alpha = alpha * noise_weight
        
        return alpha * detail_path + (1.0 - alpha) * denoise_path

class LightweightDualPathBlock(nn.Module):
    """完整的轻量化双路径块"""
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        
        # 深度分离卷积特征提取（大幅减少参数）
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels*2, 1),  # 为SimpleGate准备
            SimpleGate(),  # NAFNet核心
            nn.Conv2d(out_channels, out_channels, 1)
        )
        
        # 轻量化双路径
        self.detail_path = LightweightDetailPath(out_channels, heads)
        self.denoise_path = LightweightDenoisePath(out_channels, heads)
        self.fusion = LightweightAGF(out_channels, use_noise_map=True, use_texture_mask=True)
        
    def forward(self, x, noise_map=None, texture_mask=None):
        # NAFNet风格的特征提取
        feat = self.feature_extract(x)
        
        # 双路径处理
        detail = self.detail_path(feat, noise_map, texture_mask)
        denoise = self.denoise_path(feat, noise_map, texture_mask)
        
        # 轻量化融合
        output = self.fusion(detail, denoise, feat, noise_map, texture_mask)
        
        return output, detail, denoise

class LightweightDualPathUNet(nn.Module):
    """完整的轻量化DualPath U-Net"""
    def __init__(self, args=None, **kwargs):
        super().__init__()
        
        # 减少基础通道数
        base_channels = args.get('nf', 32)  # 从48减少到32
        in_channels = args.get('in_channels', 4)
        out_channels = args.get('out_channels', 4)
        heads = args.get('heads', [1, 1, 2, 4])  # 减少注意力头数
        
        # 简化的编码器
        self.enc1 = LightweightDualPathBlock(in_channels, base_channels, heads[0])
        self.enc2 = LightweightDualPathBlock(base_channels, base_channels*2, heads[1])
        self.enc3 = LightweightDualPathBlock(base_channels*2, base_channels*4, heads[2])
        
        # 轻量化瓶颈层
        self.bottleneck = LightweightDualPathBlock(base_channels*4, base_channels*8, heads[3])
        
        # 简化的解码器
        self.dec3 = LightweightDualPathBlock(base_channels*8 + base_channels*4, base_channels*4, heads[2])
        self.dec2 = LightweightDualPathBlock(base_channels*4 + base_channels*2, base_channels*2, heads[1])
        self.dec1 = LightweightDualPathBlock(base_channels*2 + base_channels, base_channels, heads[0])
        
        # 下采样和上采样
        self.down = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        
        # 输出层
        self.final = nn.Conv2d(base_channels, out_channels, 1)
        
    def forward(self, x, noise_map=None, texture_mask=None):
        # 编码器
        enc1_out, _, _ = self.enc1(x, noise_map, texture_mask)
        enc1_down = self.down(enc1_out)
        
        enc2_out, _, _ = self.enc2(enc1_down, noise_map, texture_mask)
        enc2_down = self.down(enc2_out)
        
        enc3_out, _, _ = self.enc3(enc2_down, noise_map, texture_mask)
        enc3_down = self.down(enc3_out)
        
        # 瓶颈层
        bottleneck_out, _, _ = self.bottleneck(enc3_down, noise_map, texture_mask)
        
        # 解码器
        bottleneck_up = self.up3(bottleneck_out)
        dec3_input = torch.cat([bottleneck_up, enc3_out], dim=1)
        dec3_out, _, _ = self.dec3(dec3_input, noise_map, texture_mask)
        
        dec3_up = self.up2(dec3_out)
        dec2_input = torch.cat([dec3_up, enc2_out], dim=1)
        dec2_out, _, _ = self.dec2(dec2_input, noise_map, texture_mask)
        
        dec2_up = self.up1(dec2_out)
        dec1_input = torch.cat([dec2_up, enc1_out], dim=1)
        dec1_out, detail_out, denoise_out = self.dec1(dec1_input, noise_map, texture_mask)
        
        # 最终输出
        main_output = self.final(dec1_out)
        
        return main_output, texture_mask, detail_out, denoise_out

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 使用示例
if __name__ == "__main__":
    # 配置
    args = {
        'nf': 48,  # 基础通道数从48减少到32
        'in_channels': 4,
        'out_channels': 4,
        'heads': [1, 1, 2, 4],  # 减少注意力头数
        'use_wavelet_upsample': False,
        'use_sharpness_recovery': False,
        'use_noise_map': True,
        'use_texture_detection': True,
        'enable_intermediate_supervision': True,
        'use_depthwise_separable': True
    }
    
    # 创建轻量化模型
    model = LightweightDualPathUNet(args)
    
    # 计算参数量
    param_count = count_parameters(model)
    print(f"轻量化模型参数量: {param_count/1e6:.1f}M")
    
    # 测试前向传播
    x = torch.randn(1, 4, 256, 256)
    noise_map = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        output, texture_mask, detail_out, denoise_out = model(x, noise_map)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"细节路径输出: {detail_out.shape if detail_out is not None else 'None'}")
        print(f"降噪路径输出: {denoise_out.shape if denoise_out is not None else 'None'}")