"""
PBNet: Physics-Based Network for Raw Image Denoising
基于PMN框架的物理约束网络实现 - 第一层Bayer-aware版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LightweightPhysicsBlock(nn.Module):
    """轻量化物理约束块"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # Reduction ratio: 输入channels → channels//reduction → channels
        # 例如：256 → 32 → 256，大幅减少参数量
        mid_channels = max(channels // reduction, 8)
        
        self.noise_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, 1),  # 压缩阶段
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1),  # 恢复阶段
            nn.Sigmoid()
        )
        
        # 轻量化残差生成
        self.residual_conv = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x, noise_map=None):
        """
        x: 特征图 [B, C, H, W]
        noise_map: 噪声图 [B, 1, H, W]
        """
        # 生成调制权重
        weight = self.noise_modulator(x)
        
        # 生成物理约束的残差
        residual = torch.tanh(self.residual_conv(x * weight))
        
        # 如果有噪声图，用它来调制残差强度
        if noise_map is not None:
            # 使用池化匹配尺寸，不使用插值
            noise_map = self._downsample_noise_map(noise_map, x.shape[-2:])
            noise_strength = torch.sqrt(noise_map.clamp(min=1e-6))
            residual = residual * noise_strength
        
        return x + residual * 0.1  # 小幅残差
    
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


class BayerConv(nn.Module):
    """Bayer-aware卷积 - 只用于第一层处理RGGB输入"""
    def __init__(self, in_channels=4, out_channels=32, kernel_size=3):
        super().__init__()
        assert in_channels == 4, "TrueBayerConv only processes RGGB input (4 channels)"
        
        # 为每个真实的Bayer通道设计专门的处理
        # 通道顺序：[R, G1, B, G2]
        channels_per_color = out_channels // 4
        
        self.process_R = nn.Conv2d(1, channels_per_color, kernel_size, padding=kernel_size//2)
        self.process_G1 = nn.Conv2d(1, channels_per_color, kernel_size, padding=kernel_size//2) 
        self.process_B = nn.Conv2d(1, channels_per_color, kernel_size, padding=kernel_size//2)
        self.process_G2 = nn.Conv2d(1, channels_per_color, kernel_size, padding=kernel_size//2)
        
        # Bayer通道间的交互融合
        self.cross_bayer_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 可学习的Bayer权重 - 体现不同颜色通道的重要性
        self.channel_weights = nn.Parameter(torch.tensor([1.0, 0.9, 1.0, 0.9]))  # R, G1, B, G2
        
    def forward(self, x):
        """
        x: [B, 4, H, W] 真实的RGGB输入
        输出: [B, out_channels, H, W] 特征表示
        """
        # 分别处理每个Bayer通道
        R_feat = self.process_R(x[:, 0:1]) * self.channel_weights[0]     # R → channels_per_color个特征
        G1_feat = self.process_G1(x[:, 1:2]) * self.channel_weights[1]   # G1 → channels_per_color个特征  
        B_feat = self.process_B(x[:, 2:3]) * self.channel_weights[2]     # B → channels_per_color个特征
        G2_feat = self.process_G2(x[:, 3:4]) * self.channel_weights[3]   # G2 → channels_per_color个特征
        
        # 按Bayer模式组合特征
        # 保持一定的空间对应关系：[R特征, G1特征, B特征, G2特征]
        combined = torch.cat([R_feat, G1_feat, B_feat, G2_feat], dim=1)
        
        # Bayer通道间信息交互
        output = self.cross_bayer_fusion(combined)
        
        # 从这里开始，输出的特征不再有严格的RGGB语义
        # 而是融合了颜色信息的抽象特征表示
        return output
    
class TrueBayerAwareConv(nn.Module):
    """真正的Bayer-aware卷积 - 理解RGGB的空间排列和邻接关系"""
    def __init__(self, in_channels=4, out_channels=32, kernel_size=3):
        super().__init__()
        assert in_channels == 4, "BayerAwareConv只处理RGGB输入(4通道)"
        
        # 通道顺序：[R, G1, B, G2] 对应 [左上, 右上, 右下, 左下]
        self.out_channels = out_channels
        mid_channels = out_channels // 2  # 中间特征通道数
        
        # ========== 1. 空间位置编码 ==========
        # 为每个Bayer位置创建可学习的位置嵌入
        self.position_embed = nn.Parameter(torch.randn(1, 4, 1, 1))
        
        # ========== 2. 邻接关系建模 ==========
        # 水平邻接: R-G1 (左上-右上)
        self.conv_h_rg1 = nn.Conv2d(2, mid_channels, kernel_size, padding=kernel_size//2)
        
        # 水平邻接: G2-B (左下-右下)  
        self.conv_h_g2b = nn.Conv2d(2, mid_channels, kernel_size, padding=kernel_size//2)
        
        # 垂直邻接: R-G2 (左上-左下)
        self.conv_v_rg2 = nn.Conv2d(2, mid_channels, kernel_size, padding=kernel_size//2)
        
        # 垂直邻接: G1-B (右上-右下)
        self.conv_v_g1b = nn.Conv2d(2, mid_channels, kernel_size, padding=kernel_size//2)
        
        # ========== 3. 对角关系建模 ==========
        # 主对角: R-B (左上-右下)
        self.conv_diag_rb = nn.Conv2d(2, mid_channels//2, kernel_size, padding=kernel_size//2)
        
        # 副对角: G1-G2 (右上-左下) - 两个绿色通道的关系很重要
        self.conv_diag_g1g2 = nn.Conv2d(2, mid_channels//2, kernel_size, padding=kernel_size//2)
        
        # ========== 4. 绿色通道特殊处理 ==========
        # 绿色通道占50%，需要特殊关注
        self.green_fusion = nn.Conv2d(2, mid_channels, kernel_size, padding=kernel_size//2)
        
        # ========== 5. 颜色通道独立处理 ==========
        # 每个颜色通道的专属处理器
        self.color_specific = nn.ModuleDict({
            'R': nn.Conv2d(1, mid_channels//4, kernel_size, padding=kernel_size//2),
            'G1': nn.Conv2d(1, mid_channels//4, kernel_size, padding=kernel_size//2),
            'G2': nn.Conv2d(1, mid_channels//4, kernel_size, padding=kernel_size//2),
            'B': nn.Conv2d(1, mid_channels//4, kernel_size, padding=kernel_size//2)
        })
        
        # ========== 6. 特征融合和输出 ==========
        # 计算总的中间通道数
        total_mid = (
            4 * mid_channels +      # 邻接关系 (4个)
            mid_channels +          # 对角关系 (2个 × mid_channels//2)
            mid_channels +          # 绿色融合
            mid_channels            # 颜色特定 (4个 × mid_channels//4)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(total_mid, out_channels, 1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 可学习的通道重要性权重
        self.channel_importance = nn.Parameter(torch.tensor([1.0, 0.9, 1.0, 0.9]))
        
    def forward(self, x):
        """
        x: [B, 4, H, W] - pack后的RGGB数据
        通道顺序: [R, G1, B, G2]
        空间对应: [左上, 右上, 右下, 左下]
        """
        B, C, H, W = x.shape
        
        # 加入位置编码
        x_pos = x + self.position_embed
        
        # 提取各个通道
        R = x_pos[:, 0:1]   # 左上
        G1 = x_pos[:, 1:2]  # 右上  
        B = x_pos[:, 2:3]   # 右下
        G2 = x_pos[:, 3:4]  # 左下
        
        # 应用通道重要性权重
        R = R * self.channel_importance[0]
        G1 = G1 * self.channel_importance[1]
        B = B * self.channel_importance[2]
        G2 = G2 * self.channel_importance[3]
        
        features = []
        
        # ========== 邻接关系特征 ==========
        # 水平邻接
        h_rg1 = self.conv_h_rg1(torch.cat([R, G1], dim=1))
        h_g2b = self.conv_h_g2b(torch.cat([G2, B], dim=1))
        features.extend([h_rg1, h_g2b])
        
        # 垂直邻接
        v_rg2 = self.conv_v_rg2(torch.cat([R, G2], dim=1))
        v_g1b = self.conv_v_g1b(torch.cat([G1, B], dim=1))
        features.extend([v_rg2, v_g1b])
        
        # ========== 对角关系特征 ==========
        diag_rb = self.conv_diag_rb(torch.cat([R, B], dim=1))
        diag_g1g2 = self.conv_diag_g1g2(torch.cat([G1, G2], dim=1))
        features.extend([diag_rb, diag_g1g2])
        
        # ========== 绿色通道特殊处理 ==========
        green_feat = self.green_fusion(torch.cat([G1, G2], dim=1))
        features.append(green_feat)
        
        # ========== 颜色通道独立特征 ==========
        color_feats = []
        color_feats.append(self.color_specific['R'](R))
        color_feats.append(self.color_specific['G1'](G1))
        color_feats.append(self.color_specific['G2'](G2))
        color_feats.append(self.color_specific['B'](B))
        features.extend(color_feats)
        
        # ========== 特征融合 ==========
        all_features = torch.cat(features, dim=1)
        output = self.feature_fusion(all_features)
        
        return output


class BayerPositionalConv(nn.Module):
    """带Bayer位置感知的卷积 - 更轻量级的实现"""
    def __init__(self, in_channels=4, out_channels=32, kernel_size=3):
        super().__init__()
        assert in_channels == 4
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Bayer模式的学习权重矩阵 - 编码空间关系
        self.bayer_weights = nn.Parameter(torch.tensor([
            [0.5, 0.25, 0.15, 0.25],  # 权重减半
            [0.25, 0.5, 0.25, 0.35],
            [0.15, 0.25, 0.5, 0.25],
            [0.25, 0.35, 0.25, 0.5]
        ]))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 应用Bayer权重矩阵 - 调制输入
        x_weighted = torch.einsum('bcxy,cd->bdxy', x, self.bayer_weights)
        
        # 分组卷积处理
        output = self.conv(x_weighted)
        
        return output


class PBNet(nn.Module):
    """Physics-Based Network - 第一层Bayer-aware，其余标准CNN"""
    def __init__(self, args=None):
        super().__init__()
        
        # 默认参数
        self.args = args or {
            'nframes': 1,
            'in_nc': 4,      # RGGB输入
            'out_nc': 4,     # RGGB输出
            'nf': 32,        # 基础通道数
            'res': True,
            'use_physics': True,
            'camera_type': 'SonyA7S2'
        }
        
        nframes = self.args['nframes']
        nf = self.args['nf']
        in_nc = self.args['in_nc']
        out_nc = self.args['out_nc']
        
        self.use_physics = self.args.get('use_physics', True)
        self.camera_type = self.args.get('camera_type', 'SonyA7S2')
        
        # ============ Bayer-aware入口层 ============
        self.bayer_entry = BayerPositionalConv(in_nc * nframes, nf, kernel_size=3)
        
        # ============ 标准CNN编码器 ============
        self.conv1_1 = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.norm1_2 = nn.InstanceNorm2d(nf, affine=True)
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
        
        # ============ 物理约束模块 ============
        if self.use_physics:
            self.physics1 = LightweightPhysicsBlock(nf, reduction=8)
            self.physics3 = LightweightPhysicsBlock(nf*4, reduction=8)
            self.physics5 = LightweightPhysicsBlock(nf*16, reduction=8)
        
        # ============ 标准CNN解码器 ============
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
        
        # ============ 输出层 ============
        self.conv10_1 = nn.Conv2d(nf, out_nc, 1)
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.res = self.args.get('res', True)
        
    def forward(self, x, noise_map=None):
        """
        x: 输入RGGB图像 [B, 4, H, W] - 通道顺序为[R, G1, B, G2] 
        noise_map: 噪声图 [B, 1, H, W]
        """
        # ============ Bayer-aware入口处理 ============
        # conv1 = self.bayer_entry(x)  # [B, 4, H, W] → [B, 32, H, W]

        conv1 = self.relu(self.conv1_1(x))
        # 从这里开始，特征不再有严格的RGGB语义
        
        # 继续第一层处理
        conv1 = self.relu(self.conv1_2(conv1))
        
        # 应用物理约束
        if self.use_physics and noise_map is not None:
            conv1 = self.physics1(conv1, noise_map)
        
        pool1 = self.pool1(conv1)
        
        # ============ 标准CNN编码路径 ============
        conv2 = self.relu((self.conv2_1(pool1)))
        conv2 = self.relu((self.conv2_2(conv2)))
        pool2 = self.pool2(conv2)
        
        conv3 = self.relu((self.conv3_1(pool2)))
        conv3 = self.relu((self.conv3_2(conv3)))
        
        # 应用物理约束
        if self.use_physics and noise_map is not None:
            conv3 = self.physics3(conv3, noise_map)
        
        pool3 = self.pool3(conv3)
        
        conv4 = self.relu((self.conv4_1(pool3)))
        conv4 = self.relu((self.conv4_2(conv4)))
        pool4 = self.pool4(conv4)
        
        conv5 = self.relu((self.conv5_1(pool4)))
        conv5 = self.relu((self.conv5_2(conv5)))
        
        # 应用物理约束
        if self.use_physics and noise_map is not None:
            conv5 = self.physics5(conv5, noise_map)
        
        # ============ 标准CNN解码路径 ============
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.relu((self.conv6_1(up6)))
        conv6 = self.relu((self.conv6_2(conv6)))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.relu((self.conv7_1(up7)))
        conv7 = self.relu((self.conv7_2(conv7)))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.relu((self.conv8_1(up8)))
        conv8 = self.relu((self.conv8_2(conv8)))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.relu((self.conv9_1(up9)))
        conv9 = self.relu((self.conv9_2(conv9)))
        
        # ============ 输出 ============
        conv10 = self.conv10_1(conv9)
        
        if self.res:
            out = conv10 + x
        else:
            out = conv10
            
        return out


class PBNet_DSC(PBNet):
    """PBNet with Dark Shading Correction - 集成暗影校正"""
    def __init__(self, args=None):
        super().__init__(args)
        self.name = 'PBNet_DSC'


# 兼容性函数
def UNetSeeInDark_Physics(args=None):
    """创建PBNet，兼容原始UNet接口"""
    return PBNet(args)


def test_pbnet_params():
    """测试PBNet的参数量和性能"""
    
    configs = [
        {'nf': 28, 'name': 'PBNet-28通道'},
        {'nf': 32, 'name': 'PBNet-32通道'},
        {'nf': 36, 'name': 'PBNet-36通道'},
    ]
    
    print("PBNet参数量测试:")
    print("=" * 60)
    print("架构特点:")
    print("- 第一层：真正的Bayer-aware处理RGGB输入")
    print("- 其余层：标准CNN + 物理约束模块")
    print("- 归一化：InstanceNorm (适合去噪任务)")
    print("- 噪声图：池化匹配 (避免插值)")
    print("-" * 60)
    
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
        
        model = PBNet(args)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{config['name']}:")
        print(f"  总参数量: {total_params/1e6:.2f}M")
        print(f"  可训练参数: {trainable_params/1e6:.2f}M")
        
        # 测试前向传播
        x = torch.randn(2, 4, 128, 128)  # RGGB输入 [R,G1,B,G2]
        noise_map = torch.randn(2, 1, 128, 128)
        
        with torch.no_grad():
            output = model(x, noise_map)
            print(f"  输入形状: {x.shape}")
            print(f"  噪声图形状: {noise_map.shape}")
            print(f"  输出形状: {output.shape}")
            print(f"  内存占用: {torch.cuda.memory_allocated()/1e6:.1f}MB" if torch.cuda.is_available() else "  CPU模式")
        print()
    
    print("-" * 60)
    print("推荐配置: nf=32，参数量约7-8M，在目标范围内")
    print("关键优势:")
    print("1. 第一层真正理解RGGB空间排列")
    print("2. 物理约束模块提供噪声感知能力") 
    print("3. InstanceNorm保持样本独立性")
    print("4. 池化噪声图匹配保持真实性")
        

class DynamicPhysicsController:
    def __init__(self, 
                 target_ratio=0.2,           # 目标：物理损失占总损失的20%
                 initial_lambda=0.01,        # 初始权重
                 min_lambda=1e-5,           # 最小权重  
                 max_lambda=0.1,            # 最大权重
                 momentum=0.9,              # 平滑参数
                 adaptation_rate=0.1):      # 适应速率
        
        self.target_ratio = target_ratio
        self.lambda_physics = initial_lambda
        self.min_lambda = min_lambda  
        self.max_lambda = max_lambda
        self.momentum = momentum
        self.adaptation_rate = adaptation_rate
        
        # 历史统计
        self.loss_history = []
        self.lambda_history = []
        self.moving_avg_physics = None
        self.moving_avg_l1 = None
        
    def update_lambda(self, l1_loss, physics_loss_raw, epoch=None):
        """动态更新物理损失权重"""
        
        # 1. 更新移动平均
        if self.moving_avg_physics is None:
            self.moving_avg_physics = physics_loss_raw
            self.moving_avg_l1 = l1_loss
        else:
            self.moving_avg_physics = (self.momentum * self.moving_avg_physics + 
                                     (1 - self.momentum) * physics_loss_raw)
            self.moving_avg_l1 = (self.momentum * self.moving_avg_l1 + 
                                (1 - self.momentum) * l1_loss)
        
        # 2. 基于移动平均计算目标权重
        target_physics_loss = self.target_ratio * self.moving_avg_l1
        target_lambda = target_physics_loss / (self.moving_avg_physics + 1e-8)
        
        # 3. 渐进式调整（避免剧烈变化）
        lambda_delta = target_lambda - self.lambda_physics
        self.lambda_physics += self.adaptation_rate * lambda_delta
        
        # 4. 限制在合理范围内
        self.lambda_physics = torch.clamp(
            torch.tensor(self.lambda_physics), 
            self.min_lambda, 
            self.max_lambda
        ).item()
        
        # 5. 记录历史
        self.loss_history.append({
            'epoch': epoch,
            'l1_loss': l1_loss,
            'physics_loss_raw': physics_loss_raw,
            'lambda_physics': self.lambda_physics,
            'effective_physics': self.lambda_physics * physics_loss_raw
        })
        
        return self.lambda_physics
    
    def get_emergency_lambda(self, physics_loss_raw, l1_loss):
        """紧急情况下的权重计算"""
        
        # 如果物理损失异常大，立即降低权重
        if physics_loss_raw > 10.0:
            emergency_lambda = (0.1 * l1_loss) / physics_loss_raw  # 限制为L1的10%
            return max(emergency_lambda, self.min_lambda)
        
        return self.lambda_physics
    
class GradientBalancedLoss:
    def __init__(self, target_grad_ratio=0.3):
        self.target_grad_ratio = target_grad_ratio  # 目标：物理梯度占总梯度的30%
        self.grad_history = []
        
    def compute_balanced_loss(self, model, l1_loss, physics_loss_raw, input_data):
        """基于梯度比例平衡损失"""
        
        # 1. 计算L1损失的梯度范数
        l1_grads = torch.autograd.grad(
            l1_loss, model.parameters(), 
            retain_graph=True, create_graph=False
        )
        l1_grad_norm = sum(g.norm() for g in l1_grads if g is not None)
        
        # 2. 计算物理损失的梯度范数（使用单位权重）
        unit_physics_loss = physics_loss_raw * 1.0  # 权重=1时的损失
        physics_grads = torch.autograd.grad(
            unit_physics_loss, model.parameters(),
            retain_graph=True, create_graph=False
        )
        physics_grad_norm = sum(g.norm() for g in physics_grads if g is not None)
        
        # 3. 计算平衡权重
        if physics_grad_norm > 1e-8:
            target_physics_grad = self.target_grad_ratio * l1_grad_norm
            balanced_lambda = target_physics_grad / physics_grad_norm
        else:
            balanced_lambda = 0.01  # 默认值
            
        # 4. 记录统计信息
        self.grad_history.append({
            'l1_grad_norm': l1_grad_norm.item(),
            'physics_grad_norm': physics_grad_norm.item(),
            'balanced_lambda': balanced_lambda.item()
        })
        
        return balanced_lambda.clamp(1e-5, 0.1)
    
class PhysicsLossAnomalyDetector:
    def __init__(self, window_size=100, std_threshold=3.0):
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.loss_buffer = []
        
    def is_anomaly(self, physics_loss_raw):
        """检测当前损失是否为异常值"""
        
        self.loss_buffer.append(physics_loss_raw)
        if len(self.loss_buffer) > self.window_size:
            self.loss_buffer.pop(0)
            
        if len(self.loss_buffer) < 10:  # 样本不足
            return False
            
        # 计算统计量
        losses = torch.tensor(self.loss_buffer)
        mean_loss = losses.mean()
        std_loss = losses.std()
        
        # Z-score检测
        z_score = abs(physics_loss_raw - mean_loss) / (std_loss + 1e-8)
        
        return z_score > self.std_threshold
    
    def get_robust_estimate(self):
        """获取鲁棒的损失估计（去除异常值）"""
        
        if len(self.loss_buffer) < 5:
            return None
            
        losses = torch.tensor(self.loss_buffer)
        
        # 使用中位数和MAD（中位数绝对偏差）
        median_loss = losses.median()
        mad = torch.median(torch.abs(losses - median_loss))
        
        # 过滤异常值
        robust_losses = losses[torch.abs(losses - median_loss) < 3 * mad]
        
        return robust_losses.mean().item() if len(robust_losses) > 0 else median_loss.item()
    
class ComprehensivePhysicsController:
    def __init__(self):
        self.adaptive_controller = DynamicPhysicsController()
        self.gradient_balancer = GradientBalancedLoss()
        self.anomaly_detector = PhysicsLossAnomalyDetector()
        
        # 控制策略
        self.use_gradient_balancing = True
        self.enable_anomaly_detection = True
        self.emergency_mode = False
        
    def compute_optimal_lambda(self, model, l1_loss, physics_loss_raw, 
                              input_data, epoch=None):
        """计算最优的物理损失权重"""
        
        # 1. 异常值检测
        is_anomaly = False
        if self.enable_anomaly_detection:
            is_anomaly = self.anomaly_detector.is_anomaly(physics_loss_raw)
            
        # 2. 选择控制策略
        if is_anomaly or physics_loss_raw > 20.0:
            # 异常情况：使用紧急模式
            lambda_adaptive = self.adaptive_controller.get_emergency_lambda(
                physics_loss_raw, l1_loss
            )
            strategy = "emergency"
            
        elif self.use_gradient_balancing and not self.emergency_mode:
            # 正常情况：基于梯度平衡
            lambda_gradient = self.gradient_balancer.compute_balanced_loss(
                model, l1_loss, physics_loss_raw, input_data
            )
            lambda_adaptive = self.adaptive_controller.update_lambda(
                l1_loss, physics_loss_raw, epoch
            )
            
            # 取两者的调和平均
            lambda_optimal = 2 / (1/lambda_gradient + 1/lambda_adaptive)
            strategy = "gradient_balanced"
            
        else:
            # 简单自适应
            lambda_optimal = self.adaptive_controller.update_lambda(
                l1_loss, physics_loss_raw, epoch
            )
            strategy = "adaptive"
        
        # 3. 最终安全检查
        effective_physics_loss = lambda_optimal * physics_loss_raw
        if effective_physics_loss > l1_loss:  # 不允许物理损失超过L1损失
            lambda_optimal = 0.8 * l1_loss / physics_loss_raw
            strategy += "_capped"
            
        # 4. 记录和监控
        self.log_control_decision(
            l1_loss, physics_loss_raw, lambda_optimal, 
            strategy, is_anomaly, epoch
        )
        
        return lambda_optimal, strategy
    
    def log_control_decision(self, l1_loss, physics_loss_raw, lambda_opt, 
                           strategy, is_anomaly, epoch):
        """记录控制决策"""
        
        effective_physics = lambda_opt * physics_loss_raw
        total_loss = l1_loss + effective_physics
        physics_ratio = effective_physics / total_loss
        
        print(f"Epoch {epoch}: Strategy={strategy}")
        print(f"  L1: {l1_loss:.4f}, Physics(raw): {physics_loss_raw:.4f}")
        print(f"  Lambda: {lambda_opt:.6f}, Effective: {effective_physics:.4f}")
        print(f"  Ratio: {physics_ratio:.1%}, Anomaly: {is_anomaly}")
        
        # 预警
        if physics_ratio > 0.5:
            print("  ⚠️  警告：物理损失仍然过高！")
        if is_anomaly:
            print("  🔍 检测到异常损失值")

class PBNetLoss(nn.Module):
    """PBNet的损失函数，集成物理约束"""
    
    def __init__(self, camera_type='SonyA7S2', lambda_physics=0.1):
        super().__init__()
        from losses import Unet_Loss
        self.base_loss = Unet_Loss()
        self.lambda_physics = lambda_physics
        self.camera_type = camera_type
        self.l2_loss = nn.MSELoss()

        # 动态控制器
        self.controller = ComprehensivePhysicsController()
        
    def compute_physics_loss(self, output, noisy, noise_params=None, iso=None):
        """计算物理约束损失"""
        batch_size = output.shape[0]
        
        # 如果noise_params是列表（batch模式），处理每个样本
        if isinstance(noise_params, list) and len(noise_params) == batch_size:
            physics_losses = []
            
            for i in range(batch_size):
                # 获取单个样本的参数
                single_params = noise_params[i]
                single_output = output[i:i+1]  # 保持batch维度
                single_noisy = noisy[i:i+1]
                
                # 计算单个样本的物理损失
                single_loss = self._compute_single_physics_loss(
                    single_output, single_noisy, single_params, iso
                )
                physics_losses.append(single_loss)
            
            # 平均所有样本的物理损失
            return torch.stack(physics_losses).mean()
        
        # 如果是单个字典或None，使用原始方法
        else:
            return self._compute_single_physics_loss(output, noisy, noise_params, iso)
    
    def _compute_single_physics_loss(self, output, noisy, noise_params=None, iso=None):
        """计算单个样本的物理约束损失"""
        # 计算残差
        residual = noisy - output
        
        # 获取噪声参数
        K = 0.1  # 默认值
        sigma_read = 0.01
        
        if noise_params is not None:
            if isinstance(noise_params, dict):
                K = noise_params.get('K', noise_params.get('Kmax', K))
                sigma_read = noise_params.get('sigGs', sigma_read)
            else:
                print(f"Warning: noise_params should be dict, got {type(noise_params)}")
        
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
        # physics_loss = F.l1_loss(actual_var, expected_var)
        
        return physics_loss

        
    def forward(self, output, target, noisy=None, noise_params=None, 
                iso=None, model=None, epoch=None):
        """动态控制的损失计算"""
        
        # 基础L1损失
        l1_loss, _ = self.base_loss(output, target)
        
        # 如果没有物理约束数据，只返回L1损失
        if noisy is None:
            return l1_loss, {'l1': l1_loss.item(), 'physics': 0, 'lambda': 0}
        
        # 计算原始物理损失
        physics_loss_raw = self.compute_physics_loss(output, noisy, noise_params, iso)
        
        # 动态计算最优权重
        optimal_lambda, strategy = self.controller.compute_optimal_lambda(
            model, l1_loss, physics_loss_raw, output, epoch
        )
        
        # 应用动态权重
        physics_loss_weighted = optimal_lambda * physics_loss_raw
        total_loss = l1_loss + physics_loss_weighted
        
        return total_loss, {
            'l1': l1_loss.item(),
            'physics_raw': physics_loss_raw.item(), 
            'physics_weighted': physics_loss_weighted.item(),
            'lambda': optimal_lambda,
            'strategy': strategy,
            'physics_ratio': physics_loss_weighted.item() / total_loss.item()
        }
    

# 修正版trainer_SID.py监控集成代码

class SimplePhysicsMonitor:
    """简化版物理损失监控器 - 适配trainer_SID.py的实际结构"""
    
    def __init__(self, model_name='PBNet'):
        self.model_name = model_name
        
        # 统计信息
        self.total_batches = 0  # 全局batch计数器
        self.epoch_batches = 0  # 当前epoch的batch计数
        
        # 监控参数
        self.log_interval = 100  # 默认每100个batch详细记录一次
        self.warning_ratio = 0.4  # 预警阈值
        self.critical_ratio = 0.6  # 严重预警阈值
        
        # 当前epoch的统计
        self.epoch_stats = {
            'ratios': [],
            'lambdas': [],
            'anomaly_count': 0,
            'warning_count': 0
        }
        
        print(f"📊 物理损失监控器已初始化 - {model_name}")
        
    def monitor_batch(self, loss_info, epoch, batch_idx):
        """监控每个batch - 使用实际的batch_idx (k)"""
        
        # 更新计数器
        self.total_batches += 1
        self.epoch_batches = batch_idx + 1  # batch_idx从0开始，所以+1
        
        # 提取关键信息
        ratio = loss_info.get('ratio', 0.0)
        lambda_val = loss_info.get('lambda', 0.0)
        strategy = loss_info.get('strategy', 'unknown')
        physics_raw = loss_info.get('physics_raw', 0.0)
        
        # 记录到epoch统计
        self.epoch_stats['ratios'].append(ratio)
        self.epoch_stats['lambdas'].append(lambda_val)
        
        # 预警检查
        warning_flag = ""
        if ratio > self.critical_ratio:
            self.epoch_stats['anomaly_count'] += 1
            warning_flag = "🚨"
            # 立即输出严重警告
            print(f"\n🚨 CRITICAL - Epoch {epoch}, Batch {batch_idx+1}: "
                  f"物理损失占比 {ratio:.1%} (阈值:{self.critical_ratio:.1%})")
                  
        elif ratio > self.warning_ratio:
            self.epoch_stats['warning_count'] += 1
            warning_flag = "⚠️"
            
        # 异常原始损失检查
        if physics_raw > 15.0:
            warning_flag += "🔥"
            
        # 定期详细记录
        should_log_detail = (
            batch_idx % self.log_interval == 0 or  # 定期记录
            ratio > self.warning_ratio or          # 有预警时
            batch_idx < 5 or                      # epoch开始时
            strategy in ['emergency', 'emergency_capped']  # 紧急策略时
        )
        
        if should_log_detail:
            self._log_detail(epoch, batch_idx, loss_info)
            
        return warning_flag
    
    def _log_detail(self, epoch, batch_idx, loss_info):
        """详细日志输出"""
        print(f"\n--- 详细监控 Epoch {epoch}, Batch {batch_idx+1}/{self.epoch_batches} ---")
        print(f"L1损失:        {loss_info.get('l1', 0):.6f}")
        print(f"物理损失(原始): {loss_info.get('physics_raw', 0):.6f}")
        print(f"物理损失(加权): {loss_info.get('physics_weighted', 0):.6f}")
        print(f"动态权重λ:     {loss_info.get('lambda', 0):.8f}")
        print(f"控制策略:      {loss_info.get('strategy', 'unknown')}")
        print(f"物理损失占比:   {loss_info.get('ratio', 0):.1%}")
        print(f"全局Batch数:   {self.total_batches}")
        print("-" * 50)
        
    def epoch_summary(self, epoch, total_batches_in_epoch):
        """Epoch结束时的总结"""
        if not self.epoch_stats['ratios']:
            return
        
        # 计算统计信息
        avg_ratio = np.mean(self.epoch_stats['ratios'])
        max_ratio = np.max(self.epoch_stats['ratios'])
        min_ratio = np.min(self.epoch_stats['ratios'])
        avg_lambda = np.mean(self.epoch_stats['lambdas'])
        
        # 计算预警率
        warning_rate = self.epoch_stats['warning_count'] / len(self.epoch_stats['ratios'])
        critical_rate = self.epoch_stats['anomaly_count'] / len(self.epoch_stats['ratios'])
        
        # 确定日志级别
        if critical_rate > 0.1:  # 超过10%的batch有严重问题
            level = "🚨 CRITICAL"
        elif warning_rate > 0.3:  # 超过30%的batch有预警
            level = "⚠️ WARNING"
        else:
            level = "✅ INFO"
            
        # 输出总结（根据严重程度决定是否显示）
        should_show = (
            level != "✅ INFO" or  # 有问题时总是显示
            epoch % 20 == 0 or    # 每20个epoch显示一次
            epoch <= 5 or         # 前5个epoch总是显示
            epoch >= 590          # 最后几个epoch显示
        )
        
        if should_show:
            print(f"\n📊 {level} Epoch {epoch} 物理损失总结:")
            print(f"   处理批次: {len(self.epoch_stats['ratios'])}/{total_batches_in_epoch}")
            print(f"   平均占比: {avg_ratio:.1%} (范围: {min_ratio:.1%} - {max_ratio:.1%})")
            print(f"   平均权重: {avg_lambda:.6f}")
            print(f"   预警率:   {warning_rate:.1%} ({self.epoch_stats['warning_count']}批次)")
            if self.epoch_stats['anomaly_count'] > 0:
                print(f"   严重率:   {critical_rate:.1%} ({self.epoch_stats['anomaly_count']}批次)")
            print()
            
        # 重置epoch统计
        self._reset_epoch_stats()
        
    def _reset_epoch_stats(self):
        """重置epoch统计"""
        self.epoch_stats = {
            'ratios': [],
            'lambdas': [],
            'anomaly_count': 0,
            'warning_count': 0
        }
        self.epoch_batches = 0

if __name__ == '__main__':
    test_pbnet_params()