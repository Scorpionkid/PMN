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
        
        # 为保持轻量级，使用分组卷积
        self.groups = 4
        inter_channels = out_channels
        
        # Bayer感知的分组卷积 - 每组处理特定的空间关系
        self.grouped_conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 
                     kernel_size, padding=kernel_size//2, groups=self.groups),
            nn.InstanceNorm2d(inter_channels, affine=True),  # 添加归一化
            nn.LeakyReLU(0.2, inplace=True)  # 添加激活函数
        )
        
        # 跨通道交互 - 建模Bayer模式关系
        self.pattern_aware = nn.Sequential(
            # 深度可分离卷积风格
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, groups=inter_channels),
            nn.Conv2d(inter_channels, out_channels, 1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
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
        feat = self.grouped_conv(x_weighted)
        
        # 跨通道交互和输出
        output = self.pattern_aware(feat)

        # output = torch.tanh(output)
        
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
        conv1 = self.bayer_entry(x)  # [B, 4, H, W] → [B, 32, H, W]
        # 从这里开始，特征不再有严格的RGGB语义
        
        # 继续第一层处理
        conv1 = self.relu(self.norm1_2(self.conv1_2(conv1)))
        
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
    test_pbnet_params()