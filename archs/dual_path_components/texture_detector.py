import torch
import torch.nn as nn
import torch.nn.functional as F
# from led.utils.logger import get_root_logger

class RAWTextureDetector(nn.Module):
    """RAW图像纹理检测器 - 适配RAW图像的4通道输入

    Args:
        window_sizes (list): 用于多尺度纹理检测的窗口大小列表
        base_lower_thresh (float): 标准差的基础低阈值
        base_upper_thresh (float): 标准差的基础高阈值
        adaptive_thresh (bool): 是否使用自适应阈值
        raw_channels (int): RAW图像的通道数，默认为4 (RGGB)
    """
    def __init__(self, window_sizes=[11, 25, 49], base_lower_thresh=0.05,
                 base_upper_thresh=0.2, adaptive_thresh=True, raw_channels=4,
                 noise_sensitivity=3.0):
        super(RAWTextureDetector, self).__init__()
        self.window_sizes = window_sizes
        self.base_lower_thresh = base_lower_thresh
        self.base_upper_thresh = base_upper_thresh
        self.adaptive_thresh = adaptive_thresh
        self.raw_channels = raw_channels
        # self.noise_sensitivity = nn.Parameter(torch.tensor(noise_sensitivity))

        # RAW图像特征提取，4通道输入
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(raw_channels, 32, 3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(32, 16, 3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(16, 8, 3, padding=1),
        # )

        # 通道特定权重 - 学习不同颜色通道的重要性
        # self.channel_weights = nn.Parameter(torch.ones(raw_channels) / raw_channels)

        # 多尺度融合
        # if len(window_sizes) > 1:
        #     self.scale_fusion = nn.Sequential(
        #         nn.Conv2d(len(window_sizes), 1, 1),
        #         nn.ReLU()  # 确保输出非负
        #     )
        #     # 初始化为均匀权重
        #     nn.init.constant_(self.scale_fusion[0].weight, 1.0 / len(window_sizes))
        #     nn.init.zeros_(self.scale_fusion[0].bias)

    def forward(self, x, noise_map=None):
        """前向传播

        Args:
            x (Tensor): 输入RAW图像 [B, 4, H, W]
            noise_map (Tensor, optional): 噪声图，如果有的话

        Returns:
            texture_mask (Tensor): 纹理掩码 [B, 1, H, W]，值范围[0,1]
        """
        # # 提取特征
        # features = self.feature_extractor(x)

        # # 多尺度纹理检测
        # texture_maps = []
        # for window_size in self.window_sizes:
        #     # 计算当前尺度的标准差图
        #     std_map = self._compute_std_map(features, window_size)
        #     texture_maps.append(std_map)

        # # 融合多尺度纹理图
        # if len(self.window_sizes) > 1:
        #     multi_scale_texture = torch.cat(texture_maps, dim=1)
        #     fused_std_map = self.scale_fusion(multi_scale_texture)
        # else:
        #     fused_std_map = texture_maps[0]

        # # 获取阈值 - 固定或自适应
        # if self.adaptive_thresh:
        #     lower_thresh, upper_thresh = self._compute_adaptive_thresholds(x, fused_std_map)
        # else:
        #     lower_thresh, upper_thresh = self.base_lower_thresh, self.base_upper_thresh

        # # 生成纹理掩码
        # texture_mask = self._generate_texture_mask(fused_std_map, lower_thresh, upper_thresh)
        texture_mask = self.simple_texture_detector(x)

        return texture_mask

    def _compute_std_map(self, features, window_size):
        N, C, H, W = features.shape
        pad = window_size // 2

        features_padded = F.pad(features, [pad]*4, mode='reflect')

        avg_kernel = torch.ones(1, 1, window_size, window_size,
                            device=features.device) / (window_size**2)

        result = torch.zeros(N, 1, H, W, device=features.device, dtype=torch.float32)

        if C == 4:  # 原始RGGB输入
            channel_weights = torch.tensor([1.0, 1.5, 1.0, 1.5], device=features.device)
        else:  # 特征提取后的输入
            channel_weights = torch.ones(C, device=features.device)

        channel_weights = F.softmax(channel_weights, dim=0)

        for c in range(C):
            curr_feat = features_padded[:, c:c+1]

            local_mean = F.conv2d(curr_feat, avg_kernel, stride=1, padding=0, groups=1)

            local_mean_sq = F.conv2d(curr_feat**2, avg_kernel, stride=1, padding=0, groups=1)
            local_var = torch.clamp(local_mean_sq - local_mean**2, min=1e-8)

            if torch.isnan(local_var).any():
                # 将 NaN 替换为小正数
                self.logger.info(f"局部标准差出现nan")
                # local_var = torch.nan_to_num(local_var, nan=1e-8)

            result += channel_weights[c] * torch.sqrt(local_var)

        result = torch.pow(result, 0.8)
        # result = result / (torch.mean(result) + 1e-6)

        return result

    def _compute_adaptive_thresholds(self, x, std_map):
        """计算完全基于图像统计量的自适应阈值"""

        # 计算std_map的分位点
        std_flat = std_map.view(-1)
        q25 = torch.quantile(std_flat, 0.25)
        q75 = torch.quantile(std_flat, 0.75)

        # 计算四分位距(IQR)
        iqr = q75 - q25

        # 基于全局标准差
        global_std = torch.std(x, dim=(2, 3), keepdim=True)

        # # 直接使用分位点计算基础阈值，而非使用预设的base_lower/upper_thresh
        lower_thresh = (q25 - 0.5 * iqr).view(1, 1, 1, 1).expand_as(global_std)
        upper_thresh = (q75 + 0.5 * iqr).view(1, 1, 1, 1).expand_as(global_std)

        # 应用全局标准差调整
        global_factor = torch.clamp(global_std * 5.0, 0.5, 2.0)
        lower_thresh = lower_thresh * global_factor
        upper_thresh = upper_thresh * global_factor

        # 应用在通道维度上的权重(保留原有功能)
        channel_weights = F.softmax(self.channel_weights, dim=0).view(1, self.raw_channels, 1, 1)
        weighted_x = x * channel_weights
        channel_std = torch.std(weighted_x, dim=(2, 3), keepdim=True)

        # 进一步基于各通道的特性调整阈值(保留原有功能)
        channel_factor = torch.clamp(channel_std * 2.0, 0.8, 1.2)
        lower_thresh = lower_thresh * channel_factor
        upper_thresh = upper_thresh * channel_factor

        return lower_thresh, upper_thresh

    def _generate_texture_mask(self, std_map, lower_thresh, upper_thresh):

        normalized_std = (std_map - lower_thresh) / (upper_thresh - lower_thresh)
        normalized_std = torch.clamp(normalized_std, 0.0, 1.0)

        # 使用sigmoid函数实现平滑过渡
        texture_mask = torch.sigmoid(6.0 * normalized_std - 3.0)  # 调整斜率使过渡更加明显
        texture_mask = torch.mean(texture_mask, dim=1, keepdim=True)

        return texture_mask

    def simple_texture_detector(self, x):
        """简单且鲁棒的纹理检测器"""
        # 转换为灰度
        if x.size(1) == 4:  # RAW RGGB格式
            gray = 0.299 * x[:,0:1] + 0.587 * (x[:,1:2] + x[:,2:3])/2 + 0.114 * x[:,3:4]
        else:
            gray = torch.mean(x, dim=1, keepdim=True)

        # Sobel边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32).reshape(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32).reshape(1, 1, 3, 3).to(x.device)

        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)

        # 计算梯度幅度
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        # 归一化
        norm_grad = grad_mag / (torch.max(grad_mag) + 1e-6)
        texture_mask = 0.1 + 0.8 * norm_grad

        return texture_mask