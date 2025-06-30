import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class HaarWavelet(nn.Module):
    """PyTorch实现的Haar小波变换"""
    def __init__(self):
        super(HaarWavelet, self).__init__()

        # Haar小波滤波器
        ll_filter = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh_filter = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        hl_filter = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        hh_filter = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        # 注册为buffer，不需要梯度
        self.register_buffer('ll_filter', ll_filter.unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh_filter', lh_filter.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl_filter', hl_filter.unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh_filter', hh_filter.unsqueeze(0).unsqueeze(0))

    def dwt(self, x):
        """离散小波变换"""
        # 输入: [B, C, H, W]

        # 将每个通道应用小波变换
        b, c, h, w = x.shape

        # 保证高度和宽度是偶数
        if h % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))
            h += 1
        if w % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0))
            w += 1

        # 重塑以便可以应用卷积
        x_reshaped = x.view(-1, 1, h, w)

        # 使用步长为2的卷积来计算小波系数
        ll = F.conv2d(x_reshaped, self.ll_filter, stride=2, padding=0)
        lh = F.conv2d(x_reshaped, self.lh_filter, stride=2, padding=0)
        hl = F.conv2d(x_reshaped, self.hl_filter, stride=2, padding=0)
        hh = F.conv2d(x_reshaped, self.hh_filter, stride=2, padding=0)

        # 重新整形返回
        ll = ll.view(b, c, h//2, w//2)
        lh = lh.view(b, c, h//2, w//2)
        hl = hl.view(b, c, h//2, w//2)
        hh = hh.view(b, c, h//2, w//2)

        return ll, lh, hl, hh

    def idwt(self, ll, lh, hl, hh):
        """逆离散小波变换"""
        # 输入: 4个张量 [B, C, H/2, W/2]

        # 确保所有输入具有相同形状
        assert all(x.shape == ll.shape for x in [lh, hl, hh]), "All wavelet components must have the same shape"

        b, c, h, w = ll.shape
        out_h, out_w = h * 2, w * 2

        # 准备进行上采样
        ll_up = torch.zeros((b, c, out_h, out_w), device=ll.device)
        lh_up = torch.zeros((b, c, out_h, out_w), device=ll.device)
        hl_up = torch.zeros((b, c, out_h, out_w), device=ll.device)
        hh_up = torch.zeros((b, c, out_h, out_w), device=ll.device)

        # 填充上采样的张量 (棋盘格图案)
        ll_up[:, :, 0::2, 0::2] = ll
        lh_up[:, :, 0::2, 1::2] = lh
        hl_up[:, :, 1::2, 0::2] = hl
        hh_up[:, :, 1::2, 1::2] = hh

        # 组合所有分量
        return ll_up + lh_up + hl_up + hh_up


class WaveletUpsample(nn.Module):
    """使用小波变换的上采样模块，替换传统上采样以保留更多细节"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(WaveletUpsample, self).__init__()
        self.scale_factor = scale_factor

        # 特征预处理
        self.preconv = nn.Conv2d(in_channels, out_channels*4, 3, 1, 1)

        # 小波变换模块
        self.wavelet = HaarWavelet()

    def forward(self, x):
        # 特征预处理
        x = self.preconv(x)

        batch, channels = x.shape[0], x.shape[1]//4

        # 分割为近似和细节系数
        ll = x[:, :channels, :, :]
        lh = x[:, channels:channels*2, :, :]
        hl = x[:, channels*2:channels*3, :, :]
        hh = x[:, channels*3:, :, :]

        # 使用逆小波变换进行上采样
        y = self.wavelet.idwt(ll, lh, hl, hh)

        return y


class RawWaveletUpsample(nn.Module):
    """为RAW图像处理设计的小波上采样"""
    def __init__(self, in_channels, out_channels):
        super(RawWaveletUpsample, self).__init__()

        # 为每个颜色通道使用单独的小波处理
        self.r_branch = WaveletUpsample(in_channels, out_channels//4)
        self.g_branch = WaveletUpsample(in_channels, out_channels//2)
        self.b_branch = WaveletUpsample(in_channels, out_channels//4)

        # 特征融合
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        # 分离RGGB通道
        # 假设输入是4通道表示RGGB
        r = x[:, 0:1, :, :]
        g1 = x[:, 1:2, :, :]
        g2 = x[:, 3:4, :, :]  # 注意RGGB格式，G2实际上是第4个通道
        b = x[:, 2:3, :, :]

        g = (g1 + g2) / 2

        # 对每个通道进行小波上采样
        r_up = self.r_branch(r)
        g_up = self.g_branch(g)
        b_up = self.b_branch(b)

        # 合并三个颜色通道
        combined = torch.cat([r_up, g_up, b_up], dim=1)
        out = self.fusion(combined)

        return out


# 高级小波上采样变体，使用PyWavelets库进行更精确的小波变换
class PyWTWaveletUpsample(nn.Module):
    """使用PyWavelets库的小波上采样模块"""
    def __init__(self, in_channels, out_channels, wavelet='haar'):
        super(PyWTWaveletUpsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet = wavelet

        # 预处理卷积
        self.preconv = nn.Conv2d(in_channels, out_channels*4, 3, 1, 1)

        # 后处理卷积
        self.postconv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        # 特征预处理
        x = self.preconv(x)

        batch, channels = x.shape[0], x.shape[1]//4
        h, w = x.shape[2], x.shape[3]

        # 分割为近似和细节系数
        ll = x[:, :channels, :, :]
        lh = x[:, channels:channels*2, :, :]
        hl = x[:, channels*2:channels*3, :, :]
        hh = x[:, channels*3:, :, :]

        # 使用PyWavelets进行逆小波变换
        result = []
        for b in range(batch):
            coeffs = []
            for c in range(channels):
                # 构建小波系数
                coeffs_c = [ll[b, c].detach().cpu().numpy(),
                          (hl[b, c].detach().cpu().numpy(),
                           lh[b, c].detach().cpu().numpy(),
                           hh[b, c].detach().cpu().numpy())]

                # 使用PyWavelets进行IDWT
                reconstructed = pywt.idwt2(coeffs_c, self.wavelet)
                result.append(torch.from_numpy(reconstructed).to(x.device))

        # 重新整形
        y = torch.stack(result, dim=0).reshape(batch, channels, h*2, w*2)

        # 应用后处理
        y = self.postconv(y)

        return y


# 纯PyTorch实现的离散小波变换上采样
class DiscreteWaveletUpsample(nn.Module):
    """纯PyTorch实现的离散小波变换上采样，支持反向传播"""
    def __init__(self, in_channels, out_channels):
        super(DiscreteWaveletUpsample, self).__init__()

        # 输入通道预处理
        self.preconv = nn.Conv2d(in_channels, out_channels*4, 3, 1, 1)

        # 小波变换系数 - Haar小波
        # 生成用于上采样的核心滤波器
        self.register_buffer('up_filter', self._create_filters())

        # 后处理卷积增强特征
        self.postconv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def _create_filters(self):
        """创建Haar小波的上采样滤波器"""
        filter_lo = torch.tensor([1/2, 1/2], dtype=torch.float32)
        filter_hi = torch.tensor([1/2, -1/2], dtype=torch.float32)

        # 2D滤波器
        filters = torch.zeros(4, 1, 2, 2, dtype=torch.float32)

        # LL, LH, HL, HH滤波器
        filters[0, 0, 0, 0] = filter_lo[0] * filter_lo[0]
        filters[0, 0, 0, 1] = filter_lo[0] * filter_lo[1]
        filters[0, 0, 1, 0] = filter_lo[1] * filter_lo[0]
        filters[0, 0, 1, 1] = filter_lo[1] * filter_lo[1]

        filters[1, 0, 0, 0] = filter_lo[0] * filter_hi[0]
        filters[1, 0, 0, 1] = filter_lo[0] * filter_hi[1]
        filters[1, 0, 1, 0] = filter_lo[1] * filter_hi[0]
        filters[1, 0, 1, 1] = filter_lo[1] * filter_hi[1]

        filters[2, 0, 0, 0] = filter_hi[0] * filter_lo[0]
        filters[2, 0, 0, 1] = filter_hi[0] * filter_lo[1]
        filters[2, 0, 1, 0] = filter_hi[1] * filter_lo[0]
        filters[2, 0, 1, 1] = filter_hi[1] * filter_lo[1]

        filters[3, 0, 0, 0] = filter_hi[0] * filter_hi[0]
        filters[3, 0, 0, 1] = filter_hi[0] * filter_hi[1]
        filters[3, 0, 1, 0] = filter_hi[1] * filter_hi[0]
        filters[3, 0, 1, 1] = filter_hi[1] * filter_hi[1]

        return filters

    def forward(self, x):
        # 特征预处理
        x = self.preconv(x)

        batch, channels = x.shape[0], x.shape[1]//4
        h, w = x.shape[2], x.shape[3]

        # 分割小波系数
        coeffs = torch.split(x, channels, dim=1)
        ll, lh, hl, hh = coeffs

        # 准备上采样
        y = torch.zeros(batch, channels, h*2, w*2, device=x.device)

        # 对每个通道应用IDWT
        for b in range(batch):
            for c in range(channels):
                # 获取当前通道的系数
                ll_c = ll[b, c].unsqueeze(0).unsqueeze(0)
                lh_c = lh[b, c].unsqueeze(0).unsqueeze(0)
                hl_c = hl[b, c].unsqueeze(0).unsqueeze(0)
                hh_c = hh[b, c].unsqueeze(0).unsqueeze(0)

                # 上采样每个系数分量
                ll_up = F.conv_transpose2d(ll_c, self.up_filter[0:1], stride=2)
                lh_up = F.conv_transpose2d(lh_c, self.up_filter[1:2], stride=2)
                hl_up = F.conv_transpose2d(hl_c, self.up_filter[2:3], stride=2)
                hh_up = F.conv_transpose2d(hh_c, self.up_filter[3:4], stride=2)

                # 组合所有分量
                y[b, c] = (ll_up + lh_up + hl_up + hh_up).squeeze(0).squeeze(0)

        # 应用后处理
        y = self.postconv(y)

        return y