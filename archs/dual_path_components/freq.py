import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyEnhancement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.amp_fuse = nn.Conv2d(channels, channels, 1, groups=channels)
        self.pha_fuse = nn.Conv2d(channels, channels, 1, groups=channels)
        
    def forward(self, x):
        _, _, H, W = x.shape
        
        # FFT变换
        fft = torch.fft.rfft2(x, norm='ortho')
        amp = torch.abs(fft)
        pha = torch.angle(fft)
        
        # 轻量级频域处理
        amp_fea = self.amp_fuse(amp)
        pha_fea = self.pha_fuse(pha)
        
        # 重建 - 避免inplace操作
        real = amp_fea * torch.cos(pha_fea)
        imag = amp_fea * torch.sin(pha_fea)
        fft_fea = torch.complex(real, imag)  # 创建新的复数张量
        
        # 逆变换
        output = torch.fft.irfft2(fft_fea, s=(H, W), norm='ortho')
        
        return output
    
class AdaptiveFreqEnhancement(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 轻量级的频率选择网络
        self.freq_selector = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, noise_level=None):
        # FFT变换
        freq = torch.fft.rfft2(x, norm='ortho')
        
        # 基于噪声水平的自适应频率选择
        if noise_level is not None:
            # 噪声越大，保留的高频越少
            freq_mask = self.freq_selector(noise_level)
            freq = freq * freq_mask
        
        # 增强中低频，抑制高频噪声
        freq_magnitude = torch.abs(freq)
        freq_phase = torch.angle(freq)
        
        # 软阈值处理
        threshold = self.adaptive_threshold(freq_magnitude, noise_level)
        freq_magnitude = F.relu(freq_magnitude - threshold) + threshold
        
        # 重构
        freq_complex = freq_magnitude * torch.exp(1j * freq_phase)
        enhanced = torch.fft.irfft2(freq_complex, s=x.shape[-2:], norm='ortho')
        
        return enhanced