import torch
import torch.nn as nn

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