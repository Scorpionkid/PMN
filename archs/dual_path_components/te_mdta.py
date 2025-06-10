import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class TextureEnhancedMDTA(nn.Module):
    """使用外部纹理掩码的纹理增强多头深度变换注意力模块"""

    def __init__(self, channels, num_heads=1):
        super(TextureEnhancedMDTA, self).__init__()

        self.num_heads = num_heads
        # 分别为高频和低频内容使用不同的温度参数
        self.detail_temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 1.5)  # 更高的温度保留更多细节
        self.smooth_temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)  # 更低的温度平滑区域

        # 标准MDTA组件
        self.qkv = nn.Conv2d(channels, channels*3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(channels*3, channels*3, kernel_size=3, padding=1, groups=channels*3)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # 纹理增强参数
        self.texture_boost = nn.Parameter(torch.tensor(0.3))  # 纹理区域增强因子

    def forward(self, x, texture_mask=None):
        """前向传播，内存高效实现

        Args:
            x (Tensor): 输入特征图 [B, C, H, W]
            texture_mask (Tensor, optional): 外部提供的纹理掩码 [B, 1, H, W]

        Returns:
            Tensor: 处理后的特征图
        """
        b, c, h, w = x.shape

        # 1. 常规MDTA处理
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重塑张量用于多头处理
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 应用L2归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 2. 纹理感知的温度调整
        if texture_mask is not None:

            # 全局纹理水平
            texture_level = F.adaptive_avg_pool2d(texture_mask, (1, 1)).view(b, 1, 1, 1)

            # 修改温度参数 - 改变注意力集中程度
            temp = self.detail_temperature * texture_level + self.smooth_temperature * (1 - texture_level)
        else:
            temp = (self.detail_temperature + self.smooth_temperature) / 2

        # 3. 计算注意力
        attn = (q @ k.transpose(-2, -1)) * temp

        # 4. 纹理感知处理 - 纹理信息直接应用于值(v)
        if texture_mask is not None:
            texture_boost = torch.clamp(self.texture_boost, 0.1, 0.5)
            # 将纹理掩码重塑为与v兼容的形状
            texture_mask_reshaped = texture_mask.view(b, 1, 1, h*w)
            # 对v进行加权 - 增强纹理区域的特征强度
            v = v * (1.0 + texture_mask_reshaped * texture_boost)

        # 5. 应用注意力
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        # 6. 重塑输出
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 7. 输出投影
        out = self.output_proj(out)

        return out
