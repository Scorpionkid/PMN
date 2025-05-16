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

class SharedEncoder(nn.Module):
    """共享编码器，输出后分叉成细节和降噪两条路径"""
    def __init__(self, in_channels, out_channels):
        super(SharedEncoder, self).__init__()
        
        # 共享的特征提取器
        self.shared_features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        # 共享特征提取
        shared_feat = self.shared_features(x)
        
        return shared_feat


class DualPathBlock(nn.Module):
    """双路径块，包含细节路径和降噪路径"""
    def __init__(self, in_channels, out_channels, texture_params=None, heads=1):
        super(DualPathBlock, self).__init__()

        # 特征提取
        self.features = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 纹理参数
        texture_params = texture_params or {}

        # 双路径组件
        self.detail_path = EnhancedDetailPath(out_channels, num_heads=heads)
        self.denoise_path = EnhancedDenoisePath(out_channels, num_heads=heads)
        self.fusion = AGF(out_channels, use_noise_map=True, use_texture_mask=True)

    def forward(self, x, noise_map=None, texture_mask=None):
        # 特征提取
        feat = self.features(x)
        feat = self.activation(feat)

        # 细节路径
        detail = self.detail_path(feat, noise_map, texture_mask)

        # 降噪路径
        denoise = self.denoise_path(feat, noise_map, texture_mask)

        # 动态融合
        output = self.fusion(detail, denoise, feat, noise_map, texture_mask)

        return output, detail, denoise  # 修改：返回融合输出和两条路径的独立输出

class DualPathUNet_E1_Shared_UL(nn.Module):
    """double path U-Net, apply double path design on each scale of U-Net"""
    def __init__(self, args=None,  texture_params=None, **kwargs):
        super(DualPathUNet_E1_Shared_UL, self).__init__()

        base_channels = args['nf']
        in_channels = args['in_channels']
        out_channels = args['out_channels']
        heads = args['heads']
        self.use_wavelet_upsample = args['use_wavelet_upsample']
        self.use_sharpness_recovery = args['use_sharpness_recovery']
        self.use_noise_map = args['use_noise_map']
        self.use_texture_detection = args['use_texture_detection']
        self.enable_intermediate_supervision = args['enable_intermediate_supervision']  # 新增


        # enc1_in_channels = in_channels * 2 if use_noise_map else in_channels
        enc1_in_channels = in_channels

        # texture_detector
        self.texture_params = {
            'texture_gate': 0.5,
            'texture_suppress_factor': 0.7,
            'fusion_texture_boost': 0.5,
            'sharpness_texture_boost': 0.3,
        }
        if self.use_texture_detection:
            if texture_params is not None:
                self.texture_params.update(texture_params)

            # texture_detector params
            texture_detector_params = self.texture_params.get('texture_detector_params', {})

            window_sizes = texture_detector_params.get('window_sizes', [5, 9, 17])
            adaptive_thresh = texture_detector_params.get('adaptive_thresh', True)
            noise_sensitivity = texture_detector_params.get('noise_sensitivity', 3.0)

            self.texture_detector = RAWTextureDetector(
                window_sizes=window_sizes,
                adaptive_thresh=adaptive_thresh,
                raw_channels=in_channels,
                noise_sensitivity=noise_sensitivity
            )

         # 共享编码器
        self.enc1 = SharedEncoder(
            enc1_in_channels, base_channels
        )
        self.enc2 = SharedEncoder(
            base_channels, base_channels*2
        )
        self.enc3 = SharedEncoder(
            base_channels*2, base_channels*4
        )
        self.enc4 = SharedEncoder(
            base_channels*4, base_channels*8
        )

        # 瓶颈层 - 这里开始两条路径融合
        self.bottleneck = DualPathBlock(
            base_channels*8, base_channels*16,
            self.texture_params,
            heads[4]
        )

        # decoder with skip connections - 保持现有结构
        self.dec4 = DualPathBlock(base_channels*8+base_channels*8, base_channels*8, self.texture_params, heads[3])
        self.dec3 = DualPathBlock(base_channels*4+base_channels*4, base_channels*4, self.texture_params, heads[2])
        self.dec2 = DualPathBlock(base_channels*2+base_channels*2, base_channels*2, self.texture_params, heads[1])
        self.dec1 = DualPathBlock(base_channels+base_channels, base_channels, self.texture_params, heads[0])

        # downsample and upsample
        self.down = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)

        # final upsample, with optional wavelet upsample
        if self.use_wavelet_upsample:
            self.up1 = DiscreteWaveletUpsample(base_channels*2, base_channels)
        else:
            self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)

        # output layer
        self.final = nn.Conv2d(base_channels, out_channels, 1)

        # 新增：中间监督输出层
        if self.enable_intermediate_supervision:
            # 细节路径最终输出层
            self.detail_output = nn.Conv2d(base_channels, out_channels, 1)
            # 降噪路径最终输出层
            self.denoise_output = nn.Conv2d(base_channels, out_channels, 1)

        # sharpness recovery
        if self.use_sharpness_recovery:
            self.sharpness_recovery = SharpnessRecovery(
                out_channels,
                use_noise_map=True,
                use_texture_mask=True,
                sharpness_texture_boost=self.texture_params.get('sharpness_texture_boost', 0.3)
            )

    def forward(self, x, noise_map=None, texture_mask=None):

        if self.use_texture_detection:
            computed_texture_mask = self.texture_detector(x, noise_map)
            texture_mask = computed_texture_mask if texture_mask is None else texture_mask
            texture_mask = nmp.standardize_map(texture_mask)

            texture_maps = nmp.create_multiscale_maps(texture_mask, scales=[1, 2, 4, 6, 8])
            tm_original = texture_maps['scale_1']
            tm_down1 = texture_maps['scale_2']
            tm_down2 = texture_maps['scale_4']
            tm_down3 = texture_maps['scale_6']
            tm_down4 = texture_maps['scale_8']
        else:
            computed_texture_mask = None
            tm_original = tm_down1 = tm_down2 = tm_down3 = tm_down4 = None

        if self.use_noise_map and noise_map is not None:
            noise_map = nmp.standardize_map(noise_map)

            noise_maps = nmp.create_multiscale_maps(noise_map, scales=[1, 2, 4, 6, 8])
            nm_original = noise_maps['scale_1']
            nm_down1 = noise_maps['scale_2']
            nm_down2 = noise_maps['scale_4']
            nm_down3 = noise_maps['scale_6']
            nm_down4 = noise_maps['scale_8']
        else:
            nm_original = nm_down1 = nm_down2 = nm_down3 = nm_down4 = None

        #--------------------------- 修改：独立编码器路径 ---------------------------#
        # 编码器1
         #--------------------------- 共享编码器路径 ---------------------------#
        # 编码器阶段完全共享，只输出共享特征
        enc1 = self.enc1(x)
        enc1_down = self.down(enc1)
        
        enc2 = self.enc2(enc1_down)
        enc2_down = self.down(enc2)
        
        enc3 = self.enc3(enc2_down)
        enc3_down = self.down(enc3)
        enc4 = self.enc4(enc3_down)
        enc4_down = self.down(enc4)
        

        # 瓶颈层 - 合并两条路径
        bottleneck_input = enc4_down  
        bottleneck_output, bn_detail, bn_denoise = self.bottleneck(
            bottleneck_input, nm_down4, tm_down4
        )

        #--------------------------- 解码器路径 ---------------------------#
        # 从这里开始使用双路径融合块

        # 解码器3
        bottleneck_up = self.up4(bottleneck_output)
        # 将跳跃连接从两条独立路径连接
        dec4_input = torch.cat([bottleneck_up, enc4], dim=1)
        dec4_output, dec4_detail, dec4_denoise = self.dec4(
            dec4_input, nm_down3, tm_down3
        )
        dec4_up = self.up3(dec4_output)
        dec3_input = torch.cat([dec4_up, enc3], dim=1)
        dec3_output, dec3_detail, dec3_denoise = self.dec3(
            dec3_input, nm_down2, tm_down2
        )

        # 解码器2
        dec3_up = self.up2(dec3_output)
        dec2_input = torch.cat([dec3_up, enc2], dim=1)
        dec2_output, dec2_detail, dec2_denoise = self.dec2(
            dec2_input, nm_down1, tm_down1
        )

        # 解码器1
        dec2_up = self.up1(dec2_output)
        dec1_input = torch.cat([dec2_up, enc1], dim=1)
        dec1_output, dec1_detail, dec1_denoise = self.dec1(
            dec1_input, nm_original, tm_original
        )

        #--------------------------- 新增：中间监督输出 ---------------------------#

        main_output = self.final(dec1_output)

        # 可选的中间监督输出
        detail_output = None
        denoise_output = None

        if self.enable_intermediate_supervision:
            detail_output = self.detail_output(dec1_detail)
            denoise_output = self.denoise_output(dec1_denoise)

            # detail_output = torch.tanh(detail_output) * 0.5 + 0.5
            detail_output = torch.clamp(detail_output, 0.0, 1.0)
            denoise_output = torch.clamp(denoise_output, 0.0, 1.0)

            nmp.detect_nan(detail_output, "detail path output")
            nmp.detect_nan(denoise_output, "denoise path output")

        # sharpness recovery
        if self.use_sharpness_recovery:
            main_output = self.sharpness_recovery(main_output, nm_original, tm_original)

        return main_output, texture_mask, detail_output, denoise_output