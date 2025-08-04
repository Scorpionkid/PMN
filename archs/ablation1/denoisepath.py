import torch
import torch.nn as nn
import torch.nn.functional as F
import noise_map_processor as nmp

from ..dual_path_components import (
    DynamicFusion, AGF,
    WaveletUpsample, DiscreteWaveletUpsample,
    SharpnessRecovery,
    RAWTextureDetector,
    EnhancedDenoisePath,
    EnhancedDetailPath,
    Freq
)


class IndependentPathEncoder(nn.Module):
    """编码器模块，独立处理细节或降噪路径，不进行融合"""
    def __init__(self, in_channels, out_channels, is_detail_path=True,
                 use_noise_map=False, use_texture_detection=False,
                 heads=1, texture_params=None):
        super(IndependentPathEncoder, self).__init__()

        # 特征提取
        self.features = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 根据路径类型选择相应模块
 
        self.path = EnhancedDenoisePath(out_channels, heads)

        self.use_noise_map = use_noise_map
        self.use_texture_detection = use_texture_detection

    def forward(self, x, noise_map=None, texture_mask=None):
        # 特征提取
        feat = self.features(x)
        feat = self.activation(feat)

        # 路径处理
        output = self.path(feat, noise_map, texture_mask)

        return output

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
        self.denoise_path = EnhancedDenoisePath(out_channels, num_heads=heads)

    def forward(self, x, noise_map=None, texture_mask=None):
        # 特征提取
        feat = self.features(x)
        feat = self.activation(feat)

        # 降噪路径
        denoise = self.denoise_path(feat, noise_map, texture_mask)

        return denoise  # 修改：返回融合输出和两条路径的独立输出

class DPNet_onlydenoise(nn.Module):
    """double path U-Net, apply double path design on each scale of U-Net"""
    def __init__(self, args=None,  texture_params=None, **kwargs):
        super(DPNet_onlydenoise, self).__init__()

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

        # 修改：替换编码器为独立路径编码器
        

        # 降噪路径编码器
        self.enc1_denoise = IndependentPathEncoder(
            enc1_in_channels, base_channels,
            is_detail_path=False,
            use_noise_map=self.use_noise_map,
            use_texture_detection=self.use_texture_detection,
            heads=heads[0],
            texture_params=self.texture_params
        )
        self.enc2_denoise = IndependentPathEncoder(
            base_channels, base_channels*2,
            is_detail_path=False,
            use_noise_map=self.use_noise_map,
            use_texture_detection=self.use_texture_detection,
            heads=heads[1],
            texture_params=self.texture_params
        )
        self.enc3_denoise = IndependentPathEncoder(
            base_channels*2, base_channels*4,
            is_detail_path=False,
            use_noise_map=self.use_noise_map,
            use_texture_detection=self.use_texture_detection,
            heads=heads[2],
            texture_params=self.texture_params
        )

        self.freq_enhance_detail = Freq(base_channels*4)
        self.freq_enhance_denoise = Freq(base_channels*4)

        # 瓶颈层 - 这里开始两条路径融合
        self.bottleneck = DualPathBlock(
            base_channels*4, base_channels*8,
            self.texture_params,
            heads[3]
        )

        # decoder with skip connections - 保持现有结构
        self.dec3 = DualPathBlock(base_channels*4+base_channels*4, base_channels*4, self.texture_params, heads[2])
        self.dec2 = DualPathBlock(base_channels*2+base_channels*2, base_channels*2, self.texture_params, heads[1])
        self.dec1 = DualPathBlock(base_channels+base_channels, base_channels, self.texture_params, heads[0])

        # downsample and upsample
        self.down = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)

        # final upsample, with optional wavelet upsample
        if self.use_wavelet_upsample:
            self.up1 = DiscreteWaveletUpsample(base_channels*2, base_channels)
        else:
            self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)

        # output layer
        self.final = nn.Conv2d(base_channels, out_channels, 1)

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

            texture_maps = nmp.create_multiscale_maps(texture_mask, scales=[1, 2, 4, 6])
            tm_original = texture_maps['scale_1']
            tm_down1 = texture_maps['scale_2']
            tm_down2 = texture_maps['scale_4']
            tm_down3 = texture_maps['scale_6']
        else:
            computed_texture_mask = None
            tm_original = tm_down1 = tm_down2 = tm_down3 = None

        if self.use_noise_map and noise_map is not None:
            noise_map = nmp.standardize_map(noise_map)

            noise_maps = nmp.create_multiscale_maps(noise_map, scales=[1, 2, 4, 6])
            nm_original = noise_maps['scale_1']
            nm_down1 = noise_maps['scale_2']
            nm_down2 = noise_maps['scale_4']
            nm_down3 = noise_maps['scale_6']
        else:
            nm_original = nm_down1 = nm_down2 = nm_down3 = None

        #--------------------------- 修改：独立编码器路径 ---------------------------#

        # 降噪路径编码
        enc1_denoise = self.enc1_denoise(x, nm_original, tm_original)
        enc1_denoise_down = self.down(enc1_denoise)
        enc2_denoise = self.enc2_denoise(enc1_denoise_down, nm_down1, tm_down1)
        enc2_denoise_down = self.down(enc2_denoise)
        enc3_denoise = self.enc3_denoise(enc2_denoise_down, nm_down2, tm_down2)
        enc3_denoise_down = self.down(enc3_denoise)

        # 瓶颈层 - 合并两条路径
        # 将两个独立路径的特征连接起来输入到瓶颈层
        enc3_denoise_enhanced = enc3_denoise_down + self.freq_enhance_denoise(enc3_denoise_down)

        # 然后使用增强后的特征
        bottleneck_input = enc3_denoise_enhanced
        bottleneck_output, bn_detail, bn_denoise = self.bottleneck(
            bottleneck_input, nm_down3, tm_down3
        )

        #--------------------------- 解码器路径 ---------------------------#
        # 从这里开始使用双路径融合块

        # 解码器3
        bottleneck_up = self.up3(bottleneck_output)
        # 将跳跃连接从两条独立路径连接
        dec3_input = torch.cat([bottleneck_up, enc3_denoise], dim=1)
        dec3_output = self.dec3(
            dec3_input, nm_down2, tm_down2
        )

        # 解码器2
        dec3_up = self.up2(dec3_output)
        dec2_input = torch.cat([dec3_up, enc2_denoise], dim=1)
        dec2_output = self.dec2(
            dec2_input, nm_down1, tm_down1
        )

        # 解码器1
        dec2_up = self.up1(dec2_output)
        dec1_input = torch.cat([dec2_up, enc1_denoise], dim=1)
        dec1_output = self.dec1(
            dec1_input, nm_original, tm_original
        )

        #--------------------------- 新增：中间监督输出 ---------------------------#

        main_output = self.final(dec1_output)

        # sharpness recovery
        if self.use_sharpness_recovery:
            main_output = self.sharpness_recovery(main_output, nm_original, tm_original)

        return main_output