"""
PyTorch兼容的简化噪声合成模块
集成到PMN的GPU预处理管道中，实现高效的批量噪声合成
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from utils import *
from data_process.process import SNA_torch  # PMN原有的SNA方法


class TorchSimplifiedNoiseSynthesis:
    """
    PyTorch实现的简化噪声合成
    与PMN的GPU预处理管道兼容，支持批量处理
    """
    
    def __init__(self, 
                 quantum_efficiency: float = 0.4,
                 fallback_noise_std: float = 0.1,
                 device: torch.device = torch.device('cuda')):
        """
        初始化PyTorch噪声合成器
        
        Args:
            quantum_efficiency: 假设量子效率
            fallback_noise_std: 备选噪声标准差
            device: 计算设备
        """
        self.quantum_efficiency = quantum_efficiency
        self.fallback_noise_std = fallback_noise_std
        self.device = device
        
        # 暗帧缓存（GPU内存中）
        self.dark_frame_cache = {}
        self.max_cache_size = 50  # 限制GPU内存使用
        
        # log(f"PyTorch简化噪声合成器初始化完成，设备: {device}")
    
    def hypothesize_system_gain(self, iso: torch.Tensor) -> torch.Tensor:
        """
        批量计算假设化系统增益
        
        Args:
            iso: ISO值张量 (batch_size,) 或标量
            
        Returns:
            system_gain: 系统增益张量
        """
        # 确保iso是tensor
        if not torch.is_tensor(iso):
            iso = torch.tensor(iso, dtype=torch.float32, device=self.device)
        
        # K = ISO/100 * quantum_efficiency
        system_gain = (iso / 100.0) * self.quantum_efficiency
        return system_gain
    
    def sample_random_gain(self, iso: torch.Tensor, variance_factor: float = 0.1) -> torch.Tensor:
        """
        在合理范围内随机采样系统增益（用于数据增强）
        """
        nominal_gain = self.hypothesize_system_gain(iso)
        variance = nominal_gain * variance_factor
        
        # 生成均匀分布的随机增益
        random_factor = torch.rand_like(nominal_gain, device=self.device) * 2 - 1  # [-1, 1]
        random_gain = nominal_gain + random_factor * variance
        
        return torch.clamp(random_gain, min=0.01)  # 确保增益为正
    
    def synthesize_photon_noise(self, 
                               clean_image: torch.Tensor, 
                               system_gain: torch.Tensor, 
                               ratio: torch.Tensor) -> torch.Tensor:
        """
        批量合成光子散粒噪声
        
        Args:
            clean_image: 清洁图像 (batch_size, channels, height, width)
            system_gain: 系统增益 (batch_size,) 或 (batch_size, 1, 1, 1)
            ratio: 数字增益 (batch_size,) 或 (batch_size, 1, 1, 1)
            
        Returns:
            photon_noise: 光子噪声
        """
        # 确保维度匹配
        if system_gain.dim() == 1:
            system_gain = system_gain.view(-1, 1, 1, 1)
        if ratio.dim() == 1:
            ratio = ratio.view(-1, 1, 1, 1)
        
        # 光子噪声方差 = signal * system_gain * ratio
        noise_variance = torch.clamp(clean_image * system_gain * ratio, min=1e-6)
        
        # 生成光子散粒噪声
        photon_noise = torch.randn_like(clean_image, device=self.device) * torch.sqrt(noise_variance)
        
        return photon_noise
    
    def load_dark_frame_tensor(self, dark_frame_path: str, target_shape: Tuple[int, int]) -> Optional[torch.Tensor]:
        """
        加载暗帧并转换为PyTorch张量（添加归一化处理）
        """
        try:
            from data_process.process import dataload, raw2bayer
            dark_frame = dataload(dark_frame_path)
            
            # 转换为numpy
            if torch.is_tensor(dark_frame):
                dark_frame = dark_frame.cpu().numpy()
            
            # ⭐ 关键修复：对暗帧进行归一化处理
            # 使用与正常图像相同的归一化参数
            if dark_frame.ndim == 2:  # 如果是2D RAW图像
                # 应用与PMN相同的归一化
                wp, bl = 16383, 512  # SonyA7S2参数，从配置读取更好
                dark_frame_4c = raw2bayer(dark_frame, wp=wp, bl=bl, norm=True, clip=False)
                # 取均值作为单通道暗帧噪声
                dark_frame = np.mean(dark_frame_4c, axis=0)
            elif dark_frame.ndim == 3 and dark_frame.shape[0] == 4:  # 如果已经是4通道
                # 如果数值范围还是RAW域，需要归一化
                if dark_frame.max() > 10:
                    wp, bl = 16383, 512
                    dark_frame = (dark_frame - bl) / (wp - bl)
                    dark_frame = np.clip(dark_frame, 0, 1)
                # 取均值作为单通道
                dark_frame = np.mean(dark_frame, axis=0)
            
            # 调整尺寸
            dark_frame = self._resize_frame_numpy(dark_frame, target_shape)
            
            # 转换为torch tensor并移动到GPU
            dark_frame_tensor = torch.from_numpy(dark_frame).float().to(self.device)
            
            return dark_frame_tensor
            
        except Exception as e:
            log(f"加载暗帧失败 {dark_frame_path}: {e}")
            return None
    
    def _resize_frame_numpy(self, frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """调整暗帧尺寸（numpy版本）"""
        h, w = target_shape
        fh, fw = frame.shape
        
        if fh == h and fw == w:
            return frame
        
        if fh >= h and fw >= w:
            # 中心裁剪
            start_h = (fh - h) // 2
            start_w = (fw - w) // 2
            return frame[start_h:start_h+h, start_w:start_w+w]
        else:
            # 填充
            padded = np.zeros((h, w), dtype=frame.dtype)
            start_h = max(0, (h - fh) // 2)
            start_w = max(0, (w - fw) // 2)
            end_h = min(h, start_h + fh)
            end_w = min(w, start_w + fw)
            src_h = min(fh, end_h - start_h)
            src_w = min(fw, end_w - start_w)
            padded[start_h:start_h+src_h, start_w:start_w+src_w] = frame[:src_h, :src_w]
            return padded
    
    def get_signal_independent_noise(self, 
                                   image_shape: torch.Size, 
                                   iso_list: List[int],
                                   dark_frame_paths: Dict[int, List[str]]) -> torch.Tensor:
        """
        批量获取信号无关噪声
        
        Args:
            image_shape: 批量图像尺寸 (batch_size, channels, height, width)
            iso_list: 每个样本的ISO值列表
            dark_frame_paths: 暗帧路径字典
            
        Returns:
            signal_independent_noise: 信号无关噪声张量
        """
        batch_size, channels, height, width = image_shape
        noise_batch = torch.zeros(image_shape, device=self.device)
        
        for i, iso in enumerate(iso_list):
            # 尝试使用暗帧
            if iso in dark_frame_paths and dark_frame_paths[iso]:
                # 随机选择暗帧
                selected_path = np.random.choice(dark_frame_paths[iso])
                cache_key = f"{iso}_{height}_{width}"
                
                # 检查缓存
                if cache_key in self.dark_frame_cache:
                    dark_frame = self.dark_frame_cache[cache_key]
                else:
                    dark_frame = self.load_dark_frame_tensor(selected_path, (height, width))
                    if dark_frame is not None and len(self.dark_frame_cache) < self.max_cache_size:
                        self.dark_frame_cache[cache_key] = dark_frame
                
                if dark_frame is not None:
                    # 复制到所有通道
                    for c in range(channels):
                        noise_batch[i, c] = dark_frame
                    continue
            
            # 备选方案：统计噪声
            noise_std = self.fallback_noise_std * np.sqrt(iso / 100.0)
            noise_batch[i] = torch.randn(channels, height, width, device=self.device) * noise_std
        
        return noise_batch
    
    def synthesize_batch_noise(self, 
                             clean_images: torch.Tensor,
                             iso_list: List[int],
                             ratio_list: List[float],
                             dark_frame_paths: Dict[int, List[str]] = None,
                             use_random_gain: bool = True) -> torch.Tensor:
        """
        批量噪声合成（主要接口）
        
        Args:
            clean_images: 清洁图像批次 (batch_size, channels, height, width)
            iso_list: ISO值列表
            ratio_list: 数字增益列表
            dark_frame_paths: 暗帧路径字典
            use_random_gain: 是否使用随机系统增益
            
        Returns:
            noisy_images: 合成的带噪图像批次
        """
        batch_size = clean_images.size(0)
        
        # 转换为tensor
        iso_tensor = torch.tensor(iso_list, dtype=torch.float32, device=self.device)
        ratio_tensor = torch.tensor(ratio_list, dtype=torch.float32, device=self.device)
        
        # 计算系统增益
        if use_random_gain:
            system_gain = self.sample_random_gain(iso_tensor)
        else:
            system_gain = self.hypothesize_system_gain(iso_tensor)
        
        # 合成光子散粒噪声
        photon_noise = self.synthesize_photon_noise(clean_images, system_gain, ratio_tensor)
        
        # 获取信号无关噪声
        if dark_frame_paths:
            signal_independent_noise = self.get_signal_independent_noise(
                clean_images.shape, iso_list, dark_frame_paths
            )
        else:
            # 使用统计噪声模型
            signal_independent_noise = self._fallback_statistical_noise_batch(
                clean_images.shape, iso_tensor
            )
        
        # 合成最终图像
        ratio_expanded = ratio_tensor.view(-1, 1, 1, 1)
        noisy_images = clean_images * ratio_expanded + photon_noise + signal_independent_noise
        
        return noisy_images
    
    def _fallback_statistical_noise_batch(self, 
                                        image_shape: torch.Size, 
                                        iso_tensor: torch.Tensor) -> torch.Tensor:
        """批量生成统计噪声"""
        batch_size, channels, height, width = image_shape
        
        # 计算每个样本的噪声标准差
        noise_std = self.fallback_noise_std * torch.sqrt(iso_tensor / 100.0)
        noise_std = noise_std.view(-1, 1, 1, 1)
        
        # 生成噪声
        statistical_noise = torch.randn(image_shape, device=self.device) * noise_std
        
        return statistical_noise


class EnhancedSNA_torch:
    """
    增强版SNA_torch
    结合PMN的SNA方法和raw_image_denoising的简化噪声合成
    """
    
    def __init__(self, 
                 simplified_synthesizer: TorchSimplifiedNoiseSynthesis,
                 sna_weight: float = 0.5,
                 simplified_weight: float = 0.5):
        """
        初始化增强版SNA
        
        Args:
            simplified_synthesizer: 简化噪声合成器
            sna_weight: PMN SNA方法的权重
            simplified_weight: 简化合成方法的权重
        """
        self.simplified_synthesizer = simplified_synthesizer
        self.sna_weight = sna_weight
        self.simplified_weight = simplified_weight
        
        log(f"增强版SNA初始化: SNA权重={sna_weight}, 简化权重={simplified_weight}")
    
    def enhanced_synthesis(self, 
                          clean_image: torch.Tensor,
                          iso: int,
                          ratio: float,
                          aug_wb: np.ndarray,
                          black_lr: bool = True,
                          camera_type: str = 'SonyA7S2',
                          dark_frame_paths: Dict[int, List[str]] = None,
                          use_method: str = 'hybrid') -> torch.Tensor:
        """
        增强噪声合成方法
        
        Args:
            clean_image: 清洁图像
            iso: ISO值
            ratio: 数字增益
            aug_wb: 白平衡增强参数
            black_lr: 是否使用黑图
            camera_type: 相机类型
            dark_frame_paths: 暗帧路径
            use_method: 使用的方法 ('sna', 'simplified', 'hybrid')
            
        Returns:
            noisy_image: 合成的带噪图像
        """
        if use_method == 'sna':
            # 仅使用PMN的SNA方法
            dn, dy, p = SNA_torch(
                clean_image, aug_wb, iso=iso, ratio=ratio,
                black_lr=black_lr, camera_type=camera_type
            )
            return dn
            
        elif use_method == 'simplified':
            # 仅使用简化噪声合成
            if clean_image.dim() == 3:
                clean_image = clean_image.unsqueeze(0)  # 添加batch维度
            
            noisy_batch = self.simplified_synthesizer.synthesize_batch_noise(
                clean_image, [iso], [ratio], dark_frame_paths, use_random_gain=False
            )
            return noisy_batch.squeeze(0)  # 移除batch维度
            
        elif use_method == 'hybrid':
            # 混合方法：随机选择或加权组合
            if np.random.rand() < self.sna_weight / (self.sna_weight + self.simplified_weight):
                # 使用SNA方法
                dn, dy, p = SNA_torch(
                    clean_image, aug_wb, iso=iso, ratio=ratio,
                    black_lr=black_lr, camera_type=camera_type
                )
                return dn
            else:
                # 使用简化方法
                if clean_image.dim() == 3:
                    clean_image = clean_image.unsqueeze(0)
                
                noisy_batch = self.simplified_synthesizer.synthesize_batch_noise(
                    clean_image, [iso], [ratio], dark_frame_paths, use_random_gain=True
                )
                return noisy_batch.squeeze(0)
        
        else:
            raise ValueError(f"未知的合成方法: {use_method}")


def create_enhanced_sna_torch(quantum_efficiency: float = 0.4,
                             sna_weight: float = 0.5,
                             simplified_weight: float = 0.5,
                             device: torch.device = None) -> EnhancedSNA_torch:
    """
    创建增强版SNA_torch的便捷函数
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简化噪声合成器
    simplified_synthesizer = TorchSimplifiedNoiseSynthesis(
        quantum_efficiency=quantum_efficiency,
        device=device
    )
    
    # 创建增强版SNA
    enhanced_sna = EnhancedSNA_torch(
        simplified_synthesizer=simplified_synthesizer,
        sna_weight=sna_weight,
        simplified_weight=simplified_weight
    )
    
    return enhanced_sna


def integrated_noise_synthesis_torch(clean_image: torch.Tensor,
                                   aug_wb: np.ndarray,
                                   iso: int,
                                   ratio: float,
                                   black_lr: bool = True,
                                   camera_type: str = 'SonyA7S2',
                                   dark_frame_paths: Dict[int, List[str]] = None,
                                   use_simplified: bool = True,
                                   quantum_efficiency: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    集成的噪声合成函数
    可以直接替换PMN中的SNA_torch调用
    
    返回格式与PMN的SNA_torch保持一致：
    - dn: 噪声增量（需要加到原图像上）
    - dy: 清洁图像增量（需要加到原图像上）
    - noise_params: 噪声参数
    """
    if use_simplified and dark_frame_paths and iso in dark_frame_paths:
        # 使用简化噪声合成
        device = clean_image.device
        synthesizer = TorchSimplifiedNoiseSynthesis(
            quantum_efficiency=quantum_efficiency,
            device=device
        )
        
        if clean_image.dim() == 3:
            clean_image_batch = clean_image.unsqueeze(0)
        else:
            clean_image_batch = clean_image
        
        # 合成完整的带噪图像
        noisy_batch = synthesizer.synthesize_batch_noise(
            clean_image_batch, [iso], [ratio], dark_frame_paths, use_random_gain=True
        )
        
        noisy_image = noisy_batch.squeeze(0) if clean_image.dim() == 3 else noisy_batch
        
        # 计算增量以匹配PMN接口
        scaled_clean = clean_image * ratio
        dn = noisy_image - scaled_clean  # 噪声增量
        dy = scaled_clean - clean_image  # 清洁图像的比例增量
        noise_params = torch.tensor([iso, ratio], device=device)
        
        return dn, dy, noise_params
    
    else:
        # 回退到PMN的SNA方法
        return SNA_torch(clean_image, aug_wb, iso=iso, ratio=ratio, 
                        black_lr=black_lr, camera_type=camera_type)


if __name__ == '__main__':
    # 测试PyTorch噪声合成器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 4
    channels = 4
    height = 512
    width = 512
    
    clean_images = torch.rand(batch_size, channels, height, width, device=device) * 1000
    iso_list = [800, 1600, 3200, 6400]
    ratio_list = [2.0, 4.0, 8.0, 16.0]
    
    print(f"测试数据:")
    print(f"  批次尺寸: {clean_images.shape}")
    print(f"  ISO列表: {iso_list}")
    print(f"  增益列表: {ratio_list}")
    
    # 创建噪声合成器
    synthesizer = TorchSimplifiedNoiseSynthesis(
        quantum_efficiency=0.4,
        device=device
    )
    
    # 测试噪声合成
    noisy_images = synthesizer.synthesize_batch_noise(
        clean_images, iso_list, ratio_list, use_random_gain=True
    )
    
    print(f"\n噪声合成结果:")
    print(f"  输出尺寸: {noisy_images.shape}")
    print(f"  输入信号范围: [{clean_images.min():.2f}, {clean_images.max():.2f}]")
    print(f"  输出信号范围: [{noisy_images.min():.2f}, {noisy_images.max():.2f}]")
    
    # 计算噪声功率
    ratio_tensor = torch.tensor(ratio_list, device=device).view(-1, 1, 1, 1)
    noise_power = torch.var(noisy_images - clean_images * ratio_tensor, dim=[1,2,3])
    print(f"  各样本噪声功率: {noise_power.cpu().numpy()}")
    
    # 测试增强版SNA
    enhanced_sna = create_enhanced_sna_torch(device=device)
    
    single_image = clean_images[0]
    aug_wb = np.array([0.1, 0.0, -0.1, 0.0])
    
    enhanced_result = enhanced_sna. enhanced_synthesis(
        single_image, iso_list[0], ratio_list[0], aug_wb, use_method='simplified'
    )
    
    print(f"\n增强版SNA测试:")
    print(f"  输入尺寸: {single_image.shape}")
    print(f"  输出尺寸: {enhanced_result.shape}")
    print(f"  输出范围: [{enhanced_result.min():.2f}, {enhanced_result.max():.2f}]")