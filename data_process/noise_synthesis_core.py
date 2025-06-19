"""
Raw Image Denoising 核心噪声合成模块
实现论文提出的简化噪声合成管道，包括：
1. 假设化系统增益计算（避免复杂标定）
2. 暗帧直接采样方法（避免统计建模）  
3. 简化的信号无关噪声合成
"""

import numpy as np
import torch
import os
import scipy.io as sio
from typing import Optional, Tuple, Dict, List
from utils.basic_utils import log
from data_process.process import dataload


class HypothesizedSystemGain:
    """
    假设化系统增益计算器
    实现论文中的K = ISO/100 * quantum_efficiency方法
    避免复杂的光子传递法标定过程
    """
    
    def __init__(self, quantum_efficiency: float = 0.4, base_iso: int = 100):
        """
        初始化假设化系统增益计算器
        
        Args:
            quantum_efficiency: 假设的量子效率 (论文中使用0.4，即40%)
            base_iso: 基础ISO值 (通常是100或400)
        """
        self.quantum_efficiency = quantum_efficiency
        self.base_iso = base_iso
        
        log(f"假设化系统增益初始化: QE={quantum_efficiency}, base_ISO={base_iso}")
    
    def calculate_system_gain(self, iso: float) -> float:
        """
        计算假设化系统增益
        
        论文公式: K = ISO/100 * quantum_efficiency
        这避免了传统方法需要的平场校正和光子传递法标定
        
        Args:
            iso: 目标ISO值
            
        Returns:
            system_gain: 假设化系统增益
        """
        system_gain = (iso / 100.0) * self.quantum_efficiency
        return system_gain
    
    def get_gain_range(self, iso: float, variance_factor: float = 0.1) -> Tuple[float, float]:
        """
        获取系统增益的变化范围
        论文表明去噪网络对系统增益的变化具有鲁棒性
        
        Args:
            iso: 目标ISO
            variance_factor: 变化因子
            
        Returns:
            (min_gain, max_gain): 增益范围
        """
        nominal_gain = self.calculate_system_gain(iso)
        variance = nominal_gain * variance_factor
        return (nominal_gain - variance, nominal_gain + variance)
    
    def sample_random_gain(self, iso: float, variance_factor: float = 0.1) -> float:
        """
        在合理范围内随机采样系统增益
        用于训练时的数据增强
        """
        min_gain, max_gain = self.get_gain_range(iso, variance_factor)
        return np.random.uniform(min_gain, max_gain)


class DirectDarkFrameSampler:
    """
    直接暗帧采样器
    实现论文中的核心创新：直接从传感器采集的暗帧中采样噪声
    避免复杂的统计建模和参数拟合过程
    """
    
    def __init__(self, dark_frame_paths: Dict[int, List[str]], 
                 cache_frames: bool = True, 
                 max_cache_size: int = 100):
        """
        初始化暗帧采样器
        
        Args:
            dark_frame_paths: {ISO: [暗帧文件路径]} 字典
            cache_frames: 是否缓存暗帧数据
            max_cache_size: 最大缓存数量
        """
        self.dark_frame_paths = dark_frame_paths
        self.cache_frames = cache_frames
        self.max_cache_size = max_cache_size
        
        # 暗帧数据缓存
        self.cached_frames = {}
        self.cache_usage_count = {}
        
        # 统计信息
        self.total_dark_frames = sum(len(paths) for paths in dark_frame_paths.values())
        self.available_isos = sorted(dark_frame_paths.keys())
        
        log(f"暗帧采样器初始化: {len(self.available_isos)} 个ISO级别, "
            f"总计 {self.total_dark_frames} 个暗帧")
    
    def is_available(self, iso: int) -> bool:
        """检查指定ISO的暗帧是否可用"""
        return iso in self.dark_frame_paths and len(self.dark_frame_paths[iso]) > 0
    
    def get_nearest_iso(self, target_iso: int) -> Optional[int]:
        """
        获取最接近目标ISO的可用ISO
        当目标ISO没有暗帧时使用
        """
        if not self.available_isos:
            return None
        
        # 找到最接近的ISO
        distances = [abs(iso - target_iso) for iso in self.available_isos]
        min_idx = np.argmin(distances)
        return self.available_isos[min_idx]
    
    def load_dark_frame(self, file_path: str) -> Optional[np.ndarray]:
        """
        加载单个暗帧文件
        支持.mat和其他PMN支持的格式
        """
        try:
            # 使用PMN的dataload函数，支持多种格式
            dark_frame = dataload(file_path)
            
            # 确保是numpy数组
            if torch.is_tensor(dark_frame):
                dark_frame = dark_frame.cpu().numpy()
            
            return dark_frame.astype(np.float32)
            
        except Exception as e:
            log(f"警告: 无法加载暗帧文件 {file_path}: {e}")
            return None
    
    def sample_dark_frame(self, iso: int, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        从指定ISO的暗帧中采样
        
        Args:
            iso: 目标ISO值
            image_shape: 目标图像尺寸 (H, W)
            
        Returns:
            dark_frame: 采样的暗帧数据，尺寸匹配image_shape
        """
        # 检查是否有可用的暗帧
        if not self.is_available(iso):
            # 尝试使用最接近的ISO
            nearest_iso = self.get_nearest_iso(iso)
            if nearest_iso is None:
                return None
            log(f"ISO {iso} 暗帧不可用，使用最接近的ISO {nearest_iso}")
            iso = nearest_iso
        
        # 随机选择一个暗帧文件
        frame_paths = self.dark_frame_paths[iso]
        selected_path = np.random.choice(frame_paths)
        
        # 检查缓存
        cache_key = f"{iso}_{selected_path}"
        if self.cache_frames and cache_key in self.cached_frames:
            self.cache_usage_count[cache_key] += 1
            dark_frame = self.cached_frames[cache_key]
        else:
            # 加载暗帧
            dark_frame = self.load_dark_frame(selected_path)
            if dark_frame is None:
                return None
            
            # 更新缓存
            if self.cache_frames:
                self._update_cache(cache_key, dark_frame)
        
        # 调整尺寸以匹配目标图像
        resized_frame = self._resize_frame(dark_frame, image_shape)
        return resized_frame
    
    def _update_cache(self, cache_key: str, dark_frame: np.ndarray):
        """更新暗帧缓存"""
        # 如果缓存已满，移除使用次数最少的项
        if len(self.cached_frames) >= self.max_cache_size:
            least_used_key = min(self.cache_usage_count.keys(), 
                                key=lambda k: self.cache_usage_count[k])
            del self.cached_frames[least_used_key]
            del self.cache_usage_count[least_used_key]
        
        self.cached_frames[cache_key] = dark_frame.copy()
        self.cache_usage_count[cache_key] = 1
    
    def _resize_frame(self, frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        调整暗帧尺寸以匹配目标图像
        使用中心裁剪或边缘填充
        """
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
            # 填充到目标尺寸
            padded = np.zeros((h, w), dtype=frame.dtype)
            
            # 计算放置位置（居中）
            start_h = max(0, (h - fh) // 2)
            start_w = max(0, (w - fw) // 2)
            end_h = min(h, start_h + fh)
            end_w = min(w, start_w + fw)
            
            # 确保不越界
            src_h = min(fh, end_h - start_h)
            src_w = min(fw, end_w - start_w)
            
            padded[start_h:start_h+src_h, start_w:start_w+src_w] = frame[:src_h, :src_w]
            return padded
    
    def get_statistics(self) -> Dict:
        """获取采样器统计信息"""
        return {
            'available_isos': self.available_isos,
            'total_dark_frames': self.total_dark_frames,
            'cached_frames': len(self.cached_frames),
            'cache_usage': dict(self.cache_usage_count)
        }


class SimplifiedNoiseModel:
    """
    简化噪声模型
    实现论文的核心噪声合成管道，避免复杂的统计建模
    """
    
    def __init__(self, 
                 system_gain_calculator: HypothesizedSystemGain,
                 dark_frame_sampler: DirectDarkFrameSampler,
                 fallback_noise_std: float = 0.1):
        """
        初始化简化噪声模型
        
        Args:
            system_gain_calculator: 假设化系统增益计算器
            dark_frame_sampler: 暗帧采样器
            fallback_noise_std: 当暗帧不可用时的备选噪声标准差
        """
        self.gain_calc = system_gain_calculator
        self.dark_sampler = dark_frame_sampler
        self.fallback_noise_std = fallback_noise_std
        
        log("简化噪声模型初始化完成")
    
    def synthesize_noise(self, 
                        clean_image: np.ndarray, 
                        iso: int, 
                        ratio: float = 1.0,
                        use_random_gain: bool = False) -> np.ndarray:
        """
        合成噪声图像
        实现论文的简化噪声合成管道
        
        Args:
            clean_image: 清洁图像
            iso: ISO值
            ratio: 数字增益/曝光比
            use_random_gain: 是否使用随机系统增益（用于数据增强）
            
        Returns:
            noisy_image: 合成的带噪图像
        """
        # 1. 计算系统增益
        if use_random_gain:
            system_gain = self.gain_calc.sample_random_gain(iso)
        else:
            system_gain = self.gain_calc.calculate_system_gain(iso)
        
        # 2. 合成信号相关噪声（光子散粒噪声）
        signal_dependent_noise = self._synthesize_photon_noise(
            clean_image, system_gain, ratio
        )
        
        # 3. 获取信号无关噪声（通过暗帧直接采样）
        signal_independent_noise = self._get_signal_independent_noise(
            clean_image.shape, iso
        )
        
        # 4. 合成最终噪声图像
        # 遵循传感器噪声模型: noisy = (clean * ratio + photon_noise) + read_noise
        noisy_image = clean_image * ratio + signal_dependent_noise + signal_independent_noise
        
        return noisy_image.astype(clean_image.dtype)
    
    def _synthesize_photon_noise(self, 
                                clean_image: np.ndarray, 
                                system_gain: float, 
                                ratio: float) -> np.ndarray:
        """
        合成光子散粒噪声
        遵循泊松分布特性：方差等于均值
        """
        # 光子噪声的方差 = signal * system_gain * ratio
        noise_variance = np.maximum(clean_image * system_gain * ratio, 1e-6)
        
        # 生成高斯近似的光子噪声（大信号下泊松分布可用高斯近似）
        photon_noise = np.random.normal(0, np.sqrt(noise_variance), clean_image.shape)
        
        return photon_noise.astype(clean_image.dtype)
    
    def _get_signal_independent_noise(self, 
                                    image_shape: Tuple, 
                                    iso: int) -> np.ndarray:
        """
        获取信号无关噪声
        优先使用暗帧直接采样，备选统计模型
        """
        # 处理多通道图像
        if len(image_shape) == 3:
            # 对于Bayer格式的4通道图像，每个通道使用相同的暗帧模式
            single_channel_shape = image_shape[1:]
            dark_frame = self.dark_sampler.sample_dark_frame(iso, single_channel_shape)
            
            if dark_frame is not None:
                # 复制到所有通道
                noise = np.tile(dark_frame[np.newaxis, :, :], (image_shape[0], 1, 1))
                return noise.astype(np.float32)
        else:
            # 单通道图像
            dark_frame = self.dark_sampler.sample_dark_frame(iso, image_shape)
            if dark_frame is not None:
                return dark_frame.astype(np.float32)
        
        # 备选方案：使用简化的统计噪声模型
        log(f"ISO {iso} 暗帧不可用，使用统计噪声模型")
        return self._fallback_statistical_noise(image_shape, iso)
    
    def _fallback_statistical_noise(self, 
                                   image_shape: Tuple, 
                                   iso: int) -> np.ndarray:
        """
        备选的统计噪声模型
        当暗帧不可用时使用
        """
        # 简化的ISO依赖噪声模型
        noise_std = self.fallback_noise_std * np.sqrt(iso / 100.0)
        
        # 生成高斯白噪声
        statistical_noise = np.random.normal(0, noise_std, image_shape)
        
        return statistical_noise.astype(np.float32)
    
    def get_model_info(self) -> Dict:
        """获取噪声模型信息"""
        return {
            'quantum_efficiency': self.gain_calc.quantum_efficiency,
            'available_isos': self.dark_sampler.available_isos,
            'total_dark_frames': self.dark_sampler.total_dark_frames,
            'fallback_noise_std': self.fallback_noise_std
        }


# 高级接口函数
def create_simplified_noise_synthesizer(lld_dark_frame_path: str,
                                       quantum_efficiency: float = 0.4,
                                       cache_dark_frames: bool = True) -> SimplifiedNoiseModel:
    """
    创建简化噪声合成器的便捷函数
    
    Args:
        lld_dark_frame_path: LLD暗帧数据路径
        quantum_efficiency: 假设量子效率
        cache_dark_frames: 是否缓存暗帧
        
    Returns:
        SimplifiedNoiseModel实例
    """
    # 扫描LLD暗帧数据
    dark_frame_paths = scan_lld_dark_frames(lld_dark_frame_path)
    
    if not dark_frame_paths:
        log("警告: 未找到LLD暗帧数据，将使用统计噪声模型")
        dark_frame_paths = {}
    
    # 初始化组件
    gain_calc = HypothesizedSystemGain(quantum_efficiency)
    dark_sampler = DirectDarkFrameSampler(dark_frame_paths, cache_dark_frames)
    
    # 创建噪声模型
    noise_model = SimplifiedNoiseModel(gain_calc, dark_sampler)
    
    log(f"简化噪声合成器创建完成，支持 {len(dark_frame_paths)} 个ISO级别")
    return noise_model


def scan_lld_dark_frames(lld_path: str) -> Dict[int, List[str]]:
    """
    扫描LLD数据集中的暗帧文件
    
    Args:
        lld_path: LLD数据集根路径
        
    Returns:
        {ISO: [暗帧文件路径]} 字典
    """
    dark_frame_paths = {}
    
    if not os.path.exists(lld_path):
        log(f"LLD路径不存在: {lld_path}")
        return dark_frame_paths
    
    # LLD数据集可能的暗帧目录
    potential_bias_dirs = [
        'bias',
        'dark_frames',
        'calibration',
        'SonyA7S2/bias',
        'NikonD850/bias',
        'LLD_calibration/bias'
    ]
    
    for bias_subdir in potential_bias_dirs:
        bias_dir = os.path.join(lld_path, bias_subdir)
        
        if os.path.exists(bias_dir):
            try:
                for item in os.listdir(bias_dir):
                    item_path = os.path.join(bias_dir, item)
                    
                    if os.path.isdir(item_path) and item.isdigit():
                        # ISO目录
                        iso = int(item)
                        iso_dark_frames = []
                        
                        for file_name in os.listdir(item_path):
                            if file_name.endswith(('.mat', '.npy', '.raw')):
                                file_path = os.path.join(item_path, file_name)
                                iso_dark_frames.append(file_path)
                        
                        if iso_dark_frames:
                            dark_frame_paths[iso] = iso_dark_frames
                            log(f"发现ISO {iso}: {len(iso_dark_frames)} 个暗帧")
                    
                    elif item.endswith('.mat'):
                        # 直接的.mat文件，尝试提取ISO信息
                        iso = extract_iso_from_filename(item)
                        if iso is not None:
                            if iso not in dark_frame_paths:
                                dark_frame_paths[iso] = []
                            dark_frame_paths[iso].append(item_path)
                            
            except Exception as e:
                log(f"扫描目录 {bias_dir} 时出错: {e}")
    
    return dark_frame_paths


def extract_iso_from_filename(filename: str) -> Optional[int]:
    """
    从文件名中提取ISO信息
    """
    import re
    
    # 常见的ISO标识模式
    patterns = [
        r'iso[\s_-]*(\d+)',
        r'(\d+)iso',
        r'ISO[\s_-]*(\d+)',
        r'(\d+)ISO'
    ]
    
    filename_lower = filename.lower()
    
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            return int(match.group(1))
    
    return None


if __name__ == '__main__':
    # 测试简化噪声合成器
    lld_path = "/data/LLD_calibration"
    
    # 创建噪声合成器
    noise_synthesizer = create_simplified_noise_synthesizer(
        lld_dark_frame_path=lld_path,
        quantum_efficiency=0.4,
        cache_dark_frames=True
    )
    
    # 打印模型信息
    model_info = noise_synthesizer.get_model_info()
    print("噪声模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 测试噪声合成
    test_image = np.random.rand(4, 512, 512).astype(np.float32) * 1000  # 模拟RAW图像
    test_iso = 3200
    test_ratio = 4.0
    
    print(f"\n测试噪声合成:")
    print(f"  输入图像尺寸: {test_image.shape}")
    print(f"  ISO: {test_iso}")
    print(f"  数字增益: {test_ratio}")
    
    noisy_image = noise_synthesizer.synthesize_noise(
        test_image, test_iso, test_ratio, use_random_gain=False
    )
    
    print(f"  输出图像尺寸: {noisy_image.shape}")
    print(f"  输入信号范围: [{test_image.min():.2f}, {test_image.max():.2f}]")
    print(f"  输出信号范围: [{noisy_image.min():.2f}, {noisy_image.max():.2f}]")
    print(f"  噪声功率: {np.var(noisy_image - test_image * test_ratio):.4f}")