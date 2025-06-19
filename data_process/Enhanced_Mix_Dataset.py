"""
Enhanced Mix Dataset
扩展PMN的Mix_Dataset，集成LLD暗帧数据和raw_image_denoising的噪声合成方法
"""

import os
import numpy as np
import torch
from data_process.real_datasets import Mix_Dataset
from data_process.process import dataload
from utils.basic_utils import log
import scipy.io as sio


class Enhanced_Mix_Dataset(Mix_Dataset):
    """
    增强版Mix_Dataset
    集成了LLD暗帧数据和raw_image_denoising的简化噪声合成方法
    """
    
    def __init__(self, args=None):
        super().__init__(args)
        
        # 初始化增强功能
        self.setup_enhanced_features()
        log(f'Enhanced_Mix_Dataset initialized with {len(self.lld_dark_frames)} LLD ISO levels')
    
    def setup_enhanced_features(self):
        """
        设置增强功能
        """
        # LLD暗帧配置
        self.use_lld_dark_frames = self.args.get('use_lld_dark_frames', True)
        self.lld_dark_frame_path = self.args.get('lld_dark_frame_path', '/data/LLD_calibration')
        
        # 简化噪声合成配置
        self.use_simplified_noise_synthesis = self.args.get('use_simplified_noise_synthesis', True)
        self.quantum_efficiency = self.args.get('quantum_efficiency', 0.4)
        
        # 初始化LLD暗帧数据
        self.lld_dark_frames = {}
        if self.use_lld_dark_frames:
            self.load_lld_dark_frames()
    
    def load_lld_dark_frames(self):
        """
        加载LLD数据集的暗帧数据
        LLD包含~400 dark frames per ISO, 24个ISO级别
        """
        if not os.path.exists(self.lld_dark_frame_path):
            log(f"警告: LLD暗帧路径不存在: {self.lld_dark_frame_path}")
            return
        
        # 扫描LLD数据结构
        lld_bias_paths = [
            os.path.join(self.lld_dark_frame_path, 'bias'),
            os.path.join(self.lld_dark_frame_path, 'dark_frames'),
            os.path.join(self.lld_dark_frame_path, 'SonyA7S2', 'bias'),
            os.path.join(self.lld_dark_frame_path, 'NikonD850', 'bias')
        ]
        
        for bias_path in lld_bias_paths:
            if os.path.exists(bias_path):
                self._scan_bias_directory(bias_path)
        
        log(f"LLD暗帧数据加载完成，支持的ISO: {sorted(self.lld_dark_frames.keys())}")
    
    def _scan_bias_directory(self, bias_path):
        """
        扫描暗帧目录
        """
        try:
            for item in os.listdir(bias_path):
                item_path = os.path.join(bias_path, item)
                
                if os.path.isdir(item_path) and item.isdigit():
                    # ISO目录
                    iso = int(item)
                    self._load_iso_dark_frames(iso, item_path)
                elif item.endswith('.mat'):
                    # 直接的.mat文件，尝试从文件名或内容提取ISO
                    iso = self._extract_iso_from_mat(item_path)
                    if iso is not None:
                        if iso not in self.lld_dark_frames:
                            self.lld_dark_frames[iso] = []
                        self.lld_dark_frames[iso].append(item_path)
        except Exception as e:
            log(f"扫描暗帧目录 {bias_path} 时出错: {e}")
    
    def _load_iso_dark_frames(self, iso, iso_path):
        """
        加载特定ISO的所有暗帧
        """
        if iso not in self.lld_dark_frames:
            self.lld_dark_frames[iso] = []
        
        try:
            for file_name in os.listdir(iso_path):
                if file_name.endswith('.mat'):
                    file_path = os.path.join(iso_path, file_name)
                    self.lld_dark_frames[iso].append(file_path)
            
            log(f"为ISO {iso} 加载了 {len(self.lld_dark_frames[iso])} 个暗帧文件")
        except Exception as e:
            log(f"加载ISO {iso} 暗帧时出错: {e}")
    
    def _extract_iso_from_mat(self, mat_path):
        """
        从.mat文件中提取ISO信息
        """
        try:
            # 首先尝试从文件名提取
            file_name = os.path.basename(mat_path)
            if 'iso' in file_name.lower():
                # 查找iso后面的数字
                import re
                match = re.search(r'iso[\s_-]*(\d+)', file_name.lower())
                if match:
                    return int(match.group(1))
            
            # 尝试从.mat文件内容提取
            mat_data = sio.loadmat(mat_path)
            if 'ISO' in mat_data:
                iso_value = mat_data['ISO']
                if isinstance(iso_value, np.ndarray):
                    return int(iso_value.item())
                else:
                    return int(iso_value)
        except Exception as e:
            log(f"从{mat_path}提取ISO信息失败: {e}")
        
        return None
    
    def get_lld_dark_frame(self, iso, image_shape):
        """
        获取LLD暗帧数据用于噪声合成
        
        Args:
            iso: 目标ISO值
            image_shape: 目标图像尺寸 (H, W)
        
        Returns:
            dark_frame: 暗帧数据，如果不可用则返回None
        """
        if not self.use_lld_dark_frames or iso not in self.lld_dark_frames:
            return None
        
        # 随机选择一个暗帧文件
        dark_frame_files = self.lld_dark_frames[iso]
        if not dark_frame_files:
            return None
        
        selected_file = np.random.choice(dark_frame_files)
        
        try:
            # 使用项目现有的dataload函数
            dark_frame = dataload(selected_file)
            
            # 调整尺寸以匹配目标图像
            if dark_frame.shape != image_shape:
                dark_frame = self._resize_dark_frame(dark_frame, image_shape)
            
            return dark_frame
        except Exception as e:
            log(f"加载暗帧文件 {selected_file} 失败: {e}")
            return None
    
    def _resize_dark_frame(self, dark_frame, target_shape):
        """
        调整暗帧尺寸
        """
        h, w = target_shape
        fh, fw = dark_frame.shape
        
        if fh >= h and fw >= w:
            # 中心裁剪
            start_h = (fh - h) // 2
            start_w = (fw - w) // 2
            return dark_frame[start_h:start_h+h, start_w:start_w+w]
        else:
            # 填充到目标尺寸
            padded = np.zeros((h, w), dtype=dark_frame.dtype)
            start_h = max(0, (h - fh) // 2)
            start_w = max(0, (w - fw) // 2)
            end_h = min(h, start_h + fh)
            end_w = min(w, start_w + fw)
            
            padded[start_h:end_h, start_w:end_w] = dark_frame[:end_h-start_h, :end_w-start_w]
            return padded
    
    def record_bias_frames(self):
        """
        重写父类方法，集成LLD暗帧数据
        """
        # 首先调用父类方法，记录原有的暗帧
        super().record_bias_frames()
        
        # 然后添加LLD暗帧数据
        self.merge_lld_dark_frames()
    
    def merge_lld_dark_frames(self):
        """
        将LLD暗帧数据合并到现有的blacks结构中
        """
        if not self.use_lld_dark_frames:
            return
        
        for iso, dark_frame_files in self.lld_dark_frames.items():
            # 找到对应的ISO索引
            if iso in self.legalISO:
                iso_index = self.legalISO.index(iso)
                
                # 如果当前ISO没有暗帧，或者LLD暗帧更多，则使用LLD暗帧
                if (iso_index >= len(self.blacks) or 
                    len(self.blacks[iso_index]) == 0 or 
                    len(dark_frame_files) > len(self.blacks[iso_index])):
                    
                    # 确保blacks数组足够大
                    while len(self.blacks) <= iso_index:
                        self.blacks.append([])
                    
                    # 添加LLD暗帧文件
                    self.blacks[iso_index].extend(dark_frame_files)
                    log(f"为ISO {iso} 添加了 {len(dark_frame_files)} 个LLD暗帧")
    
    def enhanced_noise_synthesis(self, clean_image, iso, ratio=1.0, aug_wb=None):
        """
        增强的噪声合成方法
        集成raw_image_denoising的简化噪声合成管道
        
        Args:
            clean_image: 清洁图像 (numpy array)
            iso: ISO值
            ratio: 数字增益
            aug_wb: 白平衡增强参数
        
        Returns:
            noisy_image: 合成的带噪图像
        """
        if not self.use_simplified_noise_synthesis:
            # 回退到PMN的SNA方法
            from data_process.process import SNA_torch
            if aug_wb is None:
                aug_wb = np.array([0, 0, 0, 0])
            dn, dy, p = SNA_torch(
                clean_image, aug_wb, iso=iso, ratio=ratio,
                black_lr=True, camera_type=self.args.get('camera_type', 'SonyA7S2')
            )
            return dn
        
        # 使用简化的噪声合成管道
        # 1. 假设化系统增益
        system_gain = self.hypothesize_system_gain(iso)
        
        # 2. 合成光子散粒噪声
        photon_noise = self.synthesize_photon_noise(clean_image, system_gain, ratio)
        
        # 3. 获取信号无关噪声（通过LLD暗帧采样）
        signal_independent_noise = self.get_signal_independent_noise(clean_image.shape, iso)
        
        # 4. 合成最终噪声图像
        noisy_image = clean_image * ratio + photon_noise + signal_independent_noise
        
        return noisy_image
    
    def hypothesize_system_gain(self, iso):
        """
        假设化系统增益计算
        K = ISO/100 * quantum_efficiency
        """
        return iso / 100.0 * self.quantum_efficiency
    
    def synthesize_photon_noise(self, clean_image, system_gain, ratio):
        """
        合成光子散粒噪声
        """
        # 光子噪声方差 = signal * system_gain * ratio
        noise_variance = np.maximum(clean_image * system_gain * ratio, 1e-6)
        photon_noise = np.random.normal(0, np.sqrt(noise_variance), clean_image.shape)
        return photon_noise.astype(clean_image.dtype)
    
    def get_signal_independent_noise(self, image_shape, iso):
        """
        获取信号无关噪声
        优先使用LLD暗帧，备选统计模型
        """
        # 尝试从LLD暗帧获取噪声
        if len(image_shape) == 3:
            # 对于多通道图像，使用单通道暗帧
            dark_frame = self.get_lld_dark_frame(iso, image_shape[1:])
            if dark_frame is not None:
                # 复制到所有通道
                noise = np.tile(dark_frame[np.newaxis, :, :], (image_shape[0], 1, 1))
                return noise.astype(np.float32)
        else:
            # 单通道图像
            dark_frame = self.get_lld_dark_frame(iso, image_shape)
            if dark_frame is not None:
                return dark_frame.astype(np.float32)
        
        # 备选方案：统计噪声模型
        noise_std = np.sqrt(iso / 100.0) * 0.1  # 简化的ISO依赖模型
        signal_independent_noise = np.random.normal(0, noise_std, image_shape)
        return signal_independent_noise.astype(np.float32)
    
    def __getitem__(self, idx):
        """
        重写数据获取方法，集成增强的噪声合成
        """
        # 首先调用父类方法获取基础数据
        data = super().__getitem__(idx)
        
        # 如果启用了增强噪声合成，则进行额外处理
        if (self.use_simplified_noise_synthesis and 
            self.args.get('mode') == 'train' and 
            'enhanced_synthesis' in self.args.get('command', '')):
            
            # 获取清洁图像和相关参数
            hr_image = data['hr']
            iso = data['ISO']
            
            # 应用增强的噪声合成
            if isinstance(hr_image, np.ndarray) and hr_image.size > 0:
                enhanced_lr = self.enhanced_noise_synthesis(
                    hr_image.copy(), 
                    iso=iso, 
                    ratio=data.get('ratio', 1.0)
                )
                data['lr_enhanced'] = enhanced_lr
        
        return data


# 辅助函数：检查LLD数据集可用性
def check_lld_availability(lld_path):
    """
    检查LLD数据集的可用性
    
    Args:
        lld_path: LLD数据集路径
    
    Returns:
        dict: 包含可用性信息的字典
    """
    availability_info = {
        'available': False,
        'iso_count': 0,
        'total_dark_frames': 0,
        'supported_isos': [],
        'errors': []
    }
    
    if not os.path.exists(lld_path):
        availability_info['errors'].append(f"LLD路径不存在: {lld_path}")
        return availability_info
    
    try:
        # 扫描可能的暗帧目录
        bias_paths = [
            os.path.join(lld_path, 'bias'),
            os.path.join(lld_path, 'dark_frames'),
            os.path.join(lld_path, 'SonyA7S2', 'bias'),
            os.path.join(lld_path, 'NikonD850', 'bias')
        ]
        
        total_frames = 0
        supported_isos = []
        
        for bias_path in bias_paths:
            if os.path.exists(bias_path):
                for item in os.listdir(bias_path):
                    item_path = os.path.join(bias_path, item)
                    
                    if os.path.isdir(item_path) and item.isdigit():
                        iso = int(item)
                        mat_files = [f for f in os.listdir(item_path) if f.endswith('.mat')]
                        if mat_files:
                            supported_isos.append(iso)
                            total_frames += len(mat_files)
        
        availability_info['available'] = len(supported_isos) > 0
        availability_info['iso_count'] = len(supported_isos)
        availability_info['total_dark_frames'] = total_frames
        availability_info['supported_isos'] = sorted(supported_isos)
        
    except Exception as e:
        availability_info['errors'].append(f"扫描LLD数据时出错: {e}")
    
    return availability_info


if __name__ == '__main__':
    # 测试Enhanced_Mix_Dataset
    test_args = {
        'use_lld_dark_frames': True,
        'lld_dark_frame_path': '/data/LLD_calibration',
        'use_simplified_noise_synthesis': True,
        'quantum_efficiency': 0.4,
        'mode': 'train',
        'command': 'enhanced_synthesis',
        'camera_type': 'SonyA7S2'
    }
    
    # 检查LLD可用性
    lld_info = check_lld_availability(test_args['lld_dark_frame_path'])
    print("LLD可用性检查结果:")
    for key, value in lld_info.items():
        print(f"  {key}: {value}")
    
    if lld_info['available']:
        print(f"\n✓ LLD数据集可用，支持 {lld_info['iso_count']} 个ISO级别")
        print(f"  总暗帧数: {lld_info['total_dark_frames']}")
        print(f"  支持的ISO: {lld_info['supported_isos']}")
    else:
        print(f"\n✗ LLD数据集不可用")
        for error in lld_info['errors']:
            print(f"  错误: {error}")