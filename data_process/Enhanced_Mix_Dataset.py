"""
Enhanced Mix Dataset
基于PMN的Mix_Dataset，添加LLD暗帧数据和raw_image_denoising的简化噪声合成方法
直接继承Mix_Dataset，避免字段缺失问题
"""

import os
import numpy as np
import torch
from data_process.real_datasets import Mix_Dataset
from data_process.process import dataload
from utils import *
import scipy.io as sio


class Enhanced_Mix_Dataset(Mix_Dataset):
    """
    增强版Mix_Dataset
    在PMN的Mix_Dataset基础上集成LLD暗帧数据和简化噪声合成方法
    """
    
    def __init__(self, args=None):
        # 首先初始化父类Mix_Dataset
        super().__init__(args)
        
        # 然后添加增强功能
        self.setup_enhanced_features()
        log(f'Enhanced_Mix_Dataset initialized with {len(self.lld_dark_frames)} LLD ISO levels')
    
    def setup_enhanced_features(self):
        """
        设置增强功能
        """
        # LLD暗帧配置
        self.use_lld_dark_frames = self.args.get('use_lld_dark_frames', True)
        self.lld_dark_frame_path = self.args.get('lld_dark_frame_path', '/data/LLD')
        
        # 简化噪声合成配置
        self.use_simplified_noise_synthesis = self.args.get('use_simplified_noise_synthesis', True)
        self.quantum_efficiency = self.args.get('quantum_efficiency', 0.4)
        
        # 噪声合成方法选择权重
        self.sna_rate = self.args.get('SNA_rate', 0.5)
        self.enhanced_rate = self.args.get('enhanced_rate', 0.5)
        
        # 初始化LLD暗帧数据
        self.lld_dark_frames = {}
        if self.use_lld_dark_frames:
            self.load_lld_dark_frames()
    
    def load_lld_dark_frames(self):
        """
        加载LLD数据集的暗帧数据
        支持LLD的目录结构：LLD/SonyA7S2/BiasFrame_ET_1_30/ISO/dark_ISO_XXXX.mat
        """
        if not os.path.exists(self.lld_dark_frame_path):
            log(f"警告: LLD暗帧路径不存在: {self.lld_dark_frame_path}")
            return
        
        # 扫描LLD数据结构
        lld_bias_paths = [
            os.path.join(self.lld_dark_frame_path, 'SonyA7S2', 'BiasFrame_ET_1_30'),  # 实际结构
            os.path.join(self.lld_dark_frame_path, 'BiasFrame_ET_1_30'),              # 备选路径
            os.path.join(self.lld_dark_frame_path, 'bias'),                           # 通用路径
            os.path.join(self.lld_dark_frame_path, 'dark_frames'),                    # 通用路径
        ]
        
        for bias_path in lld_bias_paths:
            if os.path.exists(bias_path):
                self._scan_bias_directory(bias_path)
        
        log(f"LLD暗帧数据加载完成，支持的ISO: {sorted(self.lld_dark_frames.keys())}")
    
    def _scan_bias_directory(self, bias_path):
        """
        扫描暗帧目录，支持LLD的文件命名格式
        """
        try:
            for item in os.listdir(bias_path):
                item_path = os.path.join(bias_path, item)
                
                if os.path.isdir(item_path) and item.isdigit():
                    # ISO目录（如 2500/, 3200/）
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
        从LLD .mat文件中提取ISO信息
        支持dark_2500_0001.mat格式和文件内容中的ISO字段
        """
        try:
            # 优先从文件内容提取（更准确）
            import scipy.io as sio
            mat_data = sio.loadmat(mat_path)
            
            if 'ISO' in mat_data:
                iso_value = mat_data['ISO']
                if hasattr(iso_value, 'item'):
                    return int(iso_value.item())
                else:
                    return int(iso_value)
            
            # 备选：从文件名提取 dark_XXXX_YYYY.mat 格式
            file_name = os.path.basename(mat_path)
            import re
            match = re.search(r'dark_(\d+)_\d+\.mat', file_name.lower())
            if match:
                return int(match.group(1))
            
            # 其他可能的文件名格式
            if 'iso' in file_name.lower():
                match = re.search(r'iso[\s_-]*(\d+)', file_name.lower())
                if match:
                    return int(match.group(1))
                    
        except Exception as e:
            log(f"从LLD mat文件提取ISO失败: {mat_path}, 错误: {e}")
        
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
            # 使用scipy直接加载LLD的mat文件
            import scipy.io as sio
            mat_data = sio.loadmat(selected_file)
            
            if 'Inoisy_crop' in mat_data:
                dark_frame = mat_data['Inoisy_crop'].astype(np.float32)
                
                # 验证ISO匹配
                if 'ISO' in mat_data:
                    file_iso = int(mat_data['ISO'].item() if hasattr(mat_data['ISO'], 'item') else mat_data['ISO'])
                    if file_iso != iso:
                        log(f"警告: 文件ISO({file_iso})与请求ISO({iso})不匹配: {selected_file}")
                
                # 调整尺寸以匹配目标图像
                if dark_frame.shape != image_shape:
                    dark_frame = self._resize_dark_frame(dark_frame, image_shape)
                
                return dark_frame
            else:
                log(f"警告: LLD mat文件中未找到'Inoisy_crop'键: {selected_file}")
                return None
                
        except Exception as e:
            log(f"加载LLD暗帧文件 {selected_file} 失败: {e}")
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
    
    def __getitem__(self, idx):
        """
        重写数据获取方法
        Enhanced_Mix_Dataset专注于数据加载，噪声合成交给trainer处理
        """
        # 调用父类Mix_Dataset方法获取完整数据（包含所有必要字段）
        data = super().__getitem__(idx)
        
        # 添加噪声合成相关的元数据，但不在这里执行噪声合成
        data['lld_available'] = self.use_lld_dark_frames and len(self.lld_dark_frames) > 0
        data['dark_frame_paths'] = self.get_dark_frame_paths() if self.use_lld_dark_frames else {}
        data['synthesis_config'] = {
            'use_simplified_noise_synthesis': self.use_simplified_noise_synthesis,
            'quantum_efficiency': self.quantum_efficiency,
            'sna_rate': self.sna_rate,
            'enhanced_rate': self.enhanced_rate
        }
        
        return data
    
    def get_dark_frame_paths(self):
        """
        获取暗帧路径字典，供PyTorch噪声合成器使用
        """
        return self.lld_dark_frames.copy()
    
    def get_synthesis_info(self):
        """
        获取噪声合成相关信息
        """
        return {
            'use_lld_dark_frames': self.use_lld_dark_frames,
            'use_simplified_noise_synthesis': self.use_simplified_noise_synthesis,
            'quantum_efficiency': self.quantum_efficiency,
            'sna_rate': self.sna_rate,
            'enhanced_rate': self.enhanced_rate,
            'available_lld_isos': sorted(self.lld_dark_frames.keys()),
            'total_dark_frames': sum(len(frames) for frames in self.lld_dark_frames.values()),
            'mix_dataset_info': {
                'legalISO': getattr(self, 'legalISO', []),
                'num_scenes': len(getattr(self, 'infos', []))
            }
        }


# 辅助函数：检查LLD数据集可用性
def check_lld_availability_for_mix(lld_path):
    """
    检查LLD数据集对Mix训练的可用性
    
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
        # Mix数据集常用的ISO范围
        common_isos = [100, 400, 800, 1600, 3200, 6400, 12800, 25600]
        
        # 扫描可能的暗帧目录
        bias_paths = [
            os.path.join(lld_path, 'SonyA7S2', 'BiasFrame_ET_1_30'),
            os.path.join(lld_path, 'BiasFrame_ET_1_30'),
            os.path.join(lld_path, 'bias'),
            os.path.join(lld_path, 'dark_frames')
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
        
        # 检查对常见ISO的覆盖度
        coverage = len(set(supported_isos) & set(common_isos))
        
        availability_info['available'] = len(supported_isos) > 0
        availability_info['iso_count'] = len(supported_isos)
        availability_info['total_dark_frames'] = total_frames
        availability_info['supported_isos'] = sorted(supported_isos)
        availability_info['common_iso_coverage'] = coverage
        availability_info['coverage_percent'] = (coverage / len(common_isos)) * 100
        
    except Exception as e:
        availability_info['errors'].append(f"扫描LLD数据时出错: {e}")
    
    return availability_info


if __name__ == '__main__':
    # 测试Enhanced_Mix_Dataset
    test_args = {
        'use_lld_dark_frames': True,
        'lld_dark_frame_path': '/data/LLD',
        'use_simplified_noise_synthesis': True,
        'quantum_efficiency': 0.4,
        'SNA_rate': 0.0,
        'enhanced_rate': 1.0,
        'mode': 'train',
        'command': 'enhanced_synthesis',
        'camera_type': 'SonyA7S2',
        # Mix_Dataset所需的参数
        'LRID_path': '/data/LRID',
        'SID_path': '/data/SID/Sony',
        'dstname': 'MIX'
    }
    
    # 检查LLD可用性
    lld_info = check_lld_availability_for_mix(test_args['lld_dark_frame_path'])
    print("LLD可用性检查结果（针对Mix训练）:")
    for key, value in lld_info.items():
        print(f"  {key}: {value}")
    
    if lld_info['available']:
        print(f"\n✓ LLD数据集可用，支持 {lld_info['iso_count']} 个ISO级别")
        print(f"  总暗帧数: {lld_info['total_dark_frames']}")
        print(f"  支持的ISO: {lld_info['supported_isos']}")
        print(f"  常见ISO覆盖度: {lld_info['common_iso_coverage']}/{len([100, 400, 800, 1600, 3200, 6400])} ({lld_info['coverage_percent']:.1f}%)")
        
        # 测试数据集创建
        try:
            dataset = Enhanced_Mix_Dataset(test_args)
            print(f"\n✓ Enhanced_Mix_Dataset创建成功")
            
            synthesis_info = dataset.get_synthesis_info()
            print("噪声合成信息:")
            for key, value in synthesis_info.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"\n✗ Enhanced_Mix_Dataset创建失败: {e}")
    else:
        print(f"\n✗ LLD数据集不可用")
        for error in lld_info['errors']:
            print(f"  错误: {error}")