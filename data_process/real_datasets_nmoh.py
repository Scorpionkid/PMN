"""
基于NMOH (Noise Modeling in One Hour)论文的简化暗帧处理版本
主要改动：
1. 简化暗帧校准流程，每个ISO只需10个暗帧
2. 去除复杂的统计建模
3. 支持快速在线重校准
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.io as sio
import pickle as pkl
import rawpy
from data_process.utils import pack_raw_bayer  # 正确导入
from data_process.real_datasets import RealBase_Dataset  # 继承原有基类
from utils import log

class RealBase_Dataset_NMOH(RealBase_Dataset):
    """基于NMOH简化的数据集基类"""
    
    def __init__(self, args=None):
        super().__init__(args)
        
        # NMOH配置参数
        self.nmoh_config = {
            'min_dark_frames': 10,  # 每个ISO最少暗帧数
            'online_recalib_frames': 2,  # 在线重校准所需暗帧数
            'use_simple_dark_shading': True,  # 使用简化的暗帧校正
            'skip_hbnr': True,  # 跳过高位深度噪声恢复
            'skip_statistical_profiling': True,  # 跳过统计建模
        }
        
        # 检查是否使用NMOH简化模式
        if 'nmoh' in self.args.get('command', ''):
            log('Using NMOH simplified dark frame calibration')
            self.use_nmoh = True
            self.init_nmoh_dark_frames()
        else:
            self.use_nmoh = False
            
    def init_nmoh_dark_frames(self):
        """NMOH简化的暗帧初始化"""
        log('Initializing NMOH dark frames...')
        
        # 记录暗帧目录
        self.black_dirs = sorted(os.listdir(self.args['bias_dir']), key=lambda x: int(x))
        self.legalISO = np.array([int(dirname) for dirname in self.black_dirs])
        self.black_dirs = [os.path.join(self.args['bias_dir'], dirname) for dirname in self.black_dirs]
        
        # NMOH: 每个ISO只加载必要数量的暗帧
        self.dark_frames_nmoh = {}
        self.dark_shadings_nmoh = {}
        
        for i, iso in enumerate(self.legalISO):
            dark_dir = self.black_dirs[i]
            dark_files = sorted(os.listdir(dark_dir))
            
            # 只使用前min_dark_frames个暗帧
            n_frames = min(len(dark_files), self.nmoh_config['min_dark_frames'])
            selected_files = dark_files[:n_frames]
            
            log(f'ISO {iso}: Using {n_frames} dark frames (NMOH mode)')
            
            # 加载并平均暗帧得到dark shading
            dark_frames = []
            for fname in selected_files:
                fpath = os.path.join(dark_dir, fname)
                if fname.endswith('.npy'):
                    dark = np.load(fpath).astype(np.float32)
                elif fname.endswith('.mat'):
                    mat_data = sio.loadmat(fpath)
                    dark = mat_data['Inoisy_crop'].astype(np.float32)
                else:
                    continue
                dark_frames.append(dark)
            
            if dark_frames:
                # NMOH: 简单平均即可，不需要复杂统计
                self.dark_frames_nmoh[iso] = dark_frames
                self.dark_shadings_nmoh[iso] = np.mean(dark_frames, axis=0)
                log(f'  Dark shading shape: {self.dark_shadings_nmoh[iso].shape}')
                log(f'  Dark shading range: [{self.dark_shadings_nmoh[iso].min():.1f}, '
                    f'{self.dark_shadings_nmoh[iso].max():.1f}]')
    
    def get_darkshading_nmoh(self, iso, temperature=None):
        """NMOH简化的暗帧获取"""
        if iso not in self.dark_shadings_nmoh:
            # 找最近的ISO
            idx = np.argmin(np.abs(self.legalISO - iso))
            nearest_iso = self.legalISO[idx]
            log(f'ISO {iso} not found, using nearest ISO {nearest_iso}')
            return self.dark_shadings_nmoh[nearest_iso]
        
        dark_shading = self.dark_shadings_nmoh[iso].copy()
        
        # 可选：基于温度的简单补偿（NMOH发现温度影响有限）
        if temperature is not None and self.args.get('temp_compensation', False):
            # 简单的线性温度补偿
            temp_ref = 25.0  # 参考温度
            temp_coeff = 0.01  # 温度系数（可调）
            dark_shading *= (1 + temp_coeff * (temperature - temp_ref))
            
        return dark_shading
    
    def online_recalibration(self, iso, new_dark_frames):
        """NMOH在线重校准（推理时使用）"""
        if len(new_dark_frames) >= self.nmoh_config['online_recalib_frames']:
            # 使用少量新暗帧快速重校准
            log(f'Online recalibration for ISO {iso} with {len(new_dark_frames)} frames')
            new_dark_shading = np.mean(new_dark_frames, axis=0)
            
            # 可选：与原有dark shading加权平均
            if iso in self.dark_shadings_nmoh:
                alpha = 0.3  # 新暗帧的权重
                self.dark_shadings_nmoh[iso] = (1-alpha) * self.dark_shadings_nmoh[iso] + alpha * new_dark_shading
            else:
                self.dark_shadings_nmoh[iso] = new_dark_shading
                
            return True
        return False
    
    def apply_nmoh_dark_shading_correction(self, noisy_raw, iso, ratio):
        """应用NMOH简化的暗帧校正"""
        if not self.use_nmoh:
            # 使用原有方法
            return self.apply_darkshading_correction(noisy_raw, iso, ratio)
        
        # 获取dark shading
        dark_shading = self.get_darkshading_nmoh(iso)
        
        # 根据曝光比例调整（如果需要）
        if self.args.get('ratio_adaptive_ds', False):
            # 暗帧通常在短曝光下拍摄，长曝光需要调整
            dark_shading = dark_shading * (ratio / 100.0)
        
        # 应用校正
        corrected = noisy_raw - dark_shading
        
        # 确保非负
        corrected = np.maximum(corrected, 0)
        
        return corrected
    
    def __getitem__(self, index):
        """重写getitem以支持NMOH模式，保持与原版兼容的输出格式"""
        if not self.use_nmoh:
            # 使用原有方法
            return super().__getitem__(index)
        
        # NMOH简化版本的数据加载
        info = self.infos[index % self.ori_len]
        
        # 加载raw数据
        gt_raw = self.load_raw(info['long'])
        noisy_raw = self.load_raw(info['short'])
        
        # 获取元信息
        iso = info['ISO']
        ratio = info['ratio']
        wb = info.get('wb', None)
        
        # NMOH暗帧校正
        if 'darkshading' in self.args.get('command', ''):
            noisy_raw = self.apply_nmoh_dark_shading_correction(noisy_raw, iso, ratio)
            # 对GT也可以应用（如果需要）
            if self.args.get('correct_gt', False):
                gt_raw = self.apply_nmoh_dark_shading_correction(gt_raw, iso, 1.0)
        
        # 数据增强（保持原有逻辑）
        if self.args['mode'] == 'train':
            # Crop
            h_start = self.h_start[index]
            w_start = self.w_start[index]
            h_end = self.h_end[index]
            w_end = self.w_end[index]
            
            gt_raw = gt_raw[h_start:h_end, w_start:w_end]
            noisy_raw = noisy_raw[h_start:h_end, w_start:w_end]
            
            # 数据增强
            aug_mode = self.aug_mode[index]
            gt_raw = self.data_aug(gt_raw, mode=aug_mode)
            noisy_raw = self.data_aug(noisy_raw, mode=aug_mode)
        
        # 打包为4通道
        gt_packed = pack_raw_bayer(gt_raw)
        noisy_packed = pack_raw_bayer(noisy_raw)
        
        # 转为tensor
        gt_packed = torch.from_numpy(gt_packed).float()
        noisy_packed = torch.from_numpy(noisy_packed).float()
        
        # 返回格式与原版保持一致
        # 噪声图在trainer中的preprocess函数生成，不在这里生成
        return {
            'noisy': noisy_packed,
            'gt': gt_packed,
            'ISO': iso,
            'ratio': ratio,
            'wb': wb,
            'fname': info['short'].split('/')[-1]
        }


class SID_Dataset_NMOH(RealBase_Dataset_NMOH):
    """SID数据集的NMOH版本"""
    
    def __init__(self, args=None):
        # 设置SID特定参数
        args['dataset_name'] = 'SID'
        super().__init__(args)
        
        # SID特定的处理
        self.setup_sid_specific()
    
    def setup_sid_specific(self):
        """SID数据集特定设置"""
        log(f'Setting up SID dataset with NMOH mode: {self.use_nmoh}')
        
        if self.use_nmoh:
            # 输出NMOH统计信息
            log('=== NMOH Dark Frame Statistics ===')
            for iso in sorted(self.dark_shadings_nmoh.keys()):
                ds = self.dark_shadings_nmoh[iso]
                log(f'ISO {iso}: mean={ds.mean():.2f}, std={ds.std():.2f}')
            log('===================================')


class ELD_Dataset_NMOH(RealBase_Dataset_NMOH):
    """ELD数据集的NMOH版本"""
    
    def __init__(self, args=None):
        # 设置ELD特定参数
        args['dataset_name'] = 'ELD'
        super().__init__(args)
        
        # ELD特定的处理
        self.setup_eld_specific()
    
    def setup_eld_specific(self):
        """ELD数据集特定设置"""
        log(f'Setting up ELD dataset with NMOH mode: {self.use_nmoh}')
        
        # ELD可能需要场景特定的处理
        if hasattr(self, 'scene_list'):
            log(f'ELD scenes: {self.scene_list}')