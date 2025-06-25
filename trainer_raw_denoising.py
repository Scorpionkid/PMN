"""
修复版 Raw Image Denoising Trainer
解决简化噪声合成与PMN接口不匹配的问题
"""

import os
import time
import torch
import numpy as np
import pickle as pkl
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 导入PMN的基础组件
from trainer_SID import SID_Trainer, MultiProcessPlot, timestamp, tensor_dim5to4
from utils import *
from data_process.real_datasets import *
from data_process.process import *
from data_process.Enhanced_Mix_Dataset import Enhanced_Mix_Dataset


class RawDenoising_Trainer(SID_Trainer):
    """
    修复版：集成raw_image_denoising噪声合成方法的训练器
    关键修复：确保简化噪声合成与PMN接口一致
    """
    
    def __init__(self):
        super().__init__()
        
        # 初始化raw_image_denoising特有的组件
        self.setup_raw_denoising_components()
        
        log(f"Raw Image Denoising Trainer initialized", log=self.logfile)
        log(f"使用假设化系统增益方法: {self.use_hypothesized_gain}", log=self.logfile)
        log(f"使用简化噪声合成管道: {self.use_simplified_noise_synthesis}", log=self.logfile)
    
    def setup_raw_denoising_components(self):
        """设置raw_image_denoising特有的组件"""
        dst_config = self.dst
        self.use_hypothesized_gain = dst_config.get('use_simplified_noise_synthesis', True)
        self.use_simplified_noise_synthesis = dst_config.get('use_simplified_noise_synthesis', True)
        self.quantum_efficiency = dst_config.get('quantum_efficiency', 0.4)
        
        # 噪声合成权重
        self.sna_rate = dst_config.get('SNA_rate', 0.5)
        self.enhanced_rate = dst_config.get('enhanced_rate', 0.5)
        
        # 初始化LLD暗帧加载器
        if dst_config.get('use_lld_dark_frames', False):
            self.lld_dark_frame_loader = LLD_DarkFrameLoader(
                lld_path=dst_config.get('lld_dark_frame_path', 'resources'),
                cache_dark_frames=True
            )
            log(f"LLD暗帧加载器已初始化，支持的ISO范围: {self.lld_dark_frame_loader.available_isos}", 
                log=self.logfile)
        else:
            self.lld_dark_frame_loader = None
        
        log(f"噪声合成权重 - SNA: {self.sna_rate}, Enhanced: {self.enhanced_rate}", log=self.logfile)
    
    def simplified_noise_synthesis_pmn_compatible(self, clean_image, iso, ratio, dark_frame_paths=None):
        """
        PMN兼容的简化噪声合成方法 - 论文精确实现
        基于论文公式：Ka(X + Np) ~ Ka × Poisson(I/Ka)
        返回与SNA_torch相同格式的增量：(dn, dy, noise_params)
        """
        # 论文精确参数：K = ISO/100 × 0.1
        system_gain = iso / 100.0 * 0.1
        
        # 步骤1：计算放大的清洁图像 (应用数字增益)
        scaled_clean = clean_image * ratio
        
        # 步骤2：合成光子散粒噪声 Ka(X + Np) ~ Ka × Poisson(I/Ka)
        # 这里I = scaled_clean就是放大后的图像强度
        signal_for_poisson = torch.clamp(scaled_clean / system_gain, min=1e-6)
        
        # 生成泊松噪声并放大
        poisson_samples = torch.poisson(signal_for_poisson)
        photon_noise_component = poisson_samples * system_gain
        
        # 步骤3：添加信号无关噪声（直接暗帧采样）
        signal_independent_noise = self.get_signal_independent_noise_torch(
            clean_image.shape, iso, dark_frame_paths
        )
        
        # 步骤4：合成最终带噪图像
        # 论文公式：D = Ka(X + Np) + 信号无关噪声
        final_noisy = photon_noise_component + signal_independent_noise
        
        # 计算增量，与PMN的SNA_torch接口一致
        dn = final_noisy - scaled_clean     # 噪声增量
        dy = scaled_clean - clean_image     # 清洁图像增量（数字增益效果）
        
        # 噪声参数（模拟PMN格式）
        noise_params = torch.tensor([iso, ratio], device=self.device)
        
        return dn, dy, noise_params
    
    def get_signal_independent_noise_torch(self, image_shape, iso, dark_frame_paths=None):
        """
        获取信号无关噪声 - 直接暗帧采样（论文核心创新）
        """
        if dark_frame_paths and iso in dark_frame_paths and len(dark_frame_paths[iso]) > 0:
            # 随机选择一个暗帧文件
            selected_file = np.random.choice(dark_frame_paths[iso])
            
            try:
                # 加载.mat文件中的暗帧
                import scipy.io as sio
                mat_data = sio.loadmat(selected_file)
                
                if 'Inoisy_crop' in mat_data:
                    dark_frame = mat_data['Inoisy_crop'].astype(np.float32)
                    
                    # 调整尺寸匹配
                    if len(image_shape) == 4:  # batch, channel, height, width
                        target_shape = image_shape[2:]  # height, width
                    else:  # channel, height, width
                        target_shape = image_shape[1:]
                    
                    # 调整暗帧尺寸
                    dark_frame_resized = self._resize_dark_frame_torch(dark_frame, target_shape)
                    
                    # 转换为正确的通道格式 (从HW变为CHW或BCHW)
                    if len(image_shape) == 4:  # batch
                        channels = image_shape[1]
                        if channels == 4:  # Bayer pattern
                            dark_frame_bayer = self._to_bayer_torch(dark_frame_resized)
                        else:
                            dark_frame_bayer = dark_frame_resized.unsqueeze(0).repeat(channels, 1, 1)
                        dark_frame_tensor = dark_frame_bayer.unsqueeze(0)  # Add batch dim
                    else:  # single image
                        channels = image_shape[0]
                        if channels == 4:  # Bayer pattern
                            dark_frame_tensor = self._to_bayer_torch(dark_frame_resized)
                        else:
                            dark_frame_tensor = dark_frame_resized.unsqueeze(0).repeat(channels, 1, 1)
                    
                    return dark_frame_tensor.to(self.device)
                
            except Exception as e:
                log(f"暗帧加载失败 {selected_file}: {e}")
        
        # 备选方案：统计噪声模型
        return self._fallback_statistical_noise_torch(image_shape, iso)
    
    def _resize_dark_frame_torch(self, dark_frame, target_shape):
        """使用PyTorch调整暗帧尺寸"""
        dark_tensor = torch.from_numpy(dark_frame).float()
        h, w = target_shape
        fh, fw = dark_tensor.shape
        
        if fh >= h and fw >= w:
            # 中心裁剪
            start_h = (fh - h) // 2
            start_w = (fw - w) // 2
            return dark_tensor[start_h:start_h+h, start_w:start_w+w]
        else:
            # 填充到目标尺寸
            padded = torch.zeros((h, w), dtype=dark_tensor.dtype)
            start_h = max(0, (h - fh) // 2)
            start_w = max(0, (w - fw) // 2)
            end_h = min(h, start_h + fh)
            end_w = min(w, start_w + fw)
            
            padded[start_h:end_h, start_w:end_w] = dark_tensor[:end_h-start_h, :end_w-start_w]
            return padded
    
    def _to_bayer_torch(self, gray_image):
        """将灰度暗帧转换为4通道Bayer格式 - 保持原始空间尺寸"""
        h, w = gray_image.shape
        bayer = torch.zeros(4, h, w, dtype=gray_image.dtype)  # 保持原始尺寸
        
        # RGGB Bayer pattern - 交错采样但保持尺寸
        bayer[0] = gray_image  # R通道使用完整图像
        bayer[1] = gray_image  # G1通道使用完整图像  
        bayer[2] = gray_image  # G2通道使用完整图像
        bayer[3] = gray_image  # B通道使用完整图像
        
        return bayer
    
    def _fallback_statistical_noise_torch(self, image_shape, iso):
        """备选统计噪声模型"""
        noise_std = np.sqrt(iso / 100.0) * 0.1
        return torch.normal(
            mean=torch.zeros(image_shape),
            std=noise_std
        ).to(self.device)
    
    def batch_simplified_noise_synthesis_pmn_compatible(self, clean_images, iso_list, ratio_list, aug_wb_list, dark_frame_paths=None):
        """
        批量PMN兼容的简化噪声合成
        直接修改imgs_lr和imgs_hr，与PMN的SNA处理方式一致
        """
        b = clean_images.shape[0]
        
        for i in range(b):
            # 只对有效的白平衡增强参数进行处理（与PMN逻辑一致）
            if np.abs(aug_wb_list[i]).max() != 0:
                dn, dy, p = self.simplified_noise_synthesis_pmn_compatible(
                    clean_images[i], 
                    iso=iso_list[i], 
                    ratio=ratio_list[i],
                    dark_frame_paths=dark_frame_paths
                )
                
                # 与PMN相同的增量应用方式
                clean_images[i] = clean_images[i] + dy  # 更新清洁图像
                # 注意：这里不直接修改imgs_lr，而是返回噪声增量
                yield i, dn  # 返回索引和噪声增量

    def get_synthesis_method(self):
        """根据配置决定使用哪种噪声合成方法"""
        dataset_name = self.args['dst_train']['dataset']
        
        if dataset_name != 'Enhanced_Mix_Dataset':
            return 'sna'
        
        command = self.dst.get('command', '')
        
        if 'enhanced_synthesis' in command and self.use_simplified_noise_synthesis:
            if self.sna_rate == 0:
                return 'simplified'
            elif self.enhanced_rate == 0:
                return 'sna'
            else:
                return 'hybrid'
        else:
            return 'sna'
    
    def preprocess(self, data, mode='train', preprocess=True):
        """
        修复版预处理函数
        关键修复：确保简化噪声合成与PMN接口一致
        """
        # PMN的基础预处理
        imgs_hr = tensor_dim5to4(data['hr']).type(torch.FloatTensor).to(self.device)
        imgs_lr = tensor_dim5to4(data['lr']).type(torch.FloatTensor).to(self.device)
        
        dst = self.dst_train if mode=='train' else self.dst_eval
        noise_map = None
        
        if self.use_gpu and mode=='train' and preprocess:
            b = imgs_lr.shape[0]
            
            if self.args['dst_train']['dataset'] == 'Enhanced_Mix_Dataset':
                # 使用增强的Mix_Dataset和简化噪声合成
                data['ratio'] = data['ratio'].view(-1).type(torch.FloatTensor).to(self.device)
                
                # 保持PMN的数据增强逻辑
                aug_r, aug_g, aug_b = get_aug_param_torch(data, b=b, command=self.dst['command'])
                aug_wbs = torch.stack((aug_r, aug_g, aug_b, aug_g), dim=1)
                data['rgb_gain'] = torch.ones(b) * (aug_g + 1)
                data['wb'] = data['wb'][0].repeat(b, 1)
                
                # 准备批量处理的数据
                iso_list = []
                ratio_list = []
                aug_wb_list = []
                
                for i in range(b):
                    aug_wb = aug_wbs[i].numpy()
                    if data['black_lr'][0]: 
                        aug_wb += 1
                    dgain = data['ratio'][i]
                    
                    # 应用数字增益（与PMN一致）
                    imgs_lr[i] = imgs_lr[i] if self.dst['ori'] else imgs_lr[i] * dgain
                    iso = data['ISO'][i//self.dst['crop_per_image']].item()
                    
                    iso_list.append(iso)
                    ratio_list.append(dgain.item())
                    aug_wb_list.append(aug_wb)
                    
                    # 应用白平衡增强（与PMN一致）
                    if np.abs(aug_wb).max() != 0:
                        data['wb'][i] *= (1+aug_wb[1]) / (1+aug_wb)
                
                # 决定使用哪种噪声合成方法
                synthesis_method = self.get_synthesis_method()
                
                if synthesis_method == 'simplified':
                    # 修复：使用PMN兼容的简化噪声合成
                    dark_frame_paths = getattr(self.lld_dark_frame_loader, 'dark_frame_paths', {}) if self.lld_dark_frame_loader else {}
                    
                    for i, dn in self.batch_simplified_noise_synthesis_pmn_compatible(
                        imgs_hr, iso_list, ratio_list, aug_wb_list, dark_frame_paths
                    ):
                        # 与PMN相同的增量应用方式
                        imgs_lr[i] = imgs_lr[i] + dn
                    
                elif synthesis_method == 'hybrid':
                    # 混合方法
                    dark_frame_paths = getattr(self.lld_dark_frame_loader, 'dark_frame_paths', {}) if self.lld_dark_frame_loader else {}
                    
                    for i in range(b):
                        if np.abs(aug_wb_list[i]).max() != 0:
                            if np.random.rand() < self.enhanced_rate / (self.sna_rate + self.enhanced_rate):
                                # 使用简化方法
                                dn, dy, p = self.simplified_noise_synthesis_pmn_compatible(
                                    imgs_hr[i], iso=iso_list[i], ratio=ratio_list[i], 
                                    dark_frame_paths=dark_frame_paths
                                )
                                imgs_lr[i] = imgs_lr[i] + dn
                                imgs_hr[i] = imgs_hr[i] + dy
                            else:
                                # 使用PMN的SNA方法
                                dn, dy, p = SNA_torch(
                                    imgs_hr[i], aug_wb_list[i], iso=iso_list[i], 
                                    ratio=ratio_list[i], black_lr=data['black_lr'][0],
                                    camera_type=self.dst['camera_type']
                                )
                                imgs_lr[i] = imgs_lr[i] + dn
                                imgs_hr[i] = imgs_hr[i] + dy
                
                else:
                    # 使用PMN的SNA方法（保持原有逻辑）
                    for i in range(b):
                        if np.abs(aug_wb_list[i]).max() != 0:
                            dn, dy, p = SNA_torch(
                                imgs_hr[i], aug_wb_list[i], iso=iso_list[i], 
                                ratio=ratio_list[i], black_lr=data['black_lr'][0],
                                camera_type=self.dst['camera_type']
                            )
                            imgs_lr[i] = imgs_lr[i] + dn
                            imgs_hr[i] = imgs_hr[i] + dy
                        
            elif 'SNA' in self.dst['command']:
                # 保持与原有SNA的兼容性
                return super().preprocess(data, mode, preprocess)
        
        # 最终处理（与PMN一致）
        ratio = data['ratio'].type(torch.FloatTensor).to(self.device)
        ratio = ratio.view(-1,1,1,1)
        if 'rgb_gain' in data:
            data['rgb_gain'] = data['rgb_gain'].type(torch.FloatTensor).to(self.device).view_as(ratio)
        
        return imgs_lr, imgs_hr, ratio, noise_map


class LLD_DarkFrameLoader:
    """LLD暗帧数据加载器"""
    
    def __init__(self, lld_path, cache_dark_frames=True):
        self.lld_path = lld_path
        self.cache_dark_frames = cache_dark_frames
        self.dark_frames_cache = {}
        self.available_isos = []
        self.dark_frame_paths = {}
        
        self._scan_available_data()
        
    def _scan_available_data(self):
        """扫描LLD数据集中可用的暗帧数据"""
        if not os.path.exists(self.lld_path):
            log(f"警告: LLD暗帧路径不存在: {self.lld_path}")
            return
        
        # 扫描LLD数据结构
        bias_paths = [
            os.path.join(self.lld_path, 'SonyA7S2', 'BiasFrame_ET_1_30'),
            os.path.join(self.lld_path, 'BiasFrame_ET_1_30'),
            self.lld_path
        ]
        
        for bias_path in bias_paths:
            if os.path.exists(bias_path):
                self._scan_bias_directory(bias_path)
                break
    
    def _scan_bias_directory(self, bias_path):
        """扫描暗帧目录"""
        try:
            for iso_dir in os.listdir(bias_path):
                iso_path = os.path.join(bias_path, iso_dir)
                if os.path.isdir(iso_path) and iso_dir.isdigit():
                    iso = int(iso_dir)
                    mat_files = [f for f in os.listdir(iso_path) if f.endswith('.mat')]
                    
                    if mat_files:
                        self.available_isos.append(iso)
                        self.dark_frame_paths[iso] = [
                            os.path.join(iso_path, f) for f in mat_files
                        ]
                        
            self.available_isos.sort()
            log(f"扫描完成，找到 {len(self.available_isos)} 个ISO级别的暗帧数据")
            
        except Exception as e:
            log(f"扫描LLD暗帧数据时出错: {e}")


# 测试函数
if __name__ == '__main__':
    print("修复版 Raw Image Denoising Trainer")
    print("主要修复：")
    print("1. 简化噪声合成与PMN接口一致性")
    print("2. 正确的增量式噪声应用")
    print("3. 数据流处理逻辑修复")