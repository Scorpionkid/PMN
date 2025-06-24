"""
Raw Image Denoising Trainer
基于PMN框架集成raw_image_denoising的噪声合成方法
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
    集成raw_image_denoising噪声合成方法的训练器
    继承PMN的完整训练框架，添加简化噪声合成管道
    """
    
    def __init__(self):
        super().__init__()
        
        # 初始化raw_image_denoising特有的组件
        self.setup_raw_denoising_components()
        
        log(f"Raw Image Denoising Trainer initialized", log=self.logfile)
        log(f"使用假设化系统增益方法: {self.use_hypothesized_gain}", log=self.logfile)
        log(f"使用简化噪声合成管道: {self.use_simplified_noise_synthesis}", log=self.logfile)
    
    def setup_raw_denoising_components(self):
        """
        设置raw_image_denoising特有的组件
        """
        # 从配置中读取raw_image_denoising相关参数
        dst_config = self.dst
        self.use_hypothesized_gain = dst_config.get('use_simplified_noise_synthesis', True)
        self.use_simplified_noise_synthesis = dst_config.get('use_simplified_noise_synthesis', True)
        self.quantum_efficiency = dst_config.get('quantum_efficiency', 0.4)  # 假设量子效率40%
        
        # 噪声合成权重
        self.sna_rate = dst_config.get('SNA_rate', 0.5)
        self.enhanced_rate = dst_config.get('enhanced_rate', 0.5)
        
        # 初始化LLD暗帧加载器（如果配置了的话）
        if dst_config.get('use_lld_dark_frames', False):
            self.lld_dark_frame_loader = LLD_DarkFrameLoader(
                lld_path=dst_config.get('lld_dark_frame_path', '/data/LLD_calibration'),
                cache_dark_frames=True
            )
            log(f"LLD暗帧加载器已初始化，支持的ISO范围: {self.lld_dark_frame_loader.available_isos}", 
                log=self.logfile)
        else:
            self.lld_dark_frame_loader = None
            log("未启用LLD暗帧数据", log=self.logfile)
        
        log(f"使用假设化系统增益方法: {self.use_hypothesized_gain}", log=self.logfile)
        log(f"使用简化噪声合成管道: {self.use_simplified_noise_synthesis}", log=self.logfile)
        log(f"噪声合成权重 - SNA: {self.sna_rate}, Enhanced: {self.enhanced_rate}", log=self.logfile)
    
    def hypothesize_system_gain(self, iso):
        """
        实现论文提出的假设化系统增益方法
        K = ISO/100 * quantum_efficiency
        避免复杂的系统增益标定过程
        """
        if self.use_hypothesized_gain:
            # 使用论文中的假设化公式
            system_gain = iso / 100.0 * self.quantum_efficiency
            return system_gain
        else:
            # 回退到传统的标定方法（如果有的话）
            return self.get_calibrated_system_gain(iso)
    
    def get_calibrated_system_gain(self, iso):
        """
        传统的系统增益标定方法（备用）
        """
        # 这里可以集成传统的标定方法
        # 目前使用PMN的方法作为备选
        return iso / 100.0 * 0.1
    
    def simplified_noise_synthesis(self, clean_image, iso, ratio=1.0, use_dark_frames=True):
        """
        实现论文提出的简化噪声合成管道
        仅需要暗帧收集，避免复杂的噪声建模和标定
        
        Args:
            clean_image: 清洁图像
            iso: ISO值
            ratio: 数字增益
            use_dark_frames: 是否使用暗帧
        
        Returns:
            synthesized_noisy_image: 合成的带噪图像
        """
        if not self.use_simplified_noise_synthesis:
            # 回退到PMN的SNA方法
            return self.pmn_noise_synthesis(clean_image, iso, ratio)
        
        # 获取假设化系统增益
        system_gain = self.hypothesize_system_gain(iso)
        
        # 合成光子散粒噪声（signal-dependent）
        photon_noise = self.synthesize_photon_noise(clean_image, system_gain, ratio)
        
        # 获取信号无关噪声（通过直接暗帧采样）
        if use_dark_frames and self.lld_dark_frame_loader.is_available(iso):
            signal_independent_noise = self.lld_dark_frame_loader.sample_noise(
                iso=iso, 
                image_shape=clean_image.shape
            )
        else:
            # 备选方案：使用统计模型
            signal_independent_noise = self.synthesize_signal_independent_noise(
                clean_image.shape, iso
            )
        
        # 合成最终噪声图像
        noisy_image = clean_image * ratio + photon_noise + signal_independent_noise
        
        return noisy_image
    
    def synthesize_photon_noise(self, clean_image, system_gain, ratio):
        """
        合成光子散粒噪声
        基于泊松分布的光子噪声模型
        """
        # 光子散粒噪声遵循泊松分布，方差等于均值
        # noise_variance = clean_image * system_gain * ratio
        photon_noise = torch.normal(
            mean=torch.zeros_like(clean_image),
            std=torch.sqrt(torch.clamp(clean_image * system_gain * ratio, min=1e-6))
        )
        return photon_noise
    
    def synthesize_signal_independent_noise(self, image_shape, iso):
        """
        合成信号无关噪声（备选方案）
        当暗帧不可用时使用统计模型
        """
        # 简单的高斯噪声模型，方差与ISO相关
        noise_std = np.sqrt(iso / 100.0) * 0.1  # 简化的ISO依赖噪声模型
        signal_independent_noise = torch.normal(
            mean=torch.zeros(image_shape),
            std=noise_std
        )
        return signal_independent_noise
    
    def pmn_noise_synthesis(self, clean_image, iso, ratio):
        """
        备选方案：使用PMN的SNA方法
        """
        # 这里调用原始的SNA_torch方法
        # 保持与原有PMN框架的兼容性
        aug_wb = np.array([0, 0, 0, 0])  # 不进行白平衡增强
        
        dn, dy, p = SNA_torch(
            clean_image, 
            aug_wb, 
            iso=iso, 
            ratio=ratio, 
            black_lr=True,
            camera_type='SonyA7S2'
        )
        return dn
    
    def preprocess(self, data, mode='train', preprocess=True):
        """
        重写预处理函数，集成raw_image_denoising的噪声合成方法
        """
        # 保持PMN的基础预处理逻辑
        imgs_hr = tensor_dim5to4(data['hr']).type(torch.FloatTensor).to(self.device)
        imgs_lr = tensor_dim5to4(data['lr']).type(torch.FloatTensor).to(self.device)
        
        dst = self.dst_train if mode=='train' else self.dst_eval
        noise_map = None
        
        if self.use_gpu and mode=='train' and preprocess:
            b = imgs_lr.shape[0]
            
            if self.args['dst_train']['dataset'] == 'Enhanced_Mix_Dataset':
                # 使用增强的Mix_Dataset和简化噪声合成
                data['ratio'] = data['ratio'].view(-1).type(torch.FloatTensor).to(self.device)
                
                # 获取增强参数（保持PMN的SNA逻辑）
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
                    imgs_lr[i] = imgs_lr[i] if self.dst['ori'] else imgs_lr[i] * dgain
                    iso = data['ISO'][i//self.dst['crop_per_image']].item()
                    
                    iso_list.append(iso)
                    ratio_list.append(dgain.item())
                    aug_wb_list.append(aug_wb)
                    
                    if np.abs(aug_wb).max() != 0:
                        data['wb'][i] *= (1+aug_wb[1]) / (1+aug_wb)
                
                # 决定使用哪种噪声合成方法
                synthesis_method = self.get_synthesis_method()
                
                if synthesis_method == 'simplified':
                    # 使用简化噪声合成管道（批量处理）
                    enhanced_noisy = self.batch_simplified_noise_synthesis(
                        imgs_hr, iso_list, ratio_list
                    )
                    imgs_lr = enhanced_noisy
                    
                elif synthesis_method == 'hybrid':
                    # 混合方法：一部分用SNA，一部分用简化方法
                    for i in range(b):
                        if np.random.rand() < self.dst.get('simplified_rate', 0.5):
                            # 使用简化方法
                            enhanced_noisy = self.simplified_noise_synthesis(
                                imgs_hr[i], iso=iso_list[i], ratio=ratio_list[i]
                            )
                            imgs_lr[i] = enhanced_noisy
                        else:
                            # 使用PMN的SNA方法
                            if np.abs(aug_wb_list[i]).max() != 0:
                                dn, dy, p = SNA_torch(
                                    imgs_hr[i], aug_wb_list[i], iso=iso_list[i], 
                                    ratio=ratio_list[i], black_lr=data['black_lr'][0],
                                    camera_type=self.dst['camera_type']
                                )
                                imgs_lr[i] = dn
                
                else:
                    # 使用PMN的SNA方法
                    for i in range(b):
                        if np.abs(aug_wb_list[i]).max() != 0:
                            dn, dy, p = SNA_torch(
                                imgs_hr[i], aug_wb_list[i], iso=iso_list[i], 
                                ratio=ratio_list[i], black_lr=data['black_lr'][0],
                                camera_type=self.dst['camera_type']
                            )
                            # 正确的PMN实现：dn和dy是增量，需要加到原图像上
                            imgs_lr[i] = imgs_lr[i] + dn  # 加噪声增量
                            imgs_hr[i] = imgs_hr[i] + dy  # 加清洁图像增量
                        
            elif 'SNA' in self.dst['command']:
                # 保持与原有SNA的兼容性
                return super().preprocess(data, mode, preprocess)
        
        ratio = data['ratio'] if 'ratio' in data else torch.ones(b)
        return imgs_lr, imgs_hr, ratio, noise_map
    
    def get_synthesis_method(self):
        """
        根据配置决定使用哪种噪声合成方法
        """
        # 检查数据集类型
        dataset_name = self.args['dst_train']['dataset']
        
        # 只有Enhanced_Mix_Dataset才使用新的噪声合成方法
        if dataset_name != 'Enhanced_Mix_Dataset':
            return 'sna'  # 其他数据集保持PMN原有逻辑
        
        command = self.dst.get('command', '')
        
        # 检查是否启用了enhanced_synthesis
        if 'enhanced_synthesis' in command and self.use_simplified_noise_synthesis:
            if self.sna_rate == 0:
                return 'simplified'
            elif self.enhanced_rate == 0:
                return 'sna'
            else:
                return 'hybrid'
        else:
            return 'sna'
    
    def batch_simplified_noise_synthesis(self, clean_images, iso_list, ratio_list, dark_frame_paths=None):
        """
        批量简化噪声合成
        """
        # 导入PyTorch噪声合成器
        from data_process.torch_noise_synthesis import TorchSimplifiedNoiseSynthesis
        
        synthesizer = TorchSimplifiedNoiseSynthesis(
            quantum_efficiency=self.quantum_efficiency,
            device=self.device
        )
        
        # 使用传入的暗帧路径，如果没有则尝试从加载器获取
        if dark_frame_paths is None and self.lld_dark_frame_loader:
            dark_frame_paths = getattr(self.lld_dark_frame_loader, 'dark_frame_paths', {})
        
        # 批量合成噪声
        noisy_images = synthesizer.synthesize_batch_noise(
            clean_images, iso_list, ratio_list, dark_frame_paths, use_random_gain=True
        )
        
        return noisy_images


class LLD_DarkFrameLoader:
    """
    LLD暗帧数据加载器
    支持.mat格式的暗帧数据，实现高效的暗帧采样
    """
    
    def __init__(self, lld_path, cache_dark_frames=True):
        self.lld_path = lld_path
        self.cache_dark_frames = cache_dark_frames
        self.dark_frames_cache = {}
        self.available_isos = []
        self.dark_frame_paths = {}  # 添加这个属性供外部访问
        
        # 初始化时扫描可用的暗帧数据
        self._scan_available_data()
        
    def _scan_available_data(self):
        """
        扫描LLD数据集中可用的暗帧数据
        """
        if not os.path.exists(self.lld_path):
            log(f"警告: LLD暗帧路径不存在: {self.lld_path}")
            return
        
        # 扫描LLD目录结构，查找暗帧文件
        # LLD数据集通常包含~400 dark frames per ISO, 24个ISO级别
        bias_dirs = ['bias', 'dark_frames', 'calibration']
        
        for bias_dir in bias_dirs:
            bias_path = os.path.join(self.lld_path, bias_dir)
            if os.path.exists(bias_path):
                self._scan_bias_directory(bias_path)
        
        self.available_isos = sorted(list(set(self.available_isos)))
        log(f"发现LLD暗帧数据，支持的ISO: {self.available_isos}")
    
    def _scan_bias_directory(self, bias_path):
        """
        扫描暗帧目录
        """
        try:
            for item in os.listdir(bias_path):
                item_path = os.path.join(bias_path, item)
                
                if os.path.isdir(item_path) and item.isdigit():
                    # ISO目录（如 2500/, 3200/）
                    iso = int(item)
                    dark_frame_files = []
                    
                    for file_name in os.listdir(item_path):
                        if file_name.endswith('.mat'):
                            file_path = os.path.join(item_path, file_name)
                            dark_frame_files.append(file_path)
                    
                    if dark_frame_files:
                        self.dark_frame_paths[iso] = dark_frame_files
                        self.available_isos.append(iso)
                        log(f"为ISO {iso} 找到 {len(dark_frame_files)} 个暗帧文件")
                        
                elif item.endswith('.mat'):
                    # 直接的.mat文件，尝试从文件名提取ISO
                    iso = self._extract_iso_from_filename(item)
                    if iso is not None:
                        if iso not in self.dark_frame_paths:
                            self.dark_frame_paths[iso] = []
                        self.dark_frame_paths[iso].append(item_path)
                        if iso not in self.available_isos:
                            self.available_isos.append(iso)
        except Exception as e:
            log(f"扫描暗帧目录 {bias_path} 时出错: {e}")
    
    def _extract_iso_from_filename(self, filename):
        """
        从文件名中提取ISO信息
        """
        import re
        
        # 支持 dark_2500_0001.mat 格式
        match = re.search(r'dark_(\d+)_\d+', filename.lower())
        if match:
            return int(match.group(1))
        
        # 原有的其他格式
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
    
    def is_available(self, iso):
        """
        检查指定ISO的暗帧数据是否可用
        """
        return iso in self.available_isos
    
    def load_dark_frames_for_iso(self, iso):
        """
        为指定ISO加载所有暗帧数据
        """
        if iso in self.dark_frames_cache:
            return self.dark_frames_cache[iso]
        
        dark_frames = []
        
        # 在LLD目录中查找该ISO的暗帧文件
        for bias_dir in ['bias', 'dark_frames', 'calibration']:
            iso_path = os.path.join(self.lld_path, bias_dir, str(iso))
            if os.path.exists(iso_path):
                for file_name in os.listdir(iso_path):
                    if file_name.endswith('.mat'):
                        file_path = os.path.join(iso_path, file_name)
                        try:
                            # 使用项目中现有的dataload函数
                            dark_frame = dataload(file_path)
                            dark_frames.append(dark_frame)
                        except Exception as e:
                            log(f"警告: 无法加载暗帧文件 {file_path}: {e}")
        
        if self.cache_dark_frames and dark_frames:
            self.dark_frames_cache[iso] = dark_frames
        
        log(f"为ISO {iso} 加载了 {len(dark_frames)} 个暗帧")
        return dark_frames
    
    def sample_noise(self, iso, image_shape, num_samples=1):
        """
        从暗帧中采样信号无关噪声
        实现论文中的直接暗帧采样方法
        """
        dark_frames = self.load_dark_frames_for_iso(iso)
        
        if not dark_frames:
            log(f"警告: ISO {iso} 没有可用的暗帧，使用统计噪声模型")
            return self._fallback_noise_sampling(image_shape, iso)
        
        # 随机选择暗帧
        selected_dark_frame = np.random.choice(dark_frames)
        
        # 确保尺寸匹配
        if selected_dark_frame.shape != image_shape[-2:]:  # 假设image_shape是(C, H, W)
            # 进行裁剪或填充以匹配目标尺寸
            selected_dark_frame = self._resize_dark_frame(selected_dark_frame, image_shape[-2:])
        
        # 转换为torch tensor
        noise = torch.from_numpy(selected_dark_frame.copy()).float()
        
        # 如果需要多通道，复制到所有通道
        if len(image_shape) == 3 and image_shape[0] > 1:
            noise = noise.unsqueeze(0).repeat(image_shape[0], 1, 1)
        
        return noise
    
    def _resize_dark_frame(self, dark_frame, target_shape):
        """
        调整暗帧尺寸以匹配目标图像
        """
        h, w = target_shape
        fh, fw = dark_frame.shape
        
        # 简单的中心裁剪或填充
        if fh >= h and fw >= w:
            # 中心裁剪
            start_h = (fh - h) // 2
            start_w = (fw - w) // 2
            return dark_frame[start_h:start_h+h, start_w:start_w+w]
        else:
            # 填充（这种情况较少见）
            padded = np.zeros((h, w), dtype=dark_frame.dtype)
            start_h = (h - fh) // 2
            start_w = (w - fw) // 2
            padded[start_h:start_h+fh, start_w:start_w+fw] = dark_frame
            return padded
    
    def _fallback_noise_sampling(self, image_shape, iso):
        """
        备选噪声采样方法（当暗帧不可用时）
        """
        noise_std = np.sqrt(iso / 100.0) * 0.1  # 简化的ISO依赖模型
        return torch.normal(mean=torch.zeros(image_shape), std=noise_std)


if __name__ == '__main__':
    trainer = RawDenoising_Trainer()
    
    if trainer.mode == 'train':
        trainer.train()
        
        # 保存训练历史
        savefile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.jpg')
        logfile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.pkl')
        trainer.train_psnr.plot_history(savefile=savefile, logfile=logfile)
        trainer.eval_psnr.plot_history(savefile=os.path.join(trainer.sample_dir, f'{trainer.model_name}_eval_psnr.jpg'))
        trainer.mode = 'evaltest'
    
    # 加载最佳模型进行评估
    best_model_path = os.path.join(f'{trainer.fast_ckpt}', f'{trainer.model_name}_best_model.pth')
    if os.path.exists(best_model_path) is False: 
        best_model_path = os.path.join(f'{trainer.fast_ckpt}',f'{trainer.model_name}_last_model.pth')
    
    if os.path.exists(best_model_path):
        best_model = torch.load(best_model_path, map_location=trainer.device)
        
        # 加载模型权重
        if isinstance(best_model, dict) and 'model' in best_model:
            model_weights = best_model['model']
            log(f"加载新格式模型权重用于评估")
        else:
            model_weights = best_model
            log(f"加载旧格式模型权重用于评估")
        
        trainer.net = load_weights(trainer.net, model_weights, trainer.multi_gpu, by_name=True)
        log(f'Successfully loaded model for evaluation', log=f'./logs/log_{trainer.model_name}.log')
        
        # 执行评估
        trainer.eval()
    else:
        log(f'Model file not found: {best_model_path}', log=f'./logs/log_{trainer.model_name}.log')