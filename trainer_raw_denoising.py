#!/usr/bin/env python3
"""
完整版 Raw Image Denoising Trainer
基于PMN框架集成raw_image_denoising的噪声合成方法
包含完整的训练启动逻辑
"""

import os
import time
import torch
import numpy as np
import pickle as pkl
import scipy.io as sio
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
        # 使用论文精确参数：K = ISO/100 × 0.1
        self.quantum_efficiency = dst_config.get('quantum_efficiency', 0.1)
        
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
        修复版：正确的简化噪声合成公式
        
        物理过程：原始信号 → 光子噪声 → 系统增益放大 → 信号无关噪声 → 数字增益
        论文公式：D = Kd * (Ka(X + Np) + N2)
        其中：Kd是数字增益(ratio), Ka是系统增益, X是原始信号, Np是光子噪声, N2是信号无关噪声
        """
        device = clean_image.device
    
        # 🔧 关键修复1: 借鉴SNA的域转换策略
        # 获取相机参数（从配置中读取，或使用默认值）
        wp = torch.tensor(getattr(self.dst, 'wp', 16383), dtype=clean_image.dtype, device=device)
        bl = torch.tensor(getattr(self.dst, 'bl', 512), dtype=clean_image.dtype, device=device)
        
        if not torch.is_tensor(ratio):
            ratio = torch.tensor(ratio, dtype=clean_image.dtype, device=device)
        # 🔧 关键修复2: 转换到RAW域进行处理（类似SNA_torch）
        # 将归一化的clean_image转换到RAW域
        clean_raw = clean_image * (wp - bl) / ratio
        
        # 论文精确参数：K = ISO/100 × 0.1 - Ka是物理参数，不应该调整！
        system_gain = iso / 100.0 * self.quantum_efficiency  # Ka保持不变
        
        # 步骤1：基于RAW域信号计算光子散粒噪声
        # 在RAW域，signal_for_poisson的数值会自然增大到合理范围
        signal_for_poisson = torch.clamp(clean_raw / system_gain, min=1.0, max=50000.0)
        
        # 步骤2：生成泊松光子噪声
        try:
            poisson_samples = torch.poisson(signal_for_poisson)
            if torch.any(torch.isnan(poisson_samples)) or torch.any(torch.isinf(poisson_samples)):
                raise RuntimeError("Invalid poisson samples")
        except RuntimeError:
            # 回退到高斯近似
            poisson_samples = torch.normal(
                mean=signal_for_poisson,
                std=torch.sqrt(torch.clamp(signal_for_poisson, min=1e-6))
            )
        
        # 步骤3：应用系统增益 Ka(X + Np)
        signal_with_photon_noise = poisson_samples * system_gain
        
        # 步骤4：添加信号无关噪声 N2
        signal_independent_noise_raw = self.get_signal_independent_noise_torch(
            clean_image.shape, iso, dark_frame_paths, wp, bl
        )
        

        
        # 步骤5：在RAW域合成总噪声
        total_noise_raw = signal_with_photon_noise + signal_independent_noise_raw
        
        # 步骤6：应用数字增益（在RAW域）
        final_noisy_raw = clean_raw + total_noise_raw
        
        # 🔧 关键修复3: 转换回归一化域（类似SNA_torch的处理）
        # 将RAW域的结果转换回归一化域
        final_noisy_normalized = final_noisy_raw * ratio / (wp - bl)
        scaled_clean_normalized = clean_raw * ratio / (wp - bl)
        
        # 步骤7：计算增量，与PMN接口一致
        # PMN期望：imgs_lr = imgs_lr + dn, imgs_hr = imgs_hr + dy
        dn = final_noisy_normalized - scaled_clean_normalized 
        
        # 数值安全检查
        dn = torch.clamp(dn, min=-1e4, max=1e4)
        
        # 噪声参数
        noise_params = torch.tensor([iso, ratio], device=self.device)
        
        return dn, noise_params
    
    def get_signal_independent_noise_torch(self, image_shape, iso, dark_frame_paths, wp, bl):
        """
        获取信号无关噪声 - 直接暗帧采样（论文核心创新）
        """
        if dark_frame_paths and iso in dark_frame_paths and len(dark_frame_paths[iso]) > 0:
            # 随机选择一个暗帧文件
            selected_file = np.random.choice(dark_frame_paths[iso])
            
            try:
                # 加载.mat文件中的暗帧
                mat_data = sio.loadmat(selected_file)
                
                if 'Inoisy_crop' in mat_data:
                    # 保持数据为NumPy数组
                    dark_frame = mat_data['Inoisy_crop'].astype(np.float32)

                    # 🔧 关键修复：确保暗帧在RAW域
                    if dark_frame.max() <= 2:  # 如果已经归一化了，转换回RAW域
                        wp_np = wp.cpu().numpy()
                        bl_np = bl.cpu().numpy()
                        dark_frame = dark_frame * (wp_np - bl_np) + bl_np
                    
                    # 转换为PyTorch张量
                    dark_frame_tensor = torch.from_numpy(dark_frame).to(self.device)
                    
                    # 调整尺寸匹配
                    if len(image_shape) == 4:  # batch, channel, height, width
                        target_shape = image_shape[2:]  # height, width
                    else:  # channel, height, width
                        target_shape = image_shape[1:]
                    
                    # 调整暗帧尺寸（确保返回张量）
                    dark_frame_resized = self._resize_dark_frame_torch(dark_frame_tensor, target_shape)
                    
                    # 步骤1：归一化 - 去除直流分量
                    dark_frame_mean = dark_frame_resized.mean()  # 使用张量的mean方法
                    dark_frame_normalized = dark_frame_resized - dark_frame_mean  # 对张量进行操作
                
                    # 转换为正确的通道格式 (从HW变为CHW或BCHW)
                    if len(image_shape) == 4:  # batch
                        channels = image_shape[1]
                        if channels == 4:  # Bayer pattern
                            dark_frame_bayer = self._to_bayer_torch(dark_frame_normalized)
                        else:
                            dark_frame_bayer = dark_frame_normalized.unsqueeze(0).repeat(channels, 1, 1)
                        dark_frame_output = dark_frame_bayer.unsqueeze(0)  # Add batch dim
                    else:  # single image
                        channels = image_shape[0]
                        if channels == 4:  # Bayer pattern
                            dark_frame_output = self._to_bayer_torch(dark_frame_normalized)
                        else:
                            dark_frame_output = dark_frame_normalized.unsqueeze(0).repeat(channels, 1, 1)
                    
                    return dark_frame_output.to(self.device)
                
            except Exception as e:
                log(f"暗帧加载失败 {selected_file}: {e}")
        
        # 备选方案：统计噪声模型
        return self._fallback_statistical_noise_torch(image_shape, iso)
    
    def _resize_dark_frame_torch(self, dark_frame, target_shape):
        """使用PyTorch调整暗帧尺寸"""
        dark_tensor = dark_frame
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
                dn, p = self.simplified_noise_synthesis_pmn_compatible(
                    clean_images[i], 
                    iso=iso_list[i], 
                    ratio=ratio_list[i],
                    dark_frame_paths=dark_frame_paths
                )
                
                # 与PMN相同的增量应用方式
                # clean_images[i] = clean_images[i] + dy  # 更新清洁图像
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


# 完整的训练启动逻辑（与trainer_SID.py保持一致）
if __name__ == '__main__':
    print("Raw Image Denoising Trainer")
    print("=" * 50)
    
    trainer = RawDenoising_Trainer()
    # trainer.debug_dark_frame_values()
    
    if trainer.mode == 'train':
        trainer.train()
        savefile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.jpg')
        logfile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.pkl')
        trainer.train_psnr.plot_history(savefile=savefile, logfile=logfile)
        trainer.eval_psnr.plot_history(savefile=os.path.join(trainer.sample_dir, f'{trainer.model_name}_eval_psnr.jpg'))
        trainer.mode = 'evaltest'
    
    # 加载最佳模型
    best_model_path = os.path.join(f'{trainer.fast_ckpt}', f'{trainer.model_name}_best_model.pth')
    if os.path.exists(best_model_path) is False: 
        best_model_path = os.path.join(f'{trainer.fast_ckpt}',f'{trainer.model_name}_last_model.pth')
    
    if os.path.exists(best_model_path):
        best_model = torch.load(best_model_path, map_location=trainer.device)

        # 检查加载的文件是新格式还是旧格式
        if isinstance(best_model, dict) and 'model' in best_model:
            # 新格式：包含完整训练状态
            model_weights = best_model['model']
            log(f"加载新格式模型权重用于评估")
            log(f"Epoch{best_model['epoch']}, Best_PSNR{best_model['best_psnr']}")
        else:
            # 旧格式：仅包含模型权重
            model_weights = best_model
            log(f"加载旧格式模型权重用于评估")

        trainer.net = load_weights(trainer.net, model_weights, multi_gpu=trainer.multi_gpu)
        
        if 'eval' in trainer.mode:
            # ELD评估
            trainer.change_eval_dst('eval')
            for dgain in trainer.args['dst_eval']['ratio_list']:
                info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
                if os.path.exists(info_path):
                    with open(info_path,'rb') as f:
                        trainer.infos = pkl.load(f)
                log(f'ELD Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
                trainer.dst_eval.ratio_list=[dgain]
                trainer.dst_eval.recheck_length()
                metrics = trainer.eval(-1)

        if 'test' in trainer.mode:
            # SID评估
            trainer.change_eval_dst('test')
            SID_ratio_list = [100, 250, 300]
            for dgain in SID_ratio_list:
                info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
                if os.path.exists(info_path):
                    with open(info_path,'rb') as f:
                        trainer.infos = pkl.load(f)
                log(f'SID Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
                trainer.dst_eval.change_eval_ratio(ratio=dgain)
                metrics = trainer.eval(-1)
        
        log(f'Metrics have been saved in ./metrics/{trainer.model_name}_metrics.pkl')
    else:
        log(f"未找到训练好的模型: {best_model_path}")
        log("如果是首次训练，请确保训练模式设置正确")