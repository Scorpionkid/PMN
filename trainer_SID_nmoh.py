"""
基于NMOH论文的简化训练器
继承原有SID_Trainer，仅融入NMOH的核心思想：
1. 简化的Dark Shading处理（每个ISO只需10个暗帧）
2. 使用假设的系统增益（无需精确标定）
3. 保持原有的训练-评估-测试流程不变
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer_SID import SID_Trainer  # 继承正确的训练器
from data_process.real_datasets_nmoh import SID_Dataset_NMOH, ELD_Dataset_NMOH
from utils import log, load_weights
import pickle as pkl


class SID_Trainer_NMOH(SID_Trainer):
    """基于NMOH的简化训练器，保持原有框架"""
    
    def __init__(self, args=None):
        # 添加NMOH标记到配置
        if args is not None:
            self.add_nmoh_config(args)
        
        # 调用父类初始化
        super().__init__(args)
        
        # 初始化NMOH特定设置
        self.init_nmoh_settings()
        
        # 如果需要，重新设置数据集为NMOH版本
        if hasattr(self, 'dst_train'):
            self.setup_nmoh_dataset()
        
        log('=' * 50)
        log('NMOH Simplified Training Mode Enabled')
        log(f'NMOH Config: {self.nmoh_config}')
        log('=' * 50)
    
    def add_nmoh_config(self, args):
        """在配置中添加NMOH设置"""
        # 为所有数据集配置添加nmoh标记
        for dataset_key in ['dst_train', 'dst_eval', 'dst_test']:
            if dataset_key in args:
                if 'command' in args[dataset_key]:
                    # 替换darkshading为darkshading_nmoh
                    command = args[dataset_key]['command']
                    if 'darkshading' in command and 'darkshading_nmoh' not in command:
                        command = command.replace('darkshading', 'darkshading_nmoh')
                    if 'nmoh' not in command:
                        command += ', nmoh'
                    args[dataset_key]['command'] = command
                else:
                    args[dataset_key]['command'] = 'nmoh'
    
    def init_nmoh_settings(self):
        """初始化NMOH特定设置"""
        self.nmoh_config = {
            # 基于NMOH论文的配置
            'use_hypothesized_k': True,  # 使用假设的系统增益
            'qe_hypothesis': 0.5,  # 假设的量子效率（30%-70%范围）
            'skip_precise_calibration': True,  # 跳过精确标定
            'minimal_dark_frames': 10,  # 最少暗帧数
            'enable_fast_sna': True,  # 快速SNA
            'enable_simple_dsc': True,  # 简化DSC
        }
        
        # 记录NMOH统计信息
        self.nmoh_stats = {
            'dark_frames_used': {},
            'preparation_time': 0,
            'calibration_skipped': True
        }
    
    def setup_nmoh_dataset(self):
        """替换数据集为NMOH版本"""
        log('Replacing datasets with NMOH versions...')
        
        # 替换训练数据集
        if hasattr(self, 'dst') and hasattr(self, 'dst_train'):
            if self.dst['dataset'] == 'SID_Dataset':
                self.dst_train = SID_Dataset_NMOH(self.dst)
            elif self.dst['dataset'] == 'ELD_Dataset':
                self.dst_train = ELD_Dataset_NMOH(self.dst)
            elif self.dst['dataset'] == 'Mix_Dataset':
                # 如果有混合数据集，也使用NMOH版本
                self.dst_train = SID_Dataset_NMOH(self.dst)  # 暂时用SID代替
            
            # 重新创建dataloader
            self.dataloader_train = DataLoader(
                self.dst_train,
                batch_size=self.args['hyper']['batch_size'],
                shuffle=True,
                num_workers=self.args['num_workers'],
                pin_memory=True,
                drop_last=True
            )
            
            # 记录NMOH暗帧使用情况
            if hasattr(self.dst_train, 'dark_frames_nmoh'):
                for iso, frames in self.dst_train.dark_frames_nmoh.items():
                    self.nmoh_stats['dark_frames_used'][iso] = len(frames)
                log(f'NMOH dark frames loaded: {self.nmoh_stats["dark_frames_used"]}')
            
            log(f'Train dataset replaced with NMOH version, size: {len(self.dst_train)}')
    
    def change_eval_dst(self, mode='eval'):
        """重写评估数据集切换，使用NMOH版本"""
        # 先调用父类方法
        super().change_eval_dst(mode)
        
        # 替换为NMOH版本
        self.dst = self.args[f'dst_{mode}']
        self.dstname = self.dst['dstname']
        
        if self.dst['dataset'] == 'SID_Dataset':
            self.dst_eval = SID_Dataset_NMOH(self.dst)
        elif self.dst['dataset'] == 'ELD_Dataset':
            self.dst_eval = ELD_Dataset_NMOH(self.dst)
        else:
            self.dst_eval = SID_Dataset_NMOH(self.dst)
        
        self.dataloader_eval = DataLoader(
            self.dst_eval, 
            batch_size=1, 
            shuffle=False,
            num_workers=self.args['num_workers'], 
            pin_memory=False
        )
        
        log(f'Eval dataset changed to {self.dstname} (NMOH version)')
    
    def preprocess(self, noisy, gt, infos):
        """
        重写预处理函数，使用NMOH简化的噪声模型
        保持原有的噪声图生成方式，但使用假设的系统增益
        """
        iso_list = infos['ISO']
        ratio = infos['ratio']
        wb = infos.get('wb', None)
        
        # 数据归一化等预处理（保持原有逻辑）
        gt = gt / self.wp
        noisy = noisy / self.wp
        
        # 数据增强（如果需要，保持原有逻辑）
        if self.args['dst_train'].get('augment', False):
            # 保持原有的数据增强逻辑
            pass
        
        # NMOH简化的噪声图生成
        if self.arch['use_noise_map']:
            noise_map = self.generate_noise_map_nmoh(noisy, iso_list)
        else:
            noise_map = None
        
        # 如果需要生成合成噪声（用于SNA）
        if self.training and np.random.rand() < 0.5:
            # 50%概率使用合成噪声增强
            noisy_syn = self.synthesize_noise_nmoh(gt, iso_list, ratio)
            # 混合真实和合成噪声
            alpha = np.random.uniform(0.3, 0.7)
            noisy = alpha * noisy + (1 - alpha) * noisy_syn
        
        return noisy, gt, noise_map
    
    def generate_noise_map_nmoh(self, noisy, iso_list):
        """
        NMOH简化的噪声图生成
        使用假设的量子效率，无需精确标定
        """
        batch_size = noisy.shape[0]
        device = noisy.device
        
        # NMOH: 使用假设的量子效率
        qe = self.nmoh_config['qe_hypothesis']
        
        noise_maps = []
        for i in range(batch_size):
            iso = iso_list[i].item() if torch.is_tensor(iso_list[i]) else iso_list[i]
            
            # 假设的系统增益K
            k_hypothesis = qe * iso / 65535.0  # 归一化
            
            # 简化的噪声模型：sqrt(K * signal + read_noise^2)
            read_noise = 10.0 / 65535.0  # 归一化的读噪声估计
            
            # 生成噪声图
            signal = torch.clamp(noisy[i], min=0)
            noise_std = torch.sqrt(k_hypothesis * signal + read_noise**2)
            
            # 取4个通道的平均作为噪声图（单通道）
            noise_map = noise_std.mean(dim=0, keepdim=True)
            noise_maps.append(noise_map)
        
        return torch.stack(noise_maps, dim=0)
    
    def synthesize_noise_nmoh(self, clean, iso_list, ratio):
        """
        NMOH简化的噪声合成（Shot Noise Augmentation）
        """
        batch_size = clean.shape[0]
        device = clean.device
        
        # NMOH: 使用假设的量子效率
        qe = self.nmoh_config['qe_hypothesis']
        
        noisy_syn = torch.zeros_like(clean)
        
        for i in range(batch_size):
            iso = iso_list[i].item() if torch.is_tensor(iso_list[i]) else iso_list[i]
            clean_i = clean[i]
            
            # 假设的系统增益K
            k_hypothesis = qe * iso
            
            # 添加泊松噪声（shot noise）
            # 转换到原始尺度
            clean_scaled = clean_i * self.wp
            
            # 泊松噪声
            shot_noise = torch.poisson(clean_scaled * k_hypothesis) / k_hypothesis - clean_scaled
            
            # 归一化回来
            noisy_i = (clean_scaled + shot_noise) / self.wp
            
            # 简单的读噪声
            read_noise_std = 10.0 / self.wp
            read_noise = torch.randn_like(noisy_i) * read_noise_std
            noisy_i = noisy_i + read_noise
            
            # Clip到有效范围
            noisy_syn[i] = torch.clamp(noisy_i, 0, 1)
        
        return noisy_syn
    
    def train(self):
        """保持原有的训练流程，只在必要处融入NMOH"""
        # 调用父类的训练方法
        super().train()
    
    def eval(self):
        """保持原有的评估流程"""
        # 调用父类的评估方法
        super().eval()
    
    def test(self):
        """保持原有的测试流程"""
        # 调用父类的测试方法
        super().test()
    
    def save_nmoh_stats(self, epoch):
        """保存NMOH相关统计信息"""
        stats = {
            'epoch': epoch,
            'nmoh_config': self.nmoh_config,
            'dark_frames_used': self.nmoh_stats['dark_frames_used'],
            'preparation_time': self.nmoh_stats.get('preparation_time', 0),
            'performance': {
                'psnr': self.best_psnr if hasattr(self, 'best_psnr') else 0,
                'ssim': self.best_ssim if hasattr(self, 'best_ssim') else 0
            }
        }
        
        # 保存到文件
        save_path = os.path.join(self.fast_ckpt, f'{self.model_name}_nmoh_stats_ep{epoch}.pkl')
        with open(save_path, 'wb') as f:
            pkl.dump(stats, f)
        
        log(f'NMOH stats saved to {save_path}')
    
    def print_nmoh_stats(self):
        """打印NMOH统计信息"""
        log('=' * 50)
        log('NMOH Statistics:')
        log(f'Dark frames per ISO: {self.nmoh_stats.get("dark_frames_used", {})}')
        log(f'Calibration skipped: {self.nmoh_stats.get("calibration_skipped", True)}')
        log(f'QE hypothesis: {self.nmoh_config["qe_hypothesis"]}')
        log('=' * 50)


if __name__ == '__main__':
    # 保持与原有trainer_SID.py相同的执行流程
    trainer = SID_Trainer_NMOH()  # 自动从配置文件读取设置
    
    # 打印NMOH统计信息
    trainer.print_nmoh_stats()
    
    if trainer.mode == 'train':
        trainer.train()
        savefile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.jpg')
        logfile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.pkl')
        trainer.train_psnr.plot_history(savefile=savefile, logfile=logfile)
        trainer.eval_psnr.plot_history(savefile=os.path.join(trainer.sample_dir, f'{trainer.model_name}_eval_psnr.jpg'))
        
        # 保存NMOH统计（训练结束后）
        trainer.save_nmoh_stats(trainer.current_epoch)
        
        trainer.mode = 'evaltest'
    
    # best_model
    best_model_path = os.path.join(f'{trainer.fast_ckpt}', f'{trainer.model_name}_last_model.pth')
    best_model = torch.load(best_model_path, map_location=trainer.device)
    
    # 检查加载的文件是新格式还是旧格式
    if isinstance(best_model, dict) and 'model' in best_model:
        # 新格式：包含完整训练状态
        model_weights = best_model['model']
        log(f"加载新格式模型权重用于评估 (NMOH)")
        log(f"Epoch{best_model['epoch']}, Best_PSNR{best_model['eval_psnr']['avg']}")
    else:
        # 旧格式：仅包含模型权重
        model_weights = best_model
        log(f"加载旧格式模型权重用于评估 (NMOH)")
    
    trainer.net = load_weights(trainer.net, model_weights, multi_gpu=trainer.multi_gpu)
    
    if 'eval' in trainer.mode:
        # ELD
        trainer.change_eval_dst('eval')
        for dgain in trainer.args['dst_eval']['ratio_list']:
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            log(f'ELD Datasets (NMOH): Dgain={dgain}', log=f'./logs/log_{trainer.model_name}_nmoh.log')
            trainer.dst_eval.ratio_list=[dgain]
            trainer.dst_eval.recheck_length()
            metrics = trainer.eval(-1)
    
    if 'test' in trainer.mode:
        # SID
        trainer.change_eval_dst('test')
        SID_ratio_list = [100, 250, 300]
        for dgain in SID_ratio_list:
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            log(f'SID Datasets (NMOH): Dgain={dgain}', log=f'./logs/log_{trainer.model_name}_nmoh.log')
            trainer.dst_eval.change_eval_ratio(ratio=dgain)
            metrics = trainer.eval(-1)
    
    log(f'Metrics have been saved in ./metrics/{trainer.model_name}_metrics.pkl')
    log('NMOH Training/Evaluation completed!')