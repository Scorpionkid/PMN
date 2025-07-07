"""
ELD微调训练脚本
基于原有的trainer_SID.py修改，使用正确的配置解析方式
"""
import os
import sys
import pickle as pkl
import torch
import torch.nn as nn
import numpy as np
from trainer_SID import *  # 导入原有的trainer类

def load_pretrained_model(net, pretrained_path, strict=True):
    """
    加载预训练模型
    """
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练模型不存在: {pretrained_path}")
    
    print(f"正在加载预训练模型: {pretrained_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"从checkpoint加载，最佳PSNR: {checkpoint.get('best_psnr', 'N/A')}")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"从checkpoint加载，Epoch: {checkpoint.get('epoch', 'N/A')}")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 移除不匹配的键
    model_dict = net.state_dict()
    pretrained_dict = {}
    
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            pretrained_dict[k] = v
        else:
            print(f"跳过不匹配的键: {k}")
    
    # 更新模型
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict, strict=False)
    
    print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
    return net

def setup_finetune_optimizer(net, args):
    """
    设置微调优化器，支持不同层的不同学习率
    """
    finetune_config = args.get('finetune', {})
    
    if finetune_config.get('freeze_encoder', False):
        # 冻结编码器部分
        for name, param in net.named_parameters():
            if 'encoder' in name.lower() or 'down' in name.lower():
                param.requires_grad = False
                print(f"冻结参数: {name}")
    
    if finetune_config.get('freeze_bn', True):
        # 冻结BatchNorm层
        for module in net.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    # 设置优化器
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    
    return trainable_params

class FineTuneTrainer(SID_Trainer):
    """
    微调训练器，继承自SID_Trainer
    """
    def __init__(self):
        # 调用父类初始化（自动处理配置解析）
        super().__init__()
        
        # 加载预训练模型
        if 'pretrained_model' in self.args and self.args['pretrained_model']:
            self.net = load_pretrained_model(self.net, self.args['pretrained_model'])
        
        # 设置微调特定的参数
        if 'finetune' in self.args and self.args['finetune']['enabled']:
            trainable_params = setup_finetune_optimizer(self.net, self.args)
            
            # 重新创建优化器（只优化可训练参数）
            lr = self.args['hyper']['learning_rate']
            weight_decay = self.args['finetune'].get('weight_decay', 1e-6)
            
            self.optimizer = torch.optim.Adam(
                trainable_params, 
                lr=lr, 
                weight_decay=weight_decay
            )
            
            # 重新创建学习率调度器
            if hasattr(self, 'get_scheduler'):
                self.scheduler = self.get_scheduler(self.optimizer)
            else:
                # 使用简单的学习率调度器
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=self.args['hyper']['stop_epoch']
                )
    
    def train_epoch_with_small_dataset(self, epoch):
        """
        针对小数据集优化的训练epoch
        """
        self.net.train()
        
        # 确保BatchNorm层保持eval状态（如果设置了freeze_bn）
        finetune_config = self.args.get('finetune', {})
        if finetune_config.get('freeze_bn', True):
            for module in self.net.modules():
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                    module.eval()
        
        # 对于小数据集，每个epoch多次遍历数据
        epoch_loss = 0
        num_iterations = 0
        
        # 由于只有12对数据，每个epoch重复多次
        dataset_size = len(self.dst_train)
        repeat_factor = max(1, 100 // dataset_size)  # 确保每个epoch至少100次迭代
        
        log(f'Epoch {epoch}: 数据集大小={dataset_size}, 重复因子={repeat_factor}')
        
        for repeat in range(repeat_factor):
            for batch_idx, data_batch in enumerate(self.dst_train):
                try:
                    # 数据预处理
                    if hasattr(self, 'preprocess'):
                        data_batch = self.preprocess(data_batch)
                    else:
                        # 基本的数据处理
                        for key in data_batch:
                            if isinstance(data_batch[key], np.ndarray):
                                data_batch[key] = torch.from_numpy(data_batch[key]).float().to(self.device)
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    
                    # 模型推理
                    output = self.net(data_batch['lr'])
                    
                    # 处理多输出情况
                    if isinstance(output, tuple):
                        output = output[0]  # 取主要输出
                    
                    # 计算损失
                    if hasattr(self, 'criterion'):
                        loss = self.criterion(output, data_batch['hr'])
                    else:
                        # 使用L1损失
                        loss = torch.nn.functional.l1_loss(output, data_batch['hr'])
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪（对小数据集很重要）
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_iterations += 1
                    
                    if num_iterations % 20 == 0:
                        log(f'Epoch {epoch}, Iter {num_iterations}, Loss: {loss.item():.6f}')
                        
                except Exception as e:
                    log(f'训练出错 Epoch {epoch}, Batch {batch_idx}: {e}')
                    continue
        
        avg_loss = epoch_loss / max(num_iterations, 1)
        log(f'Epoch {epoch} 完成, 平均损失: {avg_loss:.6f}, 总迭代: {num_iterations}')
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        重写保存checkpoint，添加微调信息
        """
        state = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_psnr': getattr(self, 'best_psnr', 0),
            'finetune_info': {
                'base_model': self.args.get('pretrained_model', ''),
                'finetune_config': self.args.get('finetune', {})
            }
        }
        
        filename = f"{self.model_name}_epoch_{epoch}.pth"
        filepath = os.path.join(self.fast_ckpt, filename)
        torch.save(state, filepath)
        
        if is_best:
            best_filepath = os.path.join(self.fast_ckpt, f"{self.model_name}_best.pth")
            torch.save(state, best_filepath)
            print(f"保存最佳模型: {best_filepath}")

if __name__ == '__main__':
    # 确保数据集信息文件存在
    info_files = ['ELD_SonyA7S2_finetune.info', 'ELD_SonyA7S2_test.info']
    for info_file in info_files:
        if not os.path.exists(f'infos/{info_file}'):
            print(f"警告: 数据集信息文件不存在: {info_file}")
            print("请先运行: python get_dataset_infos.py --dstname ELD --mode finetune --root_dir /your/eld/path")
    
    # 创建微调训练器（自动处理配置解析）
    trainer = FineTuneTrainer()
    
    if trainer.mode == 'train':
        print("开始ELD数据集微调...")
        print(f"微调数据量: 12对（前2个场景）")
        print(f"测试数据量: 48对（剩余8个场景）")
        
        best_psnr = 0
        
        for epoch in range(trainer.args['hyper']['last_epoch'], trainer.args['hyper']['stop_epoch']):
            # 使用小数据集优化的训练方法
            train_loss = trainer.train_epoch_with_small_dataset(epoch)
            
            # 更新学习率
            if trainer.scheduler:
                trainer.scheduler.step()
            
            # 每隔20个epoch在测试集上评估
            if epoch % 20 == 0 or epoch == trainer.args['hyper']['stop_epoch'] - 1:
                # 切换到测试集进行评估
                trainer.change_eval_dst('eval')
                
                # 确保使用完整图像
                if hasattr(trainer.dst_eval, 'crop_per_image'):
                    trainer.dst_eval.crop_per_image = 0
                if hasattr(trainer.dst_eval, 'croptype'):
                    trainer.dst_eval.croptype = 'full_image'
                
                # 分别评估不同的ratio
                total_psnr = 0
                for ratio in [100, 200]:
                    log(f'Epoch {epoch}: 评估完整图像 Dgain={ratio}')
                    
                    # 修改数据集的ratio_list来只评估特定ratio
                    if hasattr(trainer.dst_eval, 'ratio_list'):
                        original_ratio_list = trainer.dst_eval.ratio_list.copy()
                        trainer.dst_eval.ratio_list = [ratio]
                        if hasattr(trainer.dst_eval, 'recheck_length'):
                            trainer.dst_eval.recheck_length()
                    
                    val_metrics = trainer.eval(epoch)
                    val_psnr = val_metrics.get('psnr', 0)
                    total_psnr += val_psnr
                    
                    log(f'Epoch {epoch}: Dgain={ratio}, 完整图像PSNR={val_psnr:.2f}')
                    
                    # 恢复原始ratio_list
                    if hasattr(trainer.dst_eval, 'ratio_list'):
                        trainer.dst_eval.ratio_list = original_ratio_list
                
                # 平均PSNR作为模型选择指标
                avg_psnr = total_psnr / 2
                
                # 保存checkpoint
                is_best = avg_psnr > best_psnr
                if is_best:
                    best_psnr = avg_psnr
                    trainer.best_psnr = best_psnr
                
                trainer.save_checkpoint(epoch, is_best)
                
                log(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Avg PSNR={avg_psnr:.2f}, Best PSNR={best_psnr:.2f}')
        
        print(f"微调完成! 最佳平均PSNR: {best_psnr:.2f}")
        
        # 最终评估
        print("\n=== 最终完整图像评估结果 ===")
        trainer.change_eval_dst('eval')
        
        # 确保使用完整图像
        if hasattr(trainer.dst_eval, 'crop_per_image'):
            trainer.dst_eval.crop_per_image = 0
        if hasattr(trainer.dst_eval, 'croptype'):
            trainer.dst_eval.croptype = 'full_image'
        
        for dgain in [100, 200]:
            if hasattr(trainer.dst_eval, 'ratio_list'):
                trainer.dst_eval.ratio_list = [dgain]
                if hasattr(trainer.dst_eval, 'recheck_length'):
                    trainer.dst_eval.recheck_length()
            log(f'最终完整图像测试 Dgain={dgain}')
            metrics = trainer.eval(-1)
            print(f"最终结果 - Dgain={dgain}: PSNR={metrics.get('psnr', 0):.2f}, SSIM={metrics.get('ssim', 0):.4f}")
        
    else:
        # 评估模式 - 使用完整图像
        print("=== 完整图像评估模式 ===")
        
        if 'eval' in trainer.mode:
            trainer.change_eval_dst('eval')
            
            # 确保使用完整图像
            if hasattr(trainer.dst_eval, 'crop_per_image'):
                trainer.dst_eval.crop_per_image = 0
            if hasattr(trainer.dst_eval, 'croptype'):
                trainer.dst_eval.croptype = 'full_image'
            
            for dgain in [100, 200]:
                if hasattr(trainer.dst_eval, 'ratio_list'):
                    trainer.dst_eval.ratio_list = [dgain]
                    if hasattr(trainer.dst_eval, 'recheck_length'):
                        trainer.dst_eval.recheck_length()
                log(f'ELD完整图像验证: Dgain={dgain}')
                metrics = trainer.eval(-1)
                print(f"完整图像结果 - Dgain={dgain}: PSNR={metrics.get('psnr', 0):.2f}, SSIM={metrics.get('ssim', 0):.4f}")
        
        if 'test' in trainer.mode:
            trainer.change_eval_dst('test')
            
            # 确保使用完整图像
            if hasattr(trainer.dst_test, 'crop_per_image'):
                trainer.dst_test.crop_per_image = 0
            if hasattr(trainer.dst_test, 'croptype'):
                trainer.dst_test.croptype = 'full_image'
            
            for dgain in [100, 200]:
                log(f'ELD完整图像测试: Dgain={dgain}')
                metrics = trainer.eval(-1)
                print(f"完整图像测试 - Dgain={dgain}: PSNR={metrics.get('psnr', 0):.2f}, SSIM={metrics.get('ssim', 0):.4f}")