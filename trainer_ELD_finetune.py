"""
ELD微调训练脚本
极简版本，最小化修改，主要负责加载预训练模型和微调设置
"""
import os
import torch
from trainer_SID import SID_Trainer
from utils import *

def load_pretrained_model(net, pretrained_path):
    """加载预训练模型"""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练模型不存在: {pretrained_path}")
    
    print(f"正在加载预训练模型: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # 处理不同checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 加载权重
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict, strict=False)
    
    print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
    return net

class FineTuneTrainer(SID_Trainer):
    """微调训练器，极简继承"""
    
    def __init__(self):
        # 调用父类初始化
        super().__init__()
        
        # 加载预训练模型
        if 'pretrained_model' in self.args and self.args['pretrained_model']:
            self.net = load_pretrained_model(self.net, self.args['pretrained_model'])
        
        # 微调设置
        self.setup_finetune()
    
    def setup_finetune(self):
        """设置微调参数"""
        if 'finetune' not in self.args or not self.args['finetune'].get('enabled', False):
            return
        
        finetune_config = self.args['finetune']
        
        # 冻结BatchNorm层
        if finetune_config.get('freeze_bn', True):
            for module in self.net.modules():
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
            print("已冻结BatchNorm层")
        
        # 冻结编码器（可选）
        if finetune_config.get('freeze_encoder', False):
            for name, param in self.net.named_parameters():
                if 'encoder' in name.lower() or 'down' in name.lower():
                    param.requires_grad = False
            print("已冻结编码器层")
        
        # 重新创建优化器（使用微调学习率和权重衰减）
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        lr = self.args['hyper']['learning_rate']
        weight_decay = finetune_config.get('weight_decay', 1e-6)
        
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        print(f"可训练参数: {sum(p.numel() for p in trainable_params)}")
        
        # 重新设置学习率调度器
        if self.args['hyper']['lr_scheduler'] == 'WarmupCosine':
            # 如果有WarmupCosine实现就用，否则用简单的
            try:
                from utils import get_cosine_schedule_with_warmup
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer, 10, self.args['hyper']['stop_epoch'])
            except:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.args['hyper']['stop_epoch'])
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args['hyper']['stop_epoch'])

if __name__ == '__main__':
    # 检查数据集信息文件
    info_files = ['ELD_SonyA7S2_finetune.info', 'ELD_SonyA7S2_test.info']
    for info_file in info_files:
        if not os.path.exists(f'infos/{info_file}'):
            print(f"警告: 数据集信息文件不存在: {info_file}")
            print("请先运行: python get_dataset_infos.py --dstname ELD --mode finetune")
            break
    else:
        # 创建并运行微调训练器
        trainer = FineTuneTrainer()
        if trainer.mode == 'train':
            trainer.train()
            trainer.mode = 'evaltest'
        
        # best_model
        best_model_path = os.path.join(f'{trainer.fast_ckpt}', f'{trainer.model_name}_best_model.pth')
        if os.path.exists(best_model_path) is False: 
            best_model_path = os.path.join(f'{trainer.fast_ckpt}',f'{trainer.model_name}_last_model.pth')
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
        if 'test' in trainer.mode:
            # ELD
            trainer.change_eval_dst('test')
            for dgain in trainer.args['dst_test']['ratio_list']:
                info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
                if os.path.exists(info_path):
                    with open(info_path,'rb') as f:
                        trainer.infos = pkl.load(f)
                log(f'ELD Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
                trainer.dst_eval.ratio_list=[dgain]
                trainer.dst_eval.recheck_length()
                metrics = trainer.eval(-1)

        log(f'Metrics have been saved in ./metrics/{trainer.model_name}_metrics.pkl')