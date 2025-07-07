"""
ELD微调训练脚本
极简版本，最小化修改，主要负责加载预训练模型和微调设置
"""
import os
import torch
from trainer_SID import SID_Trainer

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
            print(f"开始ELD微调训练，数据量: {len(trainer.dst_train)}")
            trainer.train()  # 直接使用父类的train方法
        else:
            print("开始评估")
            trainer.eval(-1)  # 直接使用父类的eval方法