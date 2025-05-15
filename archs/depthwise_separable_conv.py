import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积实现"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, 
                 stride=1, dilation=1, bias=False, activation=None):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积：对每个通道独立进行卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        
        # 逐点卷积：1x1卷积，用于通道间信息交互
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
        # 可选的激活函数
        self.activation = activation
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_param_summary(model):
    """获取模型参数详细信息"""
    param_count = {}
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, DepthwiseSeparableConv)):
            if isinstance(module, nn.Conv2d):
                params = sum(p.numel() for p in module.parameters())
                param_count[name] = {
                    'type': 'Conv2d',
                    'params': params,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size
                }
            else:  # DepthwiseSeparableConv
                params = sum(p.numel() for p in module.parameters())
                param_count[name] = {
                    'type': 'DepthwiseSeparableConv',
                    'params': params,
                    'in_channels': module.depthwise.in_channels,
                    'out_channels': module.pointwise.out_channels,
                    'kernel_size': module.depthwise.kernel_size
                }
            total_params += params
    
    return param_count, total_params

def replace_conv3x3_with_depthwise(module, preserve_activation=True):
    """
    递归替换模型中的3x3卷积为深度可分离卷积
    
    Args:
        module: 要替换的模块
        preserve_activation: 是否保留原有的激活函数结构
    """
    replaced_count = 0
    
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            # 只替换3x3卷积，保留1x1和其他尺寸的卷积
            if child.kernel_size == (3, 3):
                # 检查是否有对应的激活函数
                activation = None
                if preserve_activation:
                    # 查找可能的激活函数（通常紧跟在卷积后面）
                    module_list = list(module.named_children())
                    current_idx = next(i for i, (n, _) in enumerate(module_list) if n == name)
                    if current_idx + 1 < len(module_list):
                        next_name, next_module = module_list[current_idx + 1]
                        if isinstance(next_module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Tanh)):
                            activation = next_module
                
                # 创建深度可分离卷积替换
                depthwise_conv = DepthwiseSeparableConv(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size[0],
                    padding=child.padding[0],
                    stride=child.stride[0],
                    dilation=child.dilation[0],
                    bias=child.bias is not None,
                    activation=activation if preserve_activation else None
                )
                
                # 可选：复制原有的权重（需要特殊处理）
                # 这里只创建新的参数，实际项目中可以考虑权重迁移
                
                setattr(module, name, depthwise_conv)
                replaced_count += 1
                
                # 如果保留了激活函数在depthwise conv中，移除原来的激活函数
                if preserve_activation and activation is not None:
                    delattr(module, next_name)
        else:
            # 递归处理子模块
            replaced_count += replace_conv3x3_with_depthwise(child, preserve_activation)
    
    return replaced_count

def verify_replacement(original_model, modified_model):
    """验证替换是否成功"""
    print("=== 模型替换验证 ===")
    
    # 1. 参数量对比
    orig_params = count_parameters(original_model)
    mod_params = count_parameters(modified_model)
    
    print(f"原始模型参数量: {orig_params:,}")
    print(f"修改后模型参数量: {mod_params:,}")
    print(f"参数减少量: {orig_params - mod_params:,}")
    print(f"参数减少比例: {(orig_params - mod_params) / orig_params * 100:.2f}%")
    
    # 2. 详细参数统计
    print("\n=== 详细参数统计 ===")
    orig_summary, _ = get_model_param_summary(original_model)
    mod_summary, _ = get_model_param_summary(modified_model)
    
    # 统计不同类型卷积的数量
    orig_conv_count = sum(1 for info in orig_summary.values() if info['type'] == 'Conv2d')
    mod_conv_count = sum(1 for info in mod_summary.values() if info['type'] == 'Conv2d')
    mod_dwconv_count = sum(1 for info in mod_summary.values() if info['type'] == 'DepthwiseSeparableConv')
    
    print(f"原始模型 - 标准卷积层数: {orig_conv_count}")
    print(f"修改后模型 - 标准卷积层数: {mod_conv_count}")
    print(f"修改后模型 - 深度可分离卷积层数: {mod_dwconv_count}")
    
    # 3. 前向传播测试
    print("\n=== 前向传播测试 ===")
    test_input = torch.randn(1, 4, 64, 64)  # 假设输入是4通道
    
    try:
        with torch.no_grad():
            orig_output = original_model(test_input)
            mod_output = modified_model(test_input)
            
        print("前向传播测试通过 ✓")
        print(f"原始输出形状: {orig_output.shape if isinstance(orig_output, torch.Tensor) else 'Tuple of tensors'}")
        print(f"修改后输出形状: {mod_output.shape if isinstance(mod_output, torch.Tensor) else 'Tuple of tensors'}")
        
    except Exception as e:
        print(f"前向传播测试失败: {e}")
        return False
    
    return True

# 计算深度可分离卷积vs标准卷积的参数差异
def calculate_param_difference(in_channels, out_channels, kernel_size=3):
    """计算标准卷积和深度可分离卷积的参数差异"""
    # 标准卷积参数量
    standard_params = in_channels * out_channels * kernel_size * kernel_size
    
    # 深度可分离卷积参数量  
    depthwise_params = in_channels * kernel_size * kernel_size  # 深度卷积
    pointwise_params = in_channels * out_channels  # 逐点卷积
    depthwise_total = depthwise_params + pointwise_params
    
    compression_ratio = standard_params / depthwise_total
    
    return {
        'standard_params': standard_params,
        'depthwise_params': depthwise_total,
        'compression_ratio': compression_ratio,
        'param_reduction': standard_params - depthwise_total
    }