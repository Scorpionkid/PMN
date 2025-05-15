# 在项目根目录创建test_depthwise.py

import torch
import sys
import copy
sys.path.append('.')

# 导入验证函数
from archs.depthwise_separable_conv import (
    verify_replacement, 
    replace_conv3x3_simple, 
    count_parameters,
    get_model_param_summary
)
from archs.dualPathNet_sharedEnc_arch import DualPathUNet_E1

def test_depthwise_conversion():
    """使用完整验证函数测试深度可分离卷积转换"""
    print("=== 使用verify_replacement测试深度可分离卷积转换 ===")
    
    # 定义模型配置（使用和trainer_SID.py中相同的配置）
    args = {
        'nf': 64,
        'in_channels': 4,
        'out_channels': 4,
        'heads': [1, 2, 4, 8],
        'use_wavelet_upsample': True,
        'use_sharpness_recovery': True,
        'use_noise_map': True,
        'use_texture_detection': True,
        'enable_intermediate_supervision': True
    }
    
    try:
        # 1. 创建原始模型
        print("1. 创建原始模型...")
        original_model = DualPathUNet_E1(args)
        
        # 2. 创建副本用于替换
        print("2. 创建模型副本...")
        modified_model = copy.deepcopy(original_model)
        
        # 3. 替换3x3卷积
        print("3. 替换3x3卷积为深度可分离卷积...")
        replace_conv3x3_simple(modified_model)
        
        # 4. 使用完整的验证函数
        print("4. 开始完整验证...")
        success = verify_replacement(original_model, modified_model)
        
        # 5. 额外的DualPathNet特定测试
        print("\n5. DualPathNet特定测试...")
        test_input = torch.randn(1, 4, 128, 128)
        noise_map = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            # 测试原始模型
            orig_output = original_model(test_input, noise_map)
            print(f"✓ 原始模型输出: {type(orig_output)}")
            
            # 测试修改后模型
            mod_output = modified_model(test_input, noise_map)
            print(f"✓ 修改后模型输出: {type(mod_output)}")
            
            # 检查输出格式一致性
            if isinstance(orig_output, tuple) and isinstance(mod_output, tuple):
                print(f"✓ 两个模型都返回元组，长度: {len(orig_output)} vs {len(mod_output)}")
                if len(orig_output) == len(mod_output):
                    for i, (orig, mod) in enumerate(zip(orig_output, mod_output)):
                        if orig is not None and mod is not None:
                            print(f"  - 输出{i}形状: {orig.shape} vs {mod.shape}")
            
        return success
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 删除了不必要的test_with_trainer函数

if __name__ == "__main__":
    print("🔍 开始深度可分离卷积测试...")
    
    # 主要测试
    success_main = test_depthwise_conversion()
    
    print("\n" + "="*50)
    if success_main:
        print("🎉 测试通过！深度可分离卷积集成成功！")
        print("💡 现在你可以在配置文件中设置 use_depthwise_separable: true")
        print("💡 然后运行 python trainer_SID.py 进行训练")
    else:
        print("❌ 测试失败，请检查代码和配置")
        
    print("="*50)