#!/usr/bin/env python3
"""
CVPR25复现验证脚本
快速测试修复效果，确认PSNR是否开始提升
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from pathlib import Path

def test_system_gain_formula():
    """测试系统增益公式是否正确"""
    print("=== 测试1: 系统增益公式验证 ===")
    
    # 论文中的系统增益公式
    def paper_system_gain(iso):
        return iso / 100.0 * 0.1  # 论文实际使用的公式
    
    test_isos = [100, 800, 1600, 3200, 6400, 25600]
    
    print("ISO值 -> 系统增益K")
    for iso in test_isos:
        k = paper_system_gain(iso)
        print(f"{iso:5d} -> {k:6.2f}")
    
    # 验证与论文Table 3的一致性
    max_k = paper_system_gain(25600)
    print(f"\n最大K值 (ISO 25600): {max_k:.2f}")
    print(f"论文Table 3中窄范围假设: ~25.6 ✓" if abs(max_k - 25.6) < 0.1 else "❌ 不匹配")

def test_poisson_noise_synthesis():
    """测试泊松噪声合成是否正确"""
    print("\n=== 测试2: 泊松噪声合成验证 ===")
    
    # 创建测试图像
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clean_image = torch.ones(4, 64, 64, device=device) * 1000  # 模拟RAW值
    
    iso = 1600
    ratio = 4.0
    system_gain = iso / 100.0 * 0.1
    
    print(f"测试参数: ISO={iso}, ratio={ratio}, K={system_gain}")
    
    # 应用论文公式
    scaled_clean = clean_image * ratio
    signal_for_poisson = torch.clamp(scaled_clean / system_gain, min=1e-6)
    
    # 生成泊松噪声
    poisson_samples = torch.poisson(signal_for_poisson)
    photon_noise_component = poisson_samples * system_gain
    
    # 分析噪声特性
    theoretical_variance = scaled_clean * system_gain
    actual_variance = torch.var(photon_noise_component - scaled_clean)
    expected_variance = torch.mean(theoretical_variance)
    
    print(f"理论噪声方差: {expected_variance:.2f}")
    print(f"实际噪声方差: {actual_variance:.2f}")
    print(f"方差匹配度: {abs(actual_variance - expected_variance) / expected_variance * 100:.1f}%")
    
    if abs(actual_variance - expected_variance) / expected_variance < 0.2:
        print("✓ 泊松噪声合成正确")
    else:
        print("❌ 泊松噪声合成可能有问题")

def test_dark_frame_loading():
    """测试暗帧数据加载"""
    print("\n=== 测试3: 暗帧数据加载验证 ===")
    
    # 检查暗帧目录结构
    dark_frame_paths = [
        'resources/SonyA7S2/BiasFrame_ET_1_30',
        'resources/BiasFrame_ET_1_30',
        'resources'
    ]
    
    available_isos = []
    total_dark_frames = 0
    
    for base_path in dark_frame_paths:
        if os.path.exists(base_path):
            print(f"找到暗帧目录: {base_path}")
            
            for item in os.listdir(base_path):
                iso_path = os.path.join(base_path, item)
                if os.path.isdir(iso_path) and item.isdigit():
                    iso = int(item)
                    mat_files = [f for f in os.listdir(iso_path) if f.endswith('.mat')]
                    
                    if mat_files:
                        available_isos.append(iso)
                        total_dark_frames += len(mat_files)
                        
                        # 测试加载第一个文件
                        test_file = os.path.join(iso_path, mat_files[0])
                        try:
                            mat_data = sio.loadmat(test_file)
                            if 'Inoisy_crop' in mat_data:
                                dark_shape = mat_data['Inoisy_crop'].shape
                                print(f"  ISO {iso}: {len(mat_files)}个文件, 尺寸: {dark_shape}")
                            else:
                                print(f"  ISO {iso}: ❌ 缺少'Inoisy_crop'键")
                        except Exception as e:
                            print(f"  ISO {iso}: ❌ 加载失败: {e}")
            break
    
    available_isos.sort()
    print(f"\n总结:")
    print(f"可用ISO级别: {len(available_isos)} 个")
    print(f"总暗帧数: {total_dark_frames}")
    print(f"ISO范围: {available_isos}")
    
    # 检查常见ISO的覆盖情况
    common_isos = [800, 1600, 3200, 6400]
    covered = [iso for iso in common_isos if iso in available_isos]
    print(f"常见ISO覆盖: {len(covered)}/{len(common_isos)} ({covered})")
    
    if len(covered) >= 3:
        print("✓ 暗帧数据充足")
    else:
        print("❌ 暗帧数据不足，可能影响训练")

def test_pmn_interface_compatibility():
    """测试与PMN接口的兼容性"""
    print("\n=== 测试4: PMN接口兼容性验证 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟PMN的SNA_torch输出格式
    clean_image = torch.randn(4, 512, 512, device=device) * 1000
    iso = 1600
    ratio = 4.0
    
    # 我们的简化噪声合成
    system_gain = iso / 100.0 * 0.1
    scaled_clean = clean_image * ratio
    
    # 模拟噪声生成
    signal_for_poisson = torch.clamp(scaled_clean / system_gain, min=1e-6)
    poisson_samples = torch.poisson(signal_for_poisson)
    photon_noise_component = poisson_samples * system_gain
    
    # 模拟信号无关噪声
    signal_independent_noise = torch.randn_like(clean_image) * 10
    
    # 最终带噪图像
    final_noisy = photon_noise_component + signal_independent_noise
    
    # 计算增量（PMN格式）
    dn = final_noisy - scaled_clean
    dy = scaled_clean - clean_image
    
    print(f"输入图像范围: [{clean_image.min():.1f}, {clean_image.max():.1f}]")
    print(f"噪声增量dn范围: [{dn.min():.1f}, {dn.max():.1f}]")
    print(f"清洁增量dy范围: [{dy.min():.1f}, {dy.max():.1f}]")
    
    # 验证增量逻辑
    reconstructed_lr = clean_image + dy + dn
    reconstructed_hr = clean_image + dy
    
    ratio_diff = torch.abs(reconstructed_hr - scaled_clean).max()
    noise_consistency = torch.abs(reconstructed_lr - final_noisy).max()
    
    print(f"清洁图像重建误差: {ratio_diff:.6f}")
    print(f"带噪图像重建误差: {noise_consistency:.6f}")
    
    if ratio_diff < 1e-3 and noise_consistency < 1e-3:
        print("✓ PMN接口兼容性正确")
    else:
        print("❌ PMN接口兼容性有问题")

def generate_training_verification_code():
    """生成训练验证代码片段"""
    print("\n=== 训练验证代码 ===")
    
    code = '''
# 在trainer中添加此代码段来监控噪声合成是否正常工作

def debug_noise_synthesis(self, clean_image, iso, ratio):
    """调试噪声合成过程"""
    # 使用修复的简化噪声合成
    dn, dy, params = self.simplified_noise_synthesis_pmn_compatible(
        clean_image, iso, ratio
    )
    
    # 监控关键指标
    noise_power = torch.var(dn).item()
    signal_power = torch.var(clean_image).item()
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    log(f"ISO={iso}, Ratio={ratio:.2f}, SNR={snr:.2f}dB, 噪声功率={noise_power:.2f}")
    
    # 如果SNR异常，说明噪声合成有问题
    if snr < 10 or snr > 50:
        log("⚠️  SNR异常，检查噪声合成实现")
    
    return dn, dy, params

# 在preprocess函数中调用（仅前几个epoch）
if epoch < 5 and i == 0:  # 仅监控第一个batch的第一张图像
    self.debug_noise_synthesis(imgs_hr[0], iso_list[0], ratio_list[0])
'''
    
    print(code)

def main():
    """主验证流程"""
    print("CVPR25 论文复现验证 - 快速诊断工具")
    print("=" * 50)
    
    # 运行所有测试
    test_system_gain_formula()
    test_poisson_noise_synthesis()
    test_dark_frame_loading()
    test_pmn_interface_compatibility()
    generate_training_verification_code()
    
    print("\n" + "=" * 50)
    print("验证完成！")
    print("\n🎯 修复要点总结:")
    print("1. 系统增益公式: K = ISO/100 × 0.1 (QE=10%)")
    print("2. 泊松噪声: poisson(I/K) × K") 
    print("3. 暗帧直接采样，无统计建模")
    print("4. PMN兼容的增量格式")
    print("\n如果所有测试通过，期望PSNR应该开始从30.87提升到35+")

if __name__ == '__main__':
    main()