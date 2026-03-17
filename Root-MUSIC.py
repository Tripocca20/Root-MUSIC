# -*- coding: GBK -*-
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
plt.rcParams['axes.unicode_minus'] = False   

def simulate_ula_signals(num_sensors, num_snapshots, doa_deg, d=0.5, wavelength=1.0, SNR_dB=20):
    """
    生成均匀线阵（ULA）的接收信号（含噪声）
    参数说明：
        num_sensors: 阵元数量(天线数)
        num_snapshots: 快拍数(采样点数)
        doa_deg: 真实波达方向(单位：度)，列表或numpy数组
        d: 相邻阵元间距(单位：波长，默认0.5，即半波长)
        wavelength: 信号波长(默认1.0，归一化波长)
        SNR_dB: 信噪比(单位：分贝)
    
    返回值：
        Y: 接收信号矩阵(维度：num_sensors × num_snapshots)
    """
    doa_rad = np.deg2rad(doa_deg)  # 角度转弧度
    num_sources = len(doa_deg)     # 信号源数量
    sensor_idx = np.arange(num_sensors)  # 阵元索引(0,1,...,num_sensors-1)
    
    # 构造阵列流形矩阵（Array Manifold Matrix）
    # 导向矢量公式：a(θ) = [1, exp(j*2π*d*sinθ), exp(j*2π*2*d*sinθ), ...]^T
    A = np.exp(1j * 2 * np.pi * d * np.sin(doa_rad)[np.newaxis, :] * sensor_idx[:, np.newaxis] / wavelength)
    
    # 生成信号源：复高斯随机信号(归一化为单位功率)
    S = (np.random.randn(num_sources, num_snapshots) + 1j * np.random.randn(num_sources, num_snapshots)) / np.sqrt(2)
    
    # 无噪声接收信号
    Y_clean = A @ S  # 维度：num_sensors × num_snapshots
    
    # 添加高斯白噪声
    signal_power = np.mean(np.abs(Y_clean)**2)  # 计算无噪声信号的平均功率
    noise_power = signal_power / (10**(SNR_dB/10))  # 根据SNR计算噪声功率
    # 生成复高斯噪声(实虚部分开生成，保证功率匹配)
    noise = np.sqrt(noise_power/2) * (np.random.randn(num_sensors, num_snapshots) + 1j * np.random.randn(num_sensors, num_snapshots))
    
    Y = Y_clean + noise
    return Y

def root_music(R, num_sources, d=0.5, wavelength=1.0):
    """
    Root-MUSIC算法实现(仅适用于均匀线阵ULA)
    参数说明：
        R: 接收信号的样本协方差矩阵(维度：num_sensors × num_sensors)
        num_sources: 信号源数量(需要估计的DOA数量)
        d: 阵元间距(单位：波长，默认0.5)
        wavelength: 信号波长(默认1.0)
    
    返回值：
        doa_estimates_deg: 估计的波达方向(单位：度，升序排列)
    """
    num_sensors = R.shape[0]  # 阵元数量
    
    # 对协方差矩阵进行特征值分解
    eigvals, eigvecs = eigh(R)
    
    # 提取噪声子空间：选取对应最小特征值的(num_sensors - num_sources)个特征向量
    En = eigvecs[:, :num_sensors - num_sources]
    
    # 构造噪声子空间投影矩阵
    Pn = En @ En.conj().T  # 维度：num_sensors × num_sensors
    
    # 利用Toeplitz结构提取多项式系数：
    # 对于ULA，Pn的每条对角线元素理论上相等，
    # 对每条对角线求和得到系数c[k](k范围：-(M-1) 到 M-1，M为阵元数)
    c = np.array([np.sum(np.diag(Pn, k)) for k in range(-num_sensors+1, num_sensors)])
    
    # 归一化：将k=0(主对角线)的系数设为1，不改变根的位置
    c = c / c[num_sensors - 1]
    
    # 构造多项式系数：注意np.roots要求系数按降幂排列，因此反转数组
    poly_coeffs = c[::-1]
    
    # 求解多项式所有根
    roots_all = np.roots(poly_coeffs)
    
    # 仅保留单位圆内的根(理论上信号相关根应靠近单位圆)
    roots_inside = roots_all[np.abs(roots_all) < 1]
    
    # 按根到单位圆的距离排序，选取距离最近的num_sources个根(信号根)
    distances = np.abs(np.abs(roots_inside) - 1)  # 计算根到单位圆的距离
    sorted_indices = np.argsort(distances)        # 按距离升序排序
    selected_roots = roots_inside[sorted_indices][:num_sources]  # 选取前num_sources个根
    
    # 根的相位满足：angle(z) = -2π*d*sin(θ)/wavelength
    # 计算beta = 2π*d/wavelength
    beta = 2 * np.pi * d / wavelength
    phi = np.angle(selected_roots)  # 提取根的相位(弧度)
    
    # 将根的相位映射为角度(弧度转角度)
    doa_estimates_rad = np.arcsin(phi / beta)
    doa_estimates_deg = np.rad2deg(doa_estimates_rad)
    
    return np.sort(doa_estimates_deg), roots_all

if __name__ == "__main__":
    # 仿真参数配置
    num_sensors = 8           # 阵列阵元数量
    num_snapshots = 1000      # 快拍数(建议增大以提升协方差矩阵估计精度)
    doa_true = [-20, 20, 30]   # 真实波达方向(度)
    SNR_dB = 20               # 信噪比(分贝)
    d = 0.5                   # 阵元间距(波长)
    wavelength = 1.0          # 信号波长(归一化)
    
    # 生成仿真接收信号(含指定方向的信号)
    Y = simulate_ula_signals(num_sensors, num_snapshots, doa_true, d, wavelength, SNR_dB)
    
    # 估计样本协方差矩阵
    R = Y @ Y.conj().T / num_snapshots
    
    # 调用Root-MUSIC算法估计DOA
    num_sources = len(doa_true)
    doa_est, roots_all = root_music(R, num_sources, d, wavelength)
    
    # 打印结果
    print("真实DOA(度)：", doa_true)
    print("估计DOA(度)：", doa_est)
    
    # 绘制多项式根的分布
    plt.figure(figsize=(6,6))
    theta = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='单位圆')
    plt.scatter(np.real(roots_all), np.imag(roots_all), marker='o', color='b', label='多项式根')
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title('Root-MUSIC 根分布')
    plt.axis('equal')  # 保证单位圆为正圆
    plt.legend()
    plt.grid(True)
    plt.show()