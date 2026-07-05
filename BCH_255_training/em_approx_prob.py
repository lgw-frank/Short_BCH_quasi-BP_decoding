import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd = tfp.distributions

def em_with_tfp(y, max_iter=100):
    """
    使用tensorflow-probability的EM实现
    """
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    
    # 初始化
    n = tf.shape(y)[0]
    y_sorted = tf.sort(y)
    split = n // 2
    
    mu1 = tf.reduce_mean(y_sorted[:split])
    mu2 = tf.reduce_mean(y_sorted[split:])
    if mu1 < mu2:
        mu1, mu2 = mu2, mu1
    
    sigma = tf.math.reduce_std(y)
    
    for i in range(max_iter):
        # E-step: calculate responsibilities
        dist1 = tfd.Normal(loc=mu1, scale=sigma)
        dist2 = tfd.Normal(loc=mu2, scale=sigma)
        
        prob1 = dist1.prob(y)
        prob2 = dist2.prob(y)
        
        gamma1 = prob1 / (prob1 + prob2 + 1e-10)
        gamma2 = prob2 / (prob1 + prob2 + 1e-10)
        
        # M-step: update parameters
        n1 = tf.reduce_sum(gamma1)
        n2 = tf.reduce_sum(gamma2)
        
        mu1_new = tf.reduce_sum(gamma1 * y) / n1
        mu2_new = tf.reduce_sum(gamma2 * y) / n2
        
        sigma_new = tf.sqrt((tf.reduce_sum(gamma1 * (y - mu1_new)**2) + 
                            tf.reduce_sum(gamma2 * (y - mu2_new)**2)) / tf.cast(n, tf.float32))
        
        # check convergence
        if (tf.abs(mu1 - mu1_new) < 1e-6 and 
            tf.abs(mu2 - mu2_new) < 1e-6 and
            tf.abs(sigma - sigma_new) < 1e-6):
            break
            
        mu1, mu2, sigma = mu1_new, mu2_new, sigma_new
    
    return {
        'mu1': mu1.numpy(),
        'mu2': mu2.numpy(),
        'sigma': sigma.numpy(),
        'sigma2': sigma.numpy()**2
    }
class BPSK_EM_Estimator:
    def __init__(self, max_iter=100, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, y, verbose=True):
        """
        使用EM算法估计高斯混合参数
        y: 观测到的接收信号 (numpy array 或 tensor)
        """
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        n = tf.cast(tf.shape(y)[0], tf.float32)
        
        # 初始化参数
        # 方法1：使用K-means风格初始化
        y_sorted = tf.sort(y)
        split_idx = tf.cast(n/2, tf.int32)
        mu1_init = tf.reduce_mean(y_sorted[:split_idx])
        mu2_init = tf.reduce_mean(y_sorted[split_idx:])
        
        # 确保mu1 > mu2（对应+1和-1）
        if mu1_init < mu2_init:
            mu1_init, mu2_init = mu2_init, mu1_init
            
        # 初始化参数
        mu1 = tf.Variable(mu1_init, dtype=tf.float32)
        mu2 = tf.Variable(mu2_init, dtype=tf.float32)
        sigma2 = tf.Variable(tf.reduce_mean((y - tf.reduce_mean(y))**2), dtype=tf.float32)
        
        # 混合权重固定为0.5（等概）
        pi = 0.5
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            with tf.GradientTape() as tape:
                # E-Step: 计算后验概率（责任）
                # 计算每个点属于两个分布的概率密度
                prob1 = pi * tf.exp(-(y - mu1)**2 / (2 * sigma2)) / tf.sqrt(2 * np.pi * sigma2)
                prob2 = (1-pi) * tf.exp(-(y - mu2)**2 / (2 * sigma2)) / tf.sqrt(2 * np.pi * sigma2)
                
                # 归一化得到后验概率
                gamma1 = prob1 / (prob1 + prob2 + 1e-10)  # 属于分布1的概率
                gamma2 = prob2 / (prob1 + prob2 + 1e-10)  # 属于分布2的概率
                
                # 计算对数似然（用于收敛判断）
                log_likelihood = tf.reduce_sum(tf.math.log(prob1 + prob2 + 1e-10))
                
                # M-Step: 更新参数
                n1 = tf.reduce_sum(gamma1)
                n2 = tf.reduce_sum(gamma2)
                
                # 更新均值
                new_mu1 = tf.reduce_sum(gamma1 * y) / (n1 + 1e-10)
                new_mu2 = tf.reduce_sum(gamma2 * y) / (n2 + 1e-10)
                
                # 更新方差
                new_sigma2 = (tf.reduce_sum(gamma1 * (y - new_mu1)**2) + 
                             tf.reduce_sum(gamma2 * (y - new_mu2)**2)) / n
                
            # 应用更新
            mu1.assign(new_mu1)
            mu2.assign(new_mu2)
            sigma2.assign(new_sigma2)
            
            # 检查收敛
            if iteration > 0:
                if tf.abs(log_likelihood - prev_log_likelihood) < self.tol:
                    if verbose:
                        print(f"收敛于第{iteration}次迭代")
                    break
                    
            prev_log_likelihood = log_likelihood
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: mu1={mu1.numpy():.4f}, mu2={mu2.numpy():.4f}, "
                      f"sigma2={sigma2.numpy():.4f}, logL={log_likelihood.numpy():.2f}")
        
        # 返回估计结果（确保mu1对应+1, mu2对应-1）
        result = {
            'mu1': mu1.numpy(),  # 应该接近+1
            'mu2': mu2.numpy(),  # 应该接近-1
            'sigma2': sigma2.numpy(),  # 估计的噪声方差
            'sigma': np.sqrt(sigma2.numpy()),
            'snr_est': 1.0 / sigma2.numpy(),  # 估计的信噪比
            'iterations': iteration
        }
        
        return result

# 生成测试数据
np.random.seed(42)
sigma2_true = 0.5  # 真实噪声方差
sigma_true = np.sqrt(sigma2_true)
n_samples = 10000

# 生成BPSK信号
bits = np.random.randint(0, 2, n_samples)
x = 2 * bits - 1  # +1 或 -1
noise = sigma_true * np.random.randn(n_samples)
y = x + noise

print("=== 真实参数 ===")
print(f"mu1 = +1.0")
print(f"mu2 = -1.0")
print(f"sigma2 = {sigma2_true}")
print(f"SNR = {1/sigma2_true:.2f} (线性)")

# 使用EM算法估计
em = BPSK_EM_Estimator(max_iter=100)
result = em.fit(y)

print("\n=== EM估计结果 ===")
print(f"mu1估计 = {result['mu1']:.4f} (理论: +1.0)")
print(f"mu2估计 = {result['mu2']:.4f} (理论: -1.0)")
print(f"sigma2估计 = {result['sigma2']:.4f} (理论: {sigma2_true})")
print(f"SNR估计 = {result['snr_est']:.4f} (理论: {1/sigma2_true:.4f})")

# 验证LLR的方差关系
# 使用估计的参数计算LLR
LLR_estimated = (2 / result['sigma2']) * y

# 条件统计（需要知道真实比特来验证，但在实际中我们不知道）
# 这里只是为了验证EM估计的准确性
LLR_given_plus1_true = LLR_estimated[x == 1]
LLR_given_minus1_true = LLR_estimated[x == -1]

print("\n=== LLR统计验证（使用估计的sigma2）===")
print(f"给定x=+1: E[LLR] = {np.mean(LLR_given_plus1_true):.4f}")
print(f"给定x=+1: Var[LLR] = {np.var(LLR_given_plus1_true):.4f}")
print(f"比值 = {np.var(LLR_given_plus1_true)/np.mean(LLR_given_plus1_true):.4f} (理论: 2.0)")


def visualize_em_convergence(y, true_sigma2):
    """
    可视化EM算法的收敛过程
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 初始直方图
    axes[0, 0].hist(y, bins=50, density=True, alpha=0.7, label='观测数据')
    axes[0, 0].set_title('观测数据分布')
    axes[0, 0].set_xlabel('y')
    axes[0, 0].set_ylabel('密度')
    
    # EM估计过程
    em = BPSK_EM_Estimator(max_iter=50)
    
    # 手动迭代以记录过程
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    n = tf.cast(tf.shape(y_tf)[0], tf.float32)
    
    # 初始化
    y_sorted = tf.sort(y_tf)
    split_idx = tf.cast(n/2, tf.int32)
    mu1 = tf.reduce_mean(y_sorted[:split_idx])
    mu2 = tf.reduce_mean(y_sorted[split_idx:])
    if mu1 < mu2:
        mu1, mu2 = mu2, mu1
    sigma2 = tf.reduce_mean((y_tf - tf.reduce_mean(y_tf))**2)
    
    history = {'mu1': [mu1.numpy()], 'mu2': [mu2.numpy()], 'sigma2': [sigma2.numpy()]}
    
    for iteration in range(50):
        # E-step
        prob1 = 0.5 * tf.exp(-(y_tf - mu1)**2 / (2 * sigma2)) / tf.sqrt(2 * np.pi * sigma2)
        prob2 = 0.5 * tf.exp(-(y_tf - mu2)**2 / (2 * sigma2)) / tf.sqrt(2 * np.pi * sigma2)
        gamma1 = prob1 / (prob1 + prob2 + 1e-10)
        
        # M-step
        n1 = tf.reduce_sum(gamma1)
        new_mu1 = tf.reduce_sum(gamma1 * y_tf) / (n1 + 1e-10)
        new_mu2 = (tf.reduce_sum((1-gamma1) * y_tf) / (n - n1 + 1e-10))
        new_sigma2 = (tf.reduce_sum(gamma1 * (y_tf - new_mu1)**2) + 
                     tf.reduce_sum((1-gamma1) * (y_tf - new_mu2)**2)) / n
        
        mu1, mu2, sigma2 = new_mu1, new_mu2, new_sigma2
        
        history['mu1'].append(mu1.numpy())
        history['mu2'].append(mu2.numpy())
        history['sigma2'].append(sigma2.numpy())
        
        if iteration % 5 == 0:
            # 绘制中间步骤的拟合分布
            ax = axes[(iteration//5) % 2, 1] if iteration < 25 else axes[1, 0]
            if iteration in [0, 5, 10, 20, 30, 40]:
                x_range = np.linspace(-3, 3, 200)
                fitted_dist = (0.5 * np.exp(-(x_range - mu1.numpy())**2/(2*sigma2.numpy()))/np.sqrt(2*np.pi*sigma2.numpy()) +
                              0.5 * np.exp(-(x_range - mu2.numpy())**2/(2*sigma2.numpy()))/np.sqrt(2*np.pi*sigma2.numpy()))
                ax.plot(x_range, fitted_dist, label=f'Iter {iteration}')
                ax.hist(y, bins=50, density=True, alpha=0.3)
                ax.set_title(f'EM迭代 {iteration}')
                ax.legend()
    
    # 绘制收敛曲线
    axes[1, 1].plot(history['mu1'], label='μ1 (估计)')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='μ1 (真实)')
    axes[1, 1].plot(history['mu2'], label='μ2 (估计)')
    axes[1, 1].axhline(y=-1.0, color='g', linestyle='--', label='μ2 (真实)')
    axes[1, 1].set_title('均值收敛过程')
    axes[1, 1].set_xlabel('迭代次数')
    axes[1, 1].set_ylabel('均值')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 添加方差收敛子图
    ax_var = axes[1, 1].twinx()
    ax_var.plot(history['sigma2'], color='purple', label='σ² (估计)', alpha=0.5)
    ax_var.axhline(y=true_sigma2, color='orange', linestyle='--', label='σ² (真实)', alpha=0.5)
    ax_var.set_ylabel('方差', color='purple')
    ax_var.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return history

# 运行可视化
history = visualize_em_convergence(y, sigma2_true)