import numpy as np

from sec_agg.mpc_function import modular_inv
def sparse_gradient(grad, sparsity_ratio):
    """
    随机稀疏化梯度，保留指定比例的非零元素。
    :p grad: 梯度向量
    :sparsity_ratio: 稀疏化比例（保留的非零元素比例）
    :sparse_grad: 稀疏化后的梯度
    """
    num_non_zero = int(len(grad) * sparsity_ratio)
    indices = np.random.choice(len(grad), num_non_zero, replace=False)
    sparse_grad = np.zeros_like(grad)
    sparse_grad[indices] = grad[indices]
    return sparse_grad

def encode_gradient(sparse_grad, alpha_s, beta_s, p):
    """
    对稀疏化后的梯度进行编码，隐藏坐标信息。
    :param sparse_grad: 稀疏化后的梯度
    :param alpha_s: 用户编码点
    :param beta_s: 服务器编码点
    :param p: 有限域大小
    :return: 编码后的梯度
    """
    encoded_grad = np.zeros_like(sparse_grad)
    # 这里可以使用拉格朗日插值或其他编码方法
    # 示例：简单地对每个非零元素进行编码
    for i in range(len(sparse_grad)):
        if sparse_grad[i] != 0:
            encoded_grad[i] = (sparse_grad[i] * alpha_s[i]) % p
    return encoded_grad

def decode_gradient(encoded_grad, alpha_s, beta_s, p):
    """
    解码梯度，恢复原始稀疏化梯度。
    :param encoded_grad: 编码后的梯度
    :param alpha_s: 用户编码点
    :param beta_s: 服务器编码点
    :param p: 有限域大小
    :return: 解码后的梯度
    """
    decoded_grad = np.zeros_like(encoded_grad)
    # 这里可以使用拉格朗日插值或其他解码方法
    # 示例：简单地对每个非零元素进行解码
    for i in range(len(encoded_grad)):
        if encoded_grad[i] != 0:
            decoded_grad[i] = (encoded_grad[i] * modular_inv(alpha_s[i], p)) % p
    return decoded_grad