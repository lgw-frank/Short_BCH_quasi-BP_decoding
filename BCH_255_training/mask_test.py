import tensorflow as tf
from tensorflow.keras import layers, Model

class MaskedModel(Model):
    def __init__(self):
        super(MaskedModel, self).__init__()
        # Masking层：标记值为0的位置为mask，后续层会忽略它们
        self.masking = layers.Masking(mask_value=0.0)
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(1)  # 输出标量
    
    def call(self, inputs):
        # inputs shape: (batch_size, 4)
        x = self.masking(inputs)  # 标记0为mask
        x = self.dense1(x)
        # 注意：Dense层会自动忽略被mask的位置
        # 但需要聚合被mask处理后的结果
        # 方法1：使用全局平均池化聚合
        x = tf.reduce_mean(x, axis=1)  # 对特征维度求平均
        return self.dense2(x)

# 创建模型
model = MaskedModel()

# 输入数据
x = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
n = 4

# 生成掩码输入
mask = 1.0 - tf.eye(n, dtype=tf.float32)
masked_inputs = tf.tile(tf.expand_dims(x, 0), [n, 1]) * mask
print(masked_inputs)
# 批量预测
outputs = model(masked_inputs)  # shape: (4, 1)
output_vector = tf.squeeze(outputs)  # shape: (4,)

print("输出矢量:", output_vector.numpy())