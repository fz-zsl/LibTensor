import numpy as np

# 定义张量类
class Tensor:
    def __init__(self, shape, data):
        self.shape = shape
        self.data = data

# 创建张量 t1
t1 = Tensor((3, [2, 2, 2]), np.array([1, 2, 3, 4, 5, 6, 7, 8]))

# 创建张量 t2
t2 = Tensor((3, [2, 2, 2]), np.array([1, 2, 3, 4, 5, 6, 7, 8]))

# 检查张量形状是否兼容
if t1.shape != t2.shape:
    raise ValueError("张量形状不兼容，无法相乘")

# 计算张量相乘
result = np.multiply(t1.data, t2.data)

print("相乘结果：")
print(result)