#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/22 下午7:26

"""
张量的创建
"""

import numpy as np
import torch
import math

a = np.array([2, 3.3])

# tensor([2.0000, 3.3000], dtype=torch.float64)
torch.from_numpy(a)

# tensor传data,Tensor传shape
b = torch.tensor([2., 3.2])

# 生成未初始化数据
c = torch.FloatTensor(2, 3)
# c = torch.empty([2, 3]) 不推荐

# 生成随机初始化数据
d = torch.rand(3, 3) # U(0, 1)
e = torch.rand_like(d) # 传入tensor,调用对应函数生成相同形状的一组数据
f = torch.randint(1, 10, [5, 5]) # [1, 10)整数
g = torch.randn(0,1) # N(0, 1) #　常用在bias和weight初始化
# 每个数据可变均值和方差的正态分布
# arange(start,stop,step),[start,stop)
# torch.full(shape, number)
# 注意区别torch.full([],7)与torch.full([1],7)
h = torch.normal(mean=torch.full([10],0),
                 std=torch.arange(1, 0, -0.1))


# 生成序列化数据
i = torch.arange(0, 10, 2) # arange(start,stop,step),[start,stop)
j = torch.linspace(0, 10, 4) # linspace(start,stop,num),[start,stop]
# linspce(start_exp, stop_exp, num),[base^start, base^stop]
k = torch.logspace(0, -1, steps=8, base=math.e)
# torch.ones() / torch.zeros() / torch.eye()
l = torch.ones(3, 3)
m = torch.ones_like(j)
n = torch.zeros(3, 3)
o = torch.eye(3, 4)
p = torch.eye(3)

# 生成随机种子(索引),用来采样(包含所有内容,顺序打乱),对应shuffle
q = torch.randperm(10) # [0,10)
data1 = torch.rand(2, 3)
data2 = torch.rand(2, 2)
idx = torch.randperm(2)
# 按照索引(list)取值,使用相同idx,保证数据对应
print(data1[idx], data2[idx])