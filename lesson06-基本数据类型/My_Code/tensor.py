#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/22 下午6:24


"""
pytorch内的张量数据类型
-torch.FloatTensor / torch.cuda.FloatTensor
-torch.DoubleTensor / torch.cuda.DoubleTensor
-torch.IntTensor / torch.cuda.IntTensor
-torch.LongTensor / torch.cuda.LongTensor
-torch.ByteTensor / torch.cuda.ByteTensor 判断张量是否相等,0和1

pytorch内置方法
torch.randn(shape) # N(0,1)初始化一个shape形状的张量
torch.rand(shape) # U(0,1)初始化一个shape形状的张量
data.cuda() # 将data搬运到GPU上面
data.cpu() # 将data搬运到CPU上面
# 标量维度为0,加方括号可变成向量，维度为１
torch.tensor(1.) # 经常用在loss

判断张量维度
-data.shape #返回torch.Size([d1,d2,...]) 相当于list
-data.size() # 与data.shape完全相同

-data.dim() # 返回维度,int

与numpy对接
torch.from_numpy(data)
"""

import torch
import numpy as np

# 获取数据类型,数据维度,每个维度的长度,数据大小
a1 = torch.tensor(2.2)
a2 = torch.tensor([2.2])
b = torch.tensor([[2.2, 2.3], [1.1, 1.2]])
print(a1.type(), type(a2))
print(a1.shape, a2.shape)
print(b.dim(), b.size(), b.shape, b.size(0), b.shape[0])
print(b.numel()) # 输出数据大小

# 如何判断张量形状?
# 先判断有几个维度:最外端有几层就是几个维度
# 再判断每个维度的长度,维度对应括号内有几个元素,就是每个维度的长度

# 随机创建一个对应类型的张量
# 标量用在loss
c = torch.Tensor(1) # 默认为FloatTensor
print(c.type())
# 一维张量用在bias,全连接层输入
d = torch.FloatTensor([1])

# 使用numpy创建张量
e = np.ones(2)
e = torch.from_numpy(e)

#　高维张量
# 3维张量适合RNN,(Words, Sentences, Features)
f = torch.randn(1, 2, 3)
# 4维张量适合CNN,(Images, Channels, Height, Width)
g = torch.randn(2, 3, 28, 28)




