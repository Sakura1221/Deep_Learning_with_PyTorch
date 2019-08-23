#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/23 下午4:22

import torch


# add / sub / mul / div
a = torch.rand(3, 4)
b = torch.rand(4)
# add
c = a + b # torch.add(a,b),对应元素相加
# sub
d = a - b # torch.sub(a,b),对应元素相减
# mul
e = a * b # torch.mul(a,b),对应元素相乘
# div
f = a / b # torch.div(a,b),对应元素相除(//整除)


# matmul矩阵相乘,非元素相乘*
# torch.mm只适用于2d,torch.matmul通用,或者使用@
# 高维矩阵乘法,还是只取最后两个维度做乘法运算(多个矩阵并行相乘)
# 前面维度要相同(或者经过broadcast后相同),后面维度要匹配
g = torch.rand(4, 784)
h = torch.rand(512, 784) # (channel_out,channel_in)
# 需要转置,维度匹配才能相乘,2维.t(),高维.transpose()
i = g @ h.t() # i = torch.matmul(g,h.t()) -> (4, 512)


# pow / sqrt
j = torch.full([2,2],3)
# pow
k = j ** 2 # k = j.pow(2),每个元素乘方
# sqrt
l = k ** 0.5 # l = k.sqrt() # 每个元素的平方根



# exp / log
# exp
m = torch.exp(torch.ones(2,2)) # 每个元素求幂指数
n = torch.log(m) # 每个元素求自然对数,还有log2,log10


# approximation
o = torch.tensor(3.14)
# round() / floor() / ceil() / trunc() / frac()
# 四舍五入,舍尾法,进一法,整数部分,小数部分
print(o.round(), o.floor(), o.ceil(), o.trunc(), o.frac())


# 数据限制clamp
# 经常会出现梯度爆炸,可以查看w的L2范数 w.grad.norm(2),正常在10左右
grad = torch.rand(2,3) * 15
print(grad)
# 可以查看张量最值与均值
print(grad.max(), grad.min(), grad.median())
# clamp可以限制数据的范围
grad_max = grad.clamp(10) # 限制最小值为10
grad_min = grad.clamp(-float('inf'),10) # 限制最大值为10
grad_mid = grad.clamp(5,10) # 限制最小值为5,最大值为10
print(grad_max, grad_min, grad_mid, sep='\n')