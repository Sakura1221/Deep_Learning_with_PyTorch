#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/21 下午7:13

"""
GPU加速运算
"""

import 	torch
import  time

#　torch.randn(row,col): 生成N(0,１)
a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

t0 = time.time()
#　torch.matmul(a,b): 矩阵乘法
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))

# 默认运算设备是cpu,torch.device('cuda')可以设置gpu
device = torch.device('cuda')
# a任然存储在内存里,需要手动存储到显存里
a = a.to(device)
b = b.to(device)

#　第一次启动cuda速度需要初始化,速度较慢
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
# Tensor.norm(2),求矩阵的二阶范数
print(a.device, t2 - t0, c.norm(2))

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

