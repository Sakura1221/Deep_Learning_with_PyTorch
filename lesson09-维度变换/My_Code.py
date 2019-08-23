#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/23 上午11:21


import numpy as np
import torch

a = torch.rand(4,1,28,28)

# 使用view() / reshape()进行维度变换
# 维度变换要保证numel()一致
# 维度要满足物理意义
b = a.view(4, 1*28*28) # 将所有图像信息整合为一维,全连接层
c = a.view(4*28, 28) # 单元为图像每一行的数据
d = a.view(4*1, 28, 28) # 单元为一张图片的数据

# 使用squeeze() / unsqueeze()增加/减少维度,不增加数据
e = a.unsqueeze(0) # 在0维之前插入一个维度
f = a.unsqueeze(-1) # 在最后一维之后插入一个维度
g = a.squeeze() # 删除a所有长度为1的维度
h = a.squeeze(2) # 删除a的第1维,如果不为1则保持不变

# expand() / repeat() 扩展每个维度的长度,增加数据
# 两者效果等效,expand()只有在需要时执行扩展,速度快占用小,建议使用
# expand()传入要扩展成的维度,repeat()传入复制次数
# 扩展要求维度相同
# 只能扩展长度为1的维度,不为1保持不变
i = torch.rand(1,1,28,1)
j = i.expand(4,1,28,28) # 扩展成与a相同的维度,方便进行某些操作
k = i.expand(-1,-1,28,28) # -1表示该维度保持不变,只扩展其他维度
l = i.repeat(4,1,28,28) # [4,1,784,28]

# .t / transpose() / permute()
m = torch.randn(3,4)
n = m.t() # 只适用二维
o = a.transpose(1,3) # 交换1维和3维
# 交换维度操作会使数据存储不连续,view()要求数据连续,因此需要contiguous()
p = (a.transpose(1,3).contiguous().view(4,1*28*28).
     view(4,28,28,1).transpose(1,3)) # 与a一致
# print(torch.all(torch.eq(a, p))) # Tensor(True)
q = a.permute(0,2,3,1) # 重新安排各维度的顺序

