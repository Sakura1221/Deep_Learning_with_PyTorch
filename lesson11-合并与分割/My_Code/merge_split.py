#-*-coding:utf-8-*-
# Author : Zhang Zhichaung
# Date : 2019/8/23 下午3:48

import torch


# cat / stack

# cat是合并两个张量,增加一个维度的长度
a = torch.rand(4, 3, 16, 32)
b = torch.rand(5, 3, 16, 32)
# 相加维度外的其他维度要相同
# 在0维相加,最终张量维度0维相加
c = torch.cat([a, b], dim=0) # (9, 3, 16, 32)

# stack是两个张量并排放,增加一个维度,增加了一个概念
# stack要求两个张量形状完全一致
# 在指定维度前增加维度
d = torch.rand(4, 3, 16, 32)
e = torch.stack([a, d], dim=2) # (4, 3, 2, 16, 32)

# split / chunk
# split将指定维度拆分成对应长度
f, g = e.split([10,6], dim=3)
print(f.shape, g.shape) # (4, 3, 2, 10, 32),(4, 3, 2, 6, 32)
h, i = e.split(1,dim=2) # 长度相同可直接传入长度
print(h.shape, i.shape) # (4, 3, 1, 16, 32),(4, 3, 1, 16, 32)

# chunk将指定维度等分成指定数量
j, k = e.chunk(2,dim=2)
print(j.shape, k.shape) # (4, 3, 1, 16, 32),(4, 3, 1, 16, 32)